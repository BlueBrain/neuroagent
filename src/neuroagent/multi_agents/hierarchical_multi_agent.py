import functools
import operator
import logging
from typing import (Annotated, Any, AsyncIterator, Hashable, List, Sequence,
                    TypedDict, Dict)
from contextlib import AsyncExitStack

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.output_parsers.openai_functions import \
    JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import ConfigDict, model_validator, Field
from langgraph.checkpoint.base import BaseCheckpointSaver

from neuroagent.agents import AgentOutput
from neuroagent.multi_agents.base_multi_agent import BaseMultiAgent
from neuroagent.tools.base_tool import BasicTool

logger = logging.getLogger(__file__)

class HierarchicalTeamAgent(BaseMultiAgent):
    """Hierarchical Team Agent managing multiple teams."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tools: Dict[str, BasicTool] = Field(default_factory=list)
    memory: BaseCheckpointSaver[Any] | None = None
    top_level_chain: Any = None
    trimmer: Any = None

    def __init__(self, 
                 llm: Any,
                 tools: Dict[str, BasicTool], 
                 agents: list[tuple[str, list[BasicTool]]] = None,
                 memory: BaseCheckpointSaver[Any] = None):
        super().__init__(llm=llm, agents=agents)
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.trimmer = trim_messages(
            max_tokens=100000,
            strategy="last",
            token_counter=self.llm,
            include_system=True,
        )
        self.top_level_chain = self.create_graph()

    @staticmethod
    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        return {
            "messages": [AIMessage(content=result["messages"][-1].content, name=name)]
        }

    def create_graph(self) -> CompiledGraph:
        # Compose subgraphs and return the compiled top-level chain
        simulation_chain = self.create_simulation_team()
        analysis_chain = self.create_analysis_team()

        # Define helper functions
        def get_last_message(state):
            return state["messages"][-1].content

        def join_graph(response):
            return {"messages": [response["messages"][-1]]}

        # Top-level state
        class TopLevelState(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            next: str

        # Top-Level Supervisor Agent
        top_level_supervisor_agent = self.create_team_supervisor(
            "You are the top-level supervisor managing all teams. Teams:"
            " SimulationTeam, AnalysisTeam.",
            ["SimulationTeam", "AnalysisTeam"],
        )

        # Define the top-level graph
        top_level_graph = StateGraph(TopLevelState)
        top_level_graph.add_node(
            "SimulationTeam", get_last_message | simulation_chain | join_graph
        )
        top_level_graph.add_node(
            "AnalysisTeam", get_last_message | analysis_chain | join_graph
        )
        top_level_graph.add_node("supervisor", top_level_supervisor_agent)

        # Define edges
        top_level_graph.add_edge("SimulationTeam", "supervisor")
        top_level_graph.add_edge("AnalysisTeam", "supervisor")
        top_level_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "SimulationTeam": "SimulationTeam",
                "AnalysisTeam": "AnalysisTeam",
                "FINISH": END,
            },
        )
        top_level_graph.add_edge(START, "supervisor")
        top_level_chain = top_level_graph.compile()

        return top_level_chain

    def create_simulation_team(self) -> CompiledGraph:
        class SimulationTeamState(TypedDict):
            # A message is added after each team member finishes
            messages: Annotated[List[BaseMessage], operator.add]
            # The team members are tracked so they are aware of
            # the others' skill-sets
            team_members: List[str]
            # Used to route work. The supervisor calls a function
            # that will update this every time it makes a decision
            next: str
        # Define tools
        simulation_tools = [
            self.tools["br_resolver_tool"],
            self.tools["morpho_tool"],
            self.tools["morphology_feature_tool"],
            self.tools["kg_morpho_feature_tool"],
            self.tools["literature_tool"],
            self.tools["me_model_tool"],
            self.tools["electrophys_feature_tool"],
            self.tools["traces_tool"],
            self.tools["me_model_tool"],
            self.tools["bluenaas_tool"],
        ]  # Add other bluenaas endpoints later on..

        # Create agents
        simulation_agent = create_react_agent(self.llm, tools=simulation_tools, checkpointer=self.memory) 

        simulation_node = functools.partial(
            self.agent_node, agent=simulation_agent, name="SimulationAgent"
        ) # might need to rename to SingleCellSimAgent when circuit level tools come

        # Supervisor
        simulation_supervisor_agent = self.create_team_supervisor(
            "You are a supervisor managing the Simulation Team. Members:"
            " SimulationAgent."  # add synapse generator & literature search agents later on
            " Given the following user request, respond with the worker to act next."
            " Each worker will perform a task and respond with their results and status."
            " When finished, respond with FINISH.",
            ["SimulationAgent"],
        )

        # Create graph
        simulation_graph = StateGraph(SimulationTeamState)
        simulation_graph.add_node("SimulationAgent", simulation_node)
        simulation_graph.add_node("supervisor", simulation_supervisor_agent)
        # TODO: add web search and web scraper nodes for best performance

        # Define edges
        simulation_graph.add_edge("SimulationAgent", "supervisor")
        simulation_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"SimulationAgent": "SimulationAgent", "FINISH": END},
        )  # add other edges as team grows
        simulation_graph.add_edge(START, "supervisor")
        chain = simulation_graph.compile()

        # The following functions interoperate between the top level graph state
        # and the state of the research sub-graph
        # this makes it so that the states of each graph don't get intermixed
        def enter_chain(message: str):
            results = {
                "messages": [HumanMessage(content=message)],
            }
            return results
        
        simulation_chain = enter_chain | chain
        
        return simulation_chain

    def create_analysis_team(self) -> CompiledGraph:
        class AnalysisTeamState(TypedDict):
            # A message is added after each team member finishes
            messages: Annotated[List[BaseMessage], operator.add]
            # The team members are tracked so they are aware of
            # the others' skill-sets
            team_members: List[str]
            # Used to route work. The supervisor calls a function
            # that will update this every time it makes a decision
            next: str

        # Define tools
        morphology_tools = [
            self.tools["br_resolver_tool"],
            self.tools["morpho_tool"],
            self.tools["morphology_feature_tool"],
            self.tools["kg_morpho_feature_tool"],
            self.tools["literature_tool"],
            self.tools["me_model_tool"],
        ]
        electrophysiology_tools = [
            self.tools["electrophys_feature_tool"],
            self.tools["traces_tool"],
            self.tools["me_model_tool"],
        ]  # Replace with your actual tools

        # Create agents
        morphology_agent = create_react_agent(self.llm, tools=morphology_tools, checkpointer=self.memory)
        morphology_node = functools.partial(
            self.agent_node, agent=morphology_agent, name="MorphologyAgent"
        )

        electrophysiology_agent = create_react_agent(
            self.llm, tools=electrophysiology_tools, checkpointer=self.memory
        )
        electrophysiology_node = functools.partial(
            self.agent_node, agent=electrophysiology_agent, name="ElectrophysiologyAgent"
        )

        # Supervisor
        analysis_supervisor_agent = self.create_team_supervisor(
            "You are a supervisor managing the Analysis Team. Members: MorphologyAgent,"
            " ElectrophysiologyAgent.",
            ["MorphologyAgent", "ElectrophysiologyAgent"],
        )

        # Create graph
        analysis_graph = StateGraph(AnalysisTeamState)
        analysis_graph.add_node("MorphologyAgent", morphology_node)
        analysis_graph.add_node("ElectrophysiologyAgent", electrophysiology_node)
        analysis_graph.add_node("supervisor", analysis_supervisor_agent)

        # Define edges
        analysis_graph.add_edge("MorphologyAgent", "supervisor")
        analysis_graph.add_edge("ElectrophysiologyAgent", "supervisor")
        analysis_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "MorphologyAgent": "MorphologyAgent",
                "ElectrophysiologyAgent": "ElectrophysiologyAgent",
                "FINISH": END,
            },
        )
        analysis_graph.add_edge(START, "supervisor")
        analysis_chain = analysis_graph.compile()

        return analysis_chain

    # @model_validator(mode="before")
    def create_team_supervisor(self, system_prompt: str, members: List[str]):
        """Create main supervisor across teams."""
        logger.info("Creating main supervisor, supervisor and all the agents with tools.")
        options = ["FINISH"] + members
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [
                            {"enum": options},
                        ],
                    },
                },
                "required": ["next"],
            },
        }
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                (
                    "Given the conversation above, who should act next?"
                    " Or should we FINISH? Select one of: {options}"
                ),
            ),
        ]).partial(options=str(options), team_members=", ".join(members))
        
        chain = (
            RunnablePassthrough()
            | prompt
            | self.trimmer
            | self.llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )
        
        return chain


    def run(self, query: str, thread_id: str) -> AgentOutput:
        res = self.top_level_chain.invoke(
            input={"messages": [HumanMessage(content=query)]},
            config=RunnableConfig(configurable={"thread_id": thread_id}),
        )
        return self._process_output(res)

    async def arun(self, query: str, thread_id: str) -> AgentOutput:
        res = await self.top_level_chain.ainvoke(
            input={"messages": [HumanMessage(content=query)]},
            config=RunnableConfig(configurable={"thread_id": thread_id}),
        )
        return self._process_output(res)

    async def astream(
        self, thread_id: str, query: str, connection_string: str | None = None
    ) -> AsyncIterator[str]:
        """Run the hierarchical team agent against a query in a streaming way.

        Parameters
        ----------
        thread_id : str
            ID of the thread of the chat.
        query : str
            Query of the user.
        connection_string : str | None, optional
            Connection string for the checkpoint database.

        Yields
        ------
        str
            Iterator streaming the processed output of the LLM and all agents.
        """
        async with (
            self.memory.__class__.from_conn_string(connection_string)
            if connection_string and self.memory
            else AsyncExitStack()
        ) as memory:
            if isinstance(memory, BaseCheckpointSaver):
                self.memory = memory
            
            config = RunnableConfig(configurable={"thread_id": thread_id})
            
            async for event in self.top_level_chain.astream_events(
                {"messages": [HumanMessage(content=query)]},
                config=config,
                version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    yield event["data"]
                elif event["event"] == "on_agent_action":
                    yield f"Agent {event['name']} is taking action: {event['data']['tool']}"
                elif event["event"] == "on_agent_finish":
                    yield f"Agent {event['name']} finished: {event['data']['return_values']['output']}"

    @staticmethod
    def _process_output(output: Any) -> AgentOutput:
        """Format the output."""
        agent_steps = []
        for message in output["messages"][1:]:
            if "steps" in message.additional_kwargs:
                agent_steps.extend(message.additional_kwargs["steps"])
        return AgentOutput(response=output["messages"][-1].content, steps=agent_steps)
