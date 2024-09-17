"""Supervisor multi-agent."""

import functools
import logging
import operator
from typing import Annotated, Any, AsyncIterator, Hashable, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import ConfigDict, model_validator

from neuroagent.agents import AgentOutput, AgentStep
from neuroagent.multi_agents.base_multi_agent import BaseMultiAgent

logger = logging.getLogger(__file__)


class AgentState(TypedDict):
    """Base class for agent state."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str  # noqa: A003


class SupervisorMultiAgent(BaseMultiAgent):
    """Base class for multi agents."""

    summarizer: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def create_main_agent(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Instantiate the clients upon class creation."""
        logger.info("Creating main agent, supervisor and all the agents with tools.")
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            " following workers: {members}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH."
        )
        agents_list = [elem[0] for elem in data["agents"]]
        logger.info(f"List of agents name: {agents_list}")

        options = ["FINISH"] + agents_list
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
                    }
                },
                "required": ["next"],
            },
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    (
                        "Given the conversation above, who should act next?"
                        " Or should we FINISH? Select one of: {options}"
                    ),
                ),
            ]
        ).partial(options=str(options), members=", ".join(agents_list))
        data["main_agent"] = (
            prompt
            | data["llm"].bind_functions(
                functions=[function_def], function_call="route"
            )
            | JsonOutputFunctionsParser()
        )
        data["summarizer"] = (
            PromptTemplate.from_template(
                """You are an helpful assistant. Here is the question of the user: {question}.
            And here are the results of the different tools used to answer: {responses}.
            You must always specify in your answers from which brain regions the information is extracted.
            Do no blindly repeat the brain region requested by the user, use the output of the tools instead.
            Keep all information that can be useful for the user such as the ids and links.
            Please formulate a complete response to give to the user ONLY based on the results.
            """
            )
            | data["llm"]
        )

        return data

    @staticmethod
    async def agent_node(
        state: AgentState, agent: CompiledGraph, name: str
    ) -> dict[str, Any]:
        """Run the agent node."""
        logger.info(f"Start running the agent: {name}")
        result = await agent.ainvoke(state)

        agent_steps = [
            AgentStep(
                tool_name=step.tool_calls[0]["name"],
                arguments=step.tool_calls[0]["args"],
            )
            for step in result["messages"]
            if isinstance(step, AIMessage) and step.additional_kwargs
        ]

        return {
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content,
                    name=name,
                    additional_kwargs={"steps": agent_steps},
                )
            ]
        }

    async def summarizer_node(self, state: AgentState) -> dict[str, Any]:
        """Create summarizer node."""
        logger.info("Entering the summarizer node")
        question = state["messages"][0].content
        responses = " \n".join([mes.content for mes in state["messages"][1:]])  # type: ignore
        result = await self.summarizer.ainvoke(
            {"question": question, "responses": responses}
        )
        return {
            "messages": [
                HumanMessage(
                    content=result.content,
                    name="summarizer",
                )
            ]
        }

    def create_graph(self) -> CompiledGraph:
        """Create graph."""
        workflow = StateGraph(AgentState)

        # Create nodes
        for agent_name, tools_list in self.agents:
            agent = create_react_agent(model=self.llm, tools=tools_list)
            node = functools.partial(self.agent_node, agent=agent, name=agent_name)
            workflow.add_node(agent_name, node)

        # Supervisor node
        workflow.add_node("Supervisor", self.main_agent)

        # Summarizer node
        summarizer_agent = functools.partial(self.summarizer_node)
        workflow.add_node("Summarizer", summarizer_agent)

        # Create edges
        for agent_name, _ in self.agents:
            workflow.add_edge(agent_name, "Supervisor")

        conditional_map: dict[Hashable, str] = {k[0]: k[0] for k in self.agents}
        conditional_map["FINISH"] = "Summarizer"
        workflow.add_conditional_edges(
            "Supervisor",
            lambda x: x["next"],
            conditional_map,
        )
        workflow.add_edge(START, "Supervisor")
        workflow.add_edge("Summarizer", END)
        graph = workflow.compile()
        return graph

    def run(self, query: str, thread_id: str) -> AgentOutput:
        """Run graph against a query."""
        graph = self.create_graph()
        config = RunnableConfig(configurable={"thread_id": thread_id})
        res = graph.invoke(
            input={"messages": [HumanMessage(content=query)]}, config=config
        )
        return self._process_output(res)

    async def arun(self, query: str, thread_id: str) -> AgentOutput:
        """Arun method of the service."""
        graph = self.create_graph()
        config = RunnableConfig(configurable={"thread_id": thread_id})
        res = await graph.ainvoke(
            input={"messages": [HumanMessage(content=query)]}, config=config
        )
        return self._process_output(res)

    async def astream(self, query: str, thread_id: str) -> AsyncIterator[str]:  # type: ignore
        """Astream method of the service."""
        graph = self.create_graph()
        config = RunnableConfig(configurable={"thread_id": thread_id})
        async for chunk in graph.astream(
            input={"messages": [HumanMessage(content=query)]}, config=config
        ):
            if "Supervisor" in chunk.keys() and chunk["Supervisor"]["next"] != "FINISH":
                yield f'\nCalling agent : {chunk["Supervisor"]["next"]}\n'
            else:
                values = [i for i in chunk.values()]  # noqa: C416
                if "messages" in values[0]:
                    yield f'\n {values[0]["messages"][0].content}'

    @staticmethod
    def _process_output(output: Any) -> AgentOutput:
        """Format the output."""
        agent_steps = []
        for message in output["messages"][1:]:
            if "steps" in message.additional_kwargs:
                agent_steps.extend(message.additional_kwargs["steps"])
        return AgentOutput(response=output["messages"][-1].content, steps=agent_steps)
