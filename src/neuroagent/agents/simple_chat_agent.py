"""Simple agent."""

import logging
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import create_react_agent
from pydantic.v1 import root_validator

from neuroagent.agents import AgentOutput, AgentStep, BaseAgent

logger = logging.getLogger(__name__)


class SimpleChatAgent(BaseAgent):
    """Simple Agent class."""

    memory: BaseCheckpointSaver

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def create_agent(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Instantiate the clients upon class creation."""
        values["agent"] = create_react_agent(
            model=values["llm"],
            tools=values["tools"],
            checkpointer=values["memory"],
            state_modifier="""You are a helpful assistant helping scientists with neuro-scientific questions.
                You must always specify in your answers from which brain regions the information is extracted.
                Do no blindly repeat the brain region requested by the user, use the output of the tools instead.""",
        )
        return values

    def run(self, session_id: str, query: str) -> Any:
        """Run the agent against a query."""
        pass

    async def arun(self, thread_id: str, query: str) -> Any:
        """Run the agent against a query."""
        config = {"configurable": {"thread_id": thread_id}}
        input_message = HumanMessage(content=query)
        result = await self.agent.ainvoke({"messages": [input_message]}, config=config)
        return self._process_output(result)

    async def astream(self, thread_id: str, query: str) -> AsyncIterator[str]:  # type: ignore
        """Run the agent against a query in streaming way.

        Parameters
        ----------
        thread_id
            ID of the thread of the chat.
        query
            Query of the user

        Returns
        -------
            Iterator streaming the processed output of the LLM
        """
        config = {"configurable": {"thread_id": thread_id}}
        streamed_response = self.agent.astream_events(
            {"messages": query}, version="v2", config=config
        )

        async for event in streamed_response:
            kind = event["event"]

            # newline everytime model starts streaming.
            if kind == "on_chat_model_start":
                yield "\n\n"
            # check for the model stream.
            if kind == "on_chat_model_stream":
                # check if we are calling the tools.
                data_chunk = event["data"]["chunk"]
                if "tool_calls" in data_chunk.additional_kwargs:
                    tool = data_chunk.additional_kwargs["tool_calls"]
                    if tool[0]["function"]["name"]:
                        yield (
                            f'\nCalling tool : {tool[0]["function"]["name"]} with'
                            " arguments : "
                        )
                    if tool[0]["function"]["arguments"]:
                        yield tool[0]["function"]["arguments"]

                content = data_chunk.content
                if content:
                    yield content
        yield "\n"

    @staticmethod
    def _process_output(output: Any) -> AgentOutput:
        """Format the output.

        Parameters
        ----------
        output
            Raw output of the LLM

        Returns
        -------
            Unified output across different agent type.
        """
        # Gather tool name and arguments together
        agent_steps = [
            AgentStep(
                tool_name=tool_call["name"],
                arguments=tool_call["args"],
            )
            for step in output["messages"]
            if isinstance(step, AIMessage) and step.additional_kwargs
            for tool_call in step.tool_calls
        ]
        return AgentOutput(response=output["messages"][-1].content, steps=agent_steps)
