"""Simple agent."""

import logging
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from pydantic import model_validator

from neuroagent.agents import AgentOutput, AgentStep, BaseAgent

logger = logging.getLogger(__name__)


class SimpleAgent(BaseAgent):
    """Simple Agent class."""

    @model_validator(mode="before")
    @classmethod
    def create_agent(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Instantiate the clients upon class creation."""
        # Initialise the agent with the tools
        data["agent"] = create_react_agent(
            model=data["llm"],
            tools=data["tools"],
            state_modifier="""You are a helpful assistant helping scientists with neuro-scientific questions.
                You must always specify in your answers from which brain regions the information is extracted.
                Do no blindly repeat the brain region requested by the user, use the output of the tools instead.""",
        )
        return data

    def run(self, query: str) -> Any:
        """Run the agent against a query.

        Parameters
        ----------
        query
            Query of the user

        Returns
        -------
            Processed output of the LLM
        """
        return self._process_output(self.agent.invoke({"messages": [("human", query)]}))

    async def arun(self, query: str) -> Any:
        """Run the agent against a query.

        Parameters
        ----------
        query
            Query of the user

        Returns
        -------
            Processed output of the LLM
        """
        result = await self.agent.ainvoke({"messages": [("human", query)]})
        return self._process_output(result)

    async def astream(self, query: str) -> AsyncIterator[str]:
        """Run the agent against a query in streaming way.

        Parameters
        ----------
        query
            Query of the user

        Returns
        -------
            Iterator streaming the processed output of the LLM
        """
        streamed_response = self.agent.astream_events({"messages": query}, version="v2")

        async for event in streamed_response:
            kind = event["event"]

            # newline everytime model starts streaming.
            if kind == "on_chat_model_start":
                yield "\n\n"  # These \n are to separate AI message from tools.
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
                        )  # This \n is for when there are multiple async tool calls.
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
