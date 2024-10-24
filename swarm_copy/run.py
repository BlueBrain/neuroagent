"""Run the agent routine."""

import asyncio
import copy
import json
from collections import defaultdict
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import ValidationError

from swarm_copy.new_types import (
    Agent,
    Response,
    Result,
)
from swarm_copy.tools import BaseTool


class AgentsRoutine:
    """Agents routine class. Wrapper for all the functions running the agent."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        if not client:
            client = AsyncOpenAI()
        self.client = client

    async def get_chat_completion(
        self,
        agent: Agent,
        history: list[dict[str, str]],
        context_variables: dict[str, Any],
        model_override: str | None,
    ) -> ChatCompletionMessage:
        """Send the OpenAI request."""
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)  # type: ignore
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history

        tools = [tool.pydantic_to_openai_schema() for tool in agent.tools]

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": False,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return await self.client.chat.completions.create(**create_params)  # type: ignore

    def handle_function_result(self, result: Result | Agent) -> Result:
        """Check if agent handoff or regular tool call."""
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    raise TypeError(error_message)

    async def execute_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        tools: list[type[BaseTool]],
        context_variables: dict[str, Any],
    ) -> Response:
        """Run async tool calls."""
        tasks = [
            asyncio.create_task(
                self.handle_tool_call(
                    tool_call=tool_call,
                    tools=tools,
                    context_variables=context_variables,
                )
            )
            for tool_call in tool_calls
        ]
        results = await asyncio.gather(*tasks)
        messages, agents = zip(*results)
        try:
            agent = next((agent for agent in reversed(agents) if agent is not None))
        except StopIteration:
            agent = None
        response = Response(
            messages=messages, agent=agent, context_variables=context_variables
        )
        return response

    async def handle_tool_call(
        self,
        tool_call: ChatCompletionMessageToolCall,
        tools: list[type[BaseTool]],
        context_variables: dict[str, Any],
    ) -> tuple[dict[str, str], Agent | None]:
        """Run individual tools."""
        tool_map = {tool.name: tool for tool in tools}

        name = tool_call.function.name
        # handle missing tool case, skip to next tool
        if name not in tool_map:
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": f"Error: Tool {name} not found.",
            }, None
        kwargs = json.loads(tool_call.function.arguments)

        tool = tool_map[name]
        try:
            input_schema = tool.__annotations__["input_schema"](**kwargs)
        except ValidationError as err:
            response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": str(err),
            }
            return response, None

        tool_metadata = tool.__annotations__["metadata"](**context_variables)
        tool_instance = tool(input_schema=input_schema, metadata=tool_metadata)
        # pass context_variables to agent functions
        try:
            raw_result = await tool_instance.arun()
        except Exception as err:
            response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": str(err),
            }
            return response, None

        result: Result = self.handle_function_result(raw_result)
        response = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "tool_name": name,
            "content": result.value,
        }
        if result.agent:
            agent = result.agent
        else:
            agent = None
        return response, agent

    async def arun(
        self,
        agent: Agent,
        messages: list[dict[str, str]],
        context_variables: dict[str, Any] = {},
        model_override: str | None = None,
        max_turns: int | float = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        """Run the agent main loop."""
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:
            # get completion with current history, agent
            completion = await self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
            )
            message = completion.choices[0].message  # type: ignore
            message.sender = active_agent.name
            history.append(json.loads(message.model_dump_json()))

            if not message.tool_calls or not execute_tools:
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = await self.execute_tool_calls(
                message.tool_calls, active_agent.tools, context_variables
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history,
            agent=active_agent,
            context_variables=context_variables,
        )
