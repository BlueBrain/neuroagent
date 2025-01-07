"""Run the agent routine."""

import asyncio
import copy
import json
from collections import defaultdict
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import ValidationError

from neuroagent.app.database.sql_schemas import Entity, Messages, ToolCalls
from neuroagent.new_types import (
    Agent,
    HILResponse,
    Response,
    Result,
)
from neuroagent.tools.base_tool import BaseTool
from neuroagent.utils import get_entity, merge_chunk, messages_to_openai_content


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
        stream: bool = False,
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
            "stream": stream,
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
        tool_calls: list[ToolCalls],
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

        hil_messages = [msg for msg in messages if isinstance(msg, HILResponse)]

        # If a validation is required, return only this information
        if hil_messages:
            return Response(
                messages=[],
                agent=None,
                context_variables=context_variables,
                hil_messages=hil_messages,
            )

        return Response(
            messages=messages, agent=agent, context_variables=context_variables
        )

    async def handle_tool_call(
        self,
        tool_call: ToolCalls,
        tools: list[type[BaseTool]],
        context_variables: dict[str, Any],
    ) -> tuple[dict[str, str], Agent | None]:
        """Run individual tools."""
        tool_map = {tool.name: tool for tool in tools}

        name = tool_call.name
        # handle missing tool case, skip to next tool
        if name not in tool_map:
            return {
                "role": "tool",
                "tool_call_id": tool_call.tool_call_id,
                "tool_name": name,
                "content": f"Error: Tool {name} not found.",
            }, None
        kwargs = json.loads(tool_call.arguments)

        tool = tool_map[name]
        try:
            input_schema = tool.__annotations__["input_schema"](**kwargs)
        except ValidationError as err:
            response = {
                "role": "tool",
                "tool_call_id": tool_call.tool_call_id,
                "tool_name": name,
                "content": str(err),
            }
            return response, None

        if tool.hil:
            # Case where the tool call hasn't been validated yet
            if tool_call.validated is None:
                return HILResponse(
                    message="Please validate the following inputs before proceeding.",
                    inputs=input_schema.model_dump(),
                    tool_call_id=tool_call.tool_call_id,
                ), None

            # Case where the tool call has been refused
            if not tool_call.validated:
                return {
                    "role": "tool",
                    "tool_call_id": tool_call.tool_call_id,
                    "tool_name": name,
                    "content": "The tool call has been invalidated by the user.",
                }, None
            # Else the tool call has been validated, we can proceed normally

        tool_metadata = tool.__annotations__["metadata"](**context_variables)
        tool_instance = tool(input_schema=input_schema, metadata=tool_metadata)
        # pass context_variables to agent functions
        try:
            raw_result = await tool_instance.arun()
        except Exception as err:
            response = {
                "role": "tool",
                "tool_call_id": tool_call.tool_call_id,
                "tool_name": name,
                "content": str(err),
            }
            return response, None

        result: Result = self.handle_function_result(raw_result)
        response = {
            "role": "tool",
            "tool_call_id": tool_call.tool_call_id,
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
        messages: list[Messages],  # What we track in the DB
        context_variables: dict[str, Any] = {},
        model_override: str | None = None,
        max_turns: int | float = float("inf"),
    ) -> Response:
        """Run the agent main loop."""
        active_agent = agent
        content = await messages_to_openai_content(messages)
        history = copy.deepcopy(content)  # What we send to OpenAI
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:
            # Run chat completion if the last message isn't a tool call (HIL case)
            if messages[-1].entity != Entity.AI_TOOL:
                # get completion with current history, agent
                completion = await self.get_chat_completion(
                    agent=active_agent,
                    history=history,
                    context_variables=context_variables,
                    model_override=model_override,
                    stream=False,
                )
                message = completion.choices[0].message  # type: ignore
                message.sender = active_agent.name

                # If tool calls requested, instantiate them as an SQL compatible class
                if message.tool_calls:
                    tool_calls = [
                        ToolCalls(
                            tool_call_id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        )
                        for tool_call in message.tool_calls
                    ]
                else:
                    tool_calls = []

                message_json = message.model_dump()
                # Append the history with the json version
                history.append(copy.deepcopy(message_json))
                message_json.pop("tool_calls")

                # Stage the new message for addition to DB
                messages.append(
                    Messages(
                        order=len(messages),
                        thread_id=messages[-1].thread_id,
                        entity=get_entity(message_json),
                        content=json.dumps(message_json),
                        tool_calls=tool_calls,
                    )
                )
            else:
                message = messages[-1]

            if not message.tool_calls:
                break

            # Handle function calls, updating context_variables, and switching agents
            partial_response = await self.execute_tool_calls(
                messages[-1].tool_calls, active_agent.tools, context_variables
            )

            # If the tool call response contains HIL validation, do not update anything and return
            if partial_response.hil_messages:
                return Response(
                    messages=history[init_len:],
                    agent=active_agent,
                    context_variables=context_variables,
                    hil_messages=partial_response.hil_messages,
                )

            history.extend(partial_response.messages)
            messages.extend(
                [
                    Messages(
                        order=len(messages) + i,
                        thread_id=messages[-1].thread_id,
                        entity=Entity.TOOL,
                        content=json.dumps(tool_response),
                    )
                    for i, tool_response in enumerate(partial_response.messages)
                ]
            )
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )

    async def astream(
        self,
        agent: Agent,
        messages: list[dict[str, Any]],
        context_variables: dict[str, Any] = {},
        model_override: str | None = None,
        max_turns: int | float = float("inf"),
    ) -> AsyncIterator[str | Response]:
        """Stream the agent response."""
        active_agent = agent

        history = copy.deepcopy(messages)
        init_len = len(messages)
        is_streaming = False

        while len(history) - init_len < max_turns:
            message: dict[str, Any] = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = await self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
            )
            async for chunk in completion:  # type: ignore
                delta = json.loads(chunk.choices[0].delta.model_dump_json())

                # Check for tool calls
                if delta["tool_calls"]:
                    tool = delta["tool_calls"][0]["function"]
                    if tool["name"]:
                        yield f"\nCalling tool : {tool['name']} with arguments : "
                    if tool["arguments"]:
                        yield tool["arguments"]

                # Check for content
                if delta["content"]:
                    if not is_streaming:
                        yield "\n<begin_llm_response>\n"
                        is_streaming = True
                    yield delta["content"]

                delta.pop("role", None)
                merge_chunk(message, delta)

            message["tool_calls"] = list(message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            history.append(message)

            if not message["tool_calls"]:
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = await self.execute_tool_calls(
                tool_calls, active_agent.tools, context_variables
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield Response(
            messages=history[init_len - 1 :],
            agent=active_agent,
            context_variables=context_variables,
        )
