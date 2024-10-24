# Standard library imports
import asyncio
import copy
import json
from collections import defaultdict
from typing import List

# Package/library imports
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import ValidationError

from swarm_copy.new_types import (
    Agent,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Response,
    Result,
)
from swarm_copy.tools import BaseTool, PrintAccountDetailsTool

# Local imports
from swarm_copy.util import debug_print

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    def __init__(self, client=None):
        if not client:
            client = AsyncOpenAI()
        self.client = client

    async def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [tool.pydantic_to_openai_schema() for tool in agent.functions]
        # hide context_variables from model
        # for tool in tools:
        #     params = tool["function"]["parameters"]
        #     params["properties"].pop(__CTX_VARS_NAME__, None)
        #     if __CTX_VARS_NAME__ in params["required"]:
        #         params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": False,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return await self.client.chat.completions.create(**create_params)

    def handle_function_result(self, result, debug) -> Result:
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
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    async def execute_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        tools: list[type[BaseTool]],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        """Run async tool calls"""
        tasks = [
            asyncio.create_task(
                self.handle_tool_call(
                    tool_call=tool_call,
                    tools=tools,
                    context_variables=context_variables,
                    debug=debug,
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
        tools: List[type[BaseTool]],
        context_variables: dict,
        debug: bool,
    ) -> dict[str, str]:
        tool_map = {tool.name: tool for tool in tools}

        name = tool_call.function.name
        # handle missing tool case, skip to next tool
        if name not in tool_map:
            debug_print(debug, f"Tool {name} not found in function map.")
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": f"Error: Tool {name} not found.",
            }
        kwargs = json.loads(tool_call.function.arguments)
        debug_print(debug, f"Processing tool call: {name} with arguments {kwargs}")

        tool = tool_map[name]
        try:
            input_schema = tool.__annotations__.get("input_schema")(**kwargs)
        except ValidationError as err:
            response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": str(err),
            }
            return response, None

        tool_metadata = tool.__annotations__.get("metadata")(**context_variables)
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

        result: Result = self.handle_function_result(raw_result, debug)
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
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
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
                debug=debug,
            )
            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = await self.execute_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
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


def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")


async def run_demo_loop(starting_agent, context_variables=None, debug=False) -> None:
    client = Swarm()
    print("Starting Swarm CLI 🐝")

    messages = []
    agent = starting_agent

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})

        response = await client.arun(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            debug=debug,
        )
        pretty_print_messages(response.messages)

        messages.extend(response.messages)
        agent = response.agent


def instructions(context_variables):
    name = context_variables.get("name", "User")
    return f"You are a helpful agent. Greet the user by name ({name})."


def print_account_details(context_variables: dict):
    user_id = context_variables.get("user_id", None)
    name = context_variables.get("name", None)
    print(f"Account Details: {name} {user_id}")
    return "Success"


if __name__ == "__main__":
    agent = Agent(
        name="Agent",
        instructions=instructions,
        functions=[PrintAccountDetailsTool],
    )

    context_variables = {"name": "James", "user_id": 123}
    asyncio.run(run_demo_loop(agent, context_variables, False))
