import json
from typing import AsyncIterator
from unittest.mock import patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from neuroagent.agent_routine import AgentsRoutine
from neuroagent.app.database.sql_schemas import Entity, Messages, ToolCalls
from neuroagent.new_types import Agent, Response, Result
from tests.mock_client import create_mock_response


class TestAgentsRoutine:
    @pytest.mark.asyncio
    async def test_get_chat_completion_simple_message(self, mock_openai_client):
        routine = AgentsRoutine(client=mock_openai_client)

        agent = Agent()
        response = await routine.get_chat_completion(
            agent=agent,
            history=[{"role": "user", "content": "Hello !"}],
            context_variables={},
            model_override=None,
        )
        mock_openai_client.assert_create_called_with(
            **{
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful agent."},
                    {"role": "user", "content": "Hello !"},
                ],
                "tools": None,
                "tool_choice": None,
                "stream": False,
            }
        )

        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "sample response content"

    @pytest.mark.asyncio
    async def test_get_chat_completion_callable_sys_prompt(self, mock_openai_client):
        routine = AgentsRoutine(client=mock_openai_client)

        def agent_instruction(context_variables):
            twng = context_variables.get("twng")
            mrt = context_variables.get("mrt")
            return f"This is your new instructions with {twng} and {mrt}."

        agent = Agent(instructions=agent_instruction)
        response = await routine.get_chat_completion(
            agent=agent,
            history=[{"role": "user", "content": "Hello !"}],
            context_variables={"mrt": "Great mrt", "twng": "Bad twng"},
            model_override=None,
        )
        mock_openai_client.assert_create_called_with(
            **{
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "This is your new instructions with Bad twng and Great mrt.",
                    },
                    {"role": "user", "content": "Hello !"},
                ],
                "tools": None,
                "tool_choice": None,
                "stream": False,
            }
        )

        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "sample response content"

    @pytest.mark.asyncio
    async def test_get_chat_completion_tools(
        self, mock_openai_client, get_weather_tool
    ):
        routine = AgentsRoutine(client=mock_openai_client)

        agent = Agent(tools=[get_weather_tool])
        response = await routine.get_chat_completion(
            agent=agent,
            history=[{"role": "user", "content": "Hello !"}],
            context_variables={},
            model_override=None,
        )
        mock_openai_client.assert_create_called_with(
            **{
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful agent."},
                    {"role": "user", "content": "Hello !"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Great description",
                            "strict": False,
                            "parameters": {
                                "properties": {
                                    "location": {"title": "Location", "type": "string"}
                                },
                                "required": ["location"],
                                "title": "FakeToolInput",
                                "type": "object",
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": None,
                "stream": False,
                "parallel_tool_calls": True,
            }
        )

        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "sample response content"

    def test_handle_function_result(self, mock_openai_client):
        routine = AgentsRoutine(client=mock_openai_client)

        # Raw result is already a result
        raw_result = Result(value="Nice weather")
        result = routine.handle_function_result(raw_result)
        assert result == raw_result

        # Raw result is an agent for handoff
        raw_result = Agent(name="Test agent 2")
        result = routine.handle_function_result(raw_result)
        assert result == Result(
            value=json.dumps({"assistant": raw_result.name}), agent=raw_result
        )

        # Raw result is a tool output (Typically dict/list dict)
        raw_result = [{"result_1": "Great result", "result_2": "Bad result"}]
        result = routine.handle_function_result(raw_result)
        assert result == Result(value=str(raw_result))

    @pytest.mark.asyncio
    async def test_execute_tool_calls_simple(
        self, mock_openai_client, get_weather_tool, agent_handoff_tool
    ):
        routine = AgentsRoutine(client=mock_openai_client)

        mock_openai_client.set_response(
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": "Geneva"}}
                ],
            ),
        )
        agent = Agent(tools=[get_weather_tool, agent_handoff_tool])
        context_variables = {}

        tool_call_message = await routine.get_chat_completion(
            agent,
            history=[{"role": "user", "content": "Hello"}],
            context_variables=context_variables,
            model_override=None,
        )
        tool_calls = tool_call_message.choices[0].message.tool_calls
        tool_calls_db = [
            ToolCalls(
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in tool_calls
        ]
        tool_calls_result = await routine.execute_tool_calls(
            tool_calls=tool_calls_db,
            tools=agent.tools,
            context_variables=context_variables,
        )
        assert isinstance(tool_calls_result, Response)
        assert tool_calls_result.messages == [
            {
                "role": "tool",
                "tool_call_id": tool_calls[0].id,
                "tool_name": "get_weather",
                "content": "It's sunny today.",
            }
        ]
        assert tool_calls_result.agent is None
        assert tool_calls_result.context_variables == context_variables

    @pytest.mark.asyncio
    async def test_execute_multiple_tool_calls(
        self, mock_openai_client, get_weather_tool, agent_handoff_tool
    ):
        routine = AgentsRoutine(client=mock_openai_client)

        mock_openai_client.set_response(
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": "Geneva"}},
                    {"name": "get_weather", "args": {"location": "Lausanne"}},
                ],
            ),
        )
        agent = Agent(tools=[get_weather_tool, agent_handoff_tool])
        context_variables = {"planet": "Earth"}

        tool_call_message = await routine.get_chat_completion(
            agent,
            history=[{"role": "user", "content": "Hello"}],
            context_variables=context_variables,
            model_override=None,
        )
        tool_calls = tool_call_message.choices[0].message.tool_calls
        tool_calls_db = [
            ToolCalls(
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in tool_calls
        ]
        tool_calls_result = await routine.execute_tool_calls(
            tool_calls=tool_calls_db,
            tools=agent.tools,
            context_variables=context_variables,
        )

        assert isinstance(tool_calls_result, Response)
        assert tool_calls_result.messages == [
            {
                "role": "tool",
                "tool_call_id": tool_calls[0].id,
                "tool_name": "get_weather",
                "content": "It's sunny today in Geneva from planet Earth.",
            },
            {
                "role": "tool",
                "tool_call_id": tool_calls[1].id,
                "tool_name": "get_weather",
                "content": "It's sunny today in Lausanne from planet Earth.",
            },
        ]
        assert tool_calls_result.agent is None
        assert tool_calls_result.context_variables == context_variables

    @pytest.mark.asyncio
    async def test_execute_tool_calls_handoff(
        self, mock_openai_client, get_weather_tool, agent_handoff_tool
    ):
        routine = AgentsRoutine(client=mock_openai_client)

        mock_openai_client.set_response(
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "agent_handoff_tool", "args": {}}],
            ),
        )
        agent_1 = Agent(name="Test agent 1", tools=[agent_handoff_tool])
        agent_2 = Agent(
            name="Test agent 2", tools=[get_weather_tool, agent_handoff_tool]
        )
        context_variables = {"to_agent": agent_2}

        tool_call_message = await routine.get_chat_completion(
            agent_1,
            history=[{"role": "user", "content": "Hello"}],
            context_variables=context_variables,
            model_override=None,
        )
        tool_calls = tool_call_message.choices[0].message.tool_calls
        tool_calls_db = [
            ToolCalls(
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in tool_calls
        ]
        tool_calls_result = await routine.execute_tool_calls(
            tool_calls=tool_calls_db,
            tools=agent_1.tools,
            context_variables=context_variables,
        )

        assert isinstance(tool_calls_result, Response)
        assert tool_calls_result.messages == [
            {
                "role": "tool",
                "tool_call_id": tool_calls[0].id,
                "tool_name": "agent_handoff_tool",
                "content": json.dumps({"assistant": agent_2.name}),
            }
        ]
        assert tool_calls_result.agent == agent_2
        assert tool_calls_result.context_variables == context_variables

    @pytest.mark.asyncio
    async def test_handle_tool_call_simple(
        self, mock_openai_client, get_weather_tool, agent_handoff_tool
    ):
        routine = AgentsRoutine(client=mock_openai_client)

        mock_openai_client.set_response(
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": "Geneva"}}
                ],
            ),
        )
        agent = Agent(tools=[get_weather_tool, agent_handoff_tool])
        context_variables = {}

        tool_call_message = await routine.get_chat_completion(
            agent,
            history=[{"role": "user", "content": "Hello"}],
            context_variables=context_variables,
            model_override=None,
        )
        tool_call = tool_call_message.choices[0].message.tool_calls[0]
        tool_call_db = ToolCalls(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        tool_call_result = await routine.handle_tool_call(
            tool_call=tool_call_db,
            tools=agent.tools,
            context_variables=context_variables,
        )

        assert tool_call_result == (
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": "get_weather",
                "content": "It's sunny today.",
            },
            None,
        )

    @pytest.mark.asyncio
    async def test_handle_tool_call_context_var(
        self, mock_openai_client, get_weather_tool, agent_handoff_tool
    ):
        routine = AgentsRoutine(client=mock_openai_client)

        mock_openai_client.set_response(
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": "Geneva"}},
                ],
            ),
        )
        agent = Agent(tools=[get_weather_tool, agent_handoff_tool])
        context_variables = {"planet": "Earth"}

        tool_call_message = await routine.get_chat_completion(
            agent,
            history=[{"role": "user", "content": "Hello"}],
            context_variables=context_variables,
            model_override=None,
        )
        tool_call = tool_call_message.choices[0].message.tool_calls[0]
        tool_call_db = ToolCalls(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        tool_calls_result = await routine.handle_tool_call(
            tool_call=tool_call_db,
            tools=agent.tools,
            context_variables=context_variables,
        )

        assert tool_calls_result == (
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": "get_weather",
                "content": "It's sunny today in Geneva from planet Earth.",
            },
            None,
        )

    @pytest.mark.asyncio
    async def test_handle_tool_call_handoff(
        self, mock_openai_client, get_weather_tool, agent_handoff_tool
    ):
        routine = AgentsRoutine(client=mock_openai_client)

        mock_openai_client.set_response(
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "agent_handoff_tool", "args": {}}],
            ),
        )
        agent_1 = Agent(name="Test agent 1", tools=[agent_handoff_tool])
        agent_2 = Agent(
            name="Test agent 2", tools=[get_weather_tool, agent_handoff_tool]
        )
        context_variables = {"to_agent": agent_2}

        tool_call_message = await routine.get_chat_completion(
            agent_1,
            history=[{"role": "user", "content": "Hello"}],
            context_variables=context_variables,
            model_override=None,
        )
        tool_call = tool_call_message.choices[0].message.tool_calls[0]
        tool_call_db = ToolCalls(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        tool_calls_result = await routine.handle_tool_call(
            tool_call=tool_call_db,
            tools=agent_1.tools,
            context_variables=context_variables,
        )

        assert tool_calls_result == (
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": "agent_handoff_tool",
                "content": json.dumps({"assistant": agent_2.name}),
            },
            agent_2,
        )

    @pytest.mark.asyncio
    async def test_arun(self, mock_openai_client, get_weather_tool, agent_handoff_tool):
        agent_1 = Agent(name="Test Agent", tools=[agent_handoff_tool])
        agent_2 = Agent(name="Test Agent", tools=[get_weather_tool])
        messages = [
            Messages(
                order=0,
                thread_id="fake_id",
                entity=Entity.USER,
                content=json.dumps(
                    {
                        "role": "user",
                        "content": {
                            "role": "user",
                            "content": "What's the weather like in San Francisco?",
                        },
                    }
                ),
            )
        ]
        context_variables = {"to_agent": agent_2, "planet": "Mars"}
        # set mock to return a response that triggers function call
        mock_openai_client.set_sequential_responses(
            [
                create_mock_response(
                    message={"role": "assistant", "content": ""},
                    function_calls=[{"name": "agent_handoff_tool", "args": {}}],
                ),
                create_mock_response(
                    message={"role": "assistant", "content": ""},
                    function_calls=[
                        {"name": "get_weather", "args": {"location": "Montreux"}}
                    ],
                ),
                create_mock_response(
                    {"role": "assistant", "content": "sample response content"}
                ),
            ]
        )

        # set up client and run
        client = AgentsRoutine(client=mock_openai_client)
        response = await client.arun(
            agent=agent_1, messages=messages, context_variables=context_variables
        )

        assert response.messages[2]["role"] == "tool"
        assert response.messages[2]["content"] == json.dumps(
            {"assistant": agent_1.name}
        )
        assert response.messages[-2]["role"] == "tool"
        assert (
            response.messages[-2]["content"]
            == "It's sunny today in Montreux from planet Mars."
        )
        assert response.messages[-1]["role"] == "assistant"
        assert response.messages[-1]["content"] == "sample response content"
        assert response.agent == agent_2
        assert response.context_variables == context_variables

    @pytest.mark.asyncio
    async def test_astream(
        self, mock_openai_client, get_weather_tool, agent_handoff_tool
    ):
        agent_1 = Agent(name="Test Agent", tools=[agent_handoff_tool])
        agent_2 = Agent(name="Test Agent", tools=[get_weather_tool])
        messages = [
            Messages(
                order=0,
                thread_id="fake_id",
                entity=Entity.USER,
                content=json.dumps(
                    {
                        "role": "user",
                        "content": {
                            "role": "user",
                            "content": "What's the weather like in San Francisco?",
                        },
                    }
                ),
            )
        ]
        context_variables = {"to_agent": agent_2, "planet": "Mars"}
        routine = AgentsRoutine(client=mock_openai_client)

        async def return_iterator(*args, **kwargs):
            async def mock_openai_streaming_response(
                history,
            ) -> AsyncIterator[ChatCompletionChunk]:
                """
                Simulates streaming chunks of a response for patching.

                Yields
                ------
                    AsyncIterator[ChatCompletionChunk]: Streaming chunks of the response.
                """
                responses = [
                    {
                        "message": {"role": "assistant", "content": ""},
                        "function_call": [{"name": "agent_handoff_tool", "args": {}}],
                    },
                    {
                        "message": {"role": "assistant", "content": ""},
                        "function_call": [
                            {"name": "get_weather", "args": {"location": "Montreux"}}
                        ],
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": "sample response content",
                        },
                    },
                ]
                response_to_call = (
                    len([hist for hist in history if hist["role"] != "tool"]) - 1
                )
                response = responses[response_to_call]

                if "message" in response and "content" in response["message"]:
                    content = response["message"]["content"]
                    for i in range(
                        0, len(content), 10
                    ):  # Stream content in chunks of 10 chars
                        yield ChatCompletionChunk(
                            id="chatcmpl-AdfVmbjxczsgRAADk9pXkmKPFsikY",
                            choices=[
                                Choice(
                                    delta=ChoiceDelta(content=content[i : i + 10]),
                                    finish_reason=None,
                                    index=0,
                                )
                            ],
                            created=1734017726,
                            model="gpt-4o-mini-2024-07-18",
                            object="chat.completion.chunk",
                            system_fingerprint="fp_bba3c8e70b",
                        )

                if "function_call" in response:
                    for function_call in response["function_call"]:
                        yield ChatCompletionChunk(
                            id="chatcmpl-AdfVmbjxczsgRAADk9pXkmKPFsikY",
                            choices=[
                                Choice(
                                    delta=ChoiceDelta(
                                        tool_calls=[
                                            ChoiceDeltaToolCall(
                                                index=0,
                                                function=ChoiceDeltaToolCallFunction(
                                                    arguments=json.dumps(
                                                        function_call["args"]
                                                    ),
                                                    name=function_call["name"],
                                                ),
                                                type="function",
                                            )
                                        ]
                                    ),
                                    finish_reason=None,
                                    index=0,
                                )
                            ],
                            created=1734017726,
                            model="gpt-4o-mini-2024-07-18",
                            object="chat.completion.chunk",
                            system_fingerprint="fp_bba3c8e70b",
                        )

                yield ChatCompletionChunk(
                    id="chatcmpl-AdfVmbjxczsgRAADk9pXkmKPFsikY",
                    choices=[
                        Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                    ],
                    created=1734017726,
                    model="gpt-4o-mini-2024-07-18",
                    object="chat.completion.chunk",
                    system_fingerprint="fp_bba3c8e70b",
                )

            return mock_openai_streaming_response(kwargs["history"])

        tokens = []
        with patch(
            "neuroagent.agent_routine.AgentsRoutine.get_chat_completion",
            new=return_iterator,
        ):
            async for token in routine.astream(
                agent=agent_1, messages=messages, context_variables=context_variables
            ):
                if isinstance(token, str):
                    tokens.append(token)
                else:
                    response = token

        assert (
            "".join(tokens)
            == '\nCalling tool : agent_handoff_tool with arguments : {}\nCalling tool : get_weather with arguments : {"location": "Montreux"}\n<begin_llm_response>\nsample response content'
        )
        assert response.messages[2]["role"] == "tool"
        assert response.messages[2]["content"] == json.dumps(
            {"assistant": agent_1.name}
        )
        assert response.messages[-2]["role"] == "tool"
        assert (
            response.messages[-2]["content"]
            == "It's sunny today in Montreux from planet Mars."
        )
        assert response.messages[-1]["role"] == "assistant"
        assert response.messages[-1]["content"] == "sample response content"
        assert response.agent == agent_2
        assert response.context_variables == context_variables
