"""Testing chat agent"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from neuroagent.agents import AgentOutput, AgentStep, SimpleChatAgent


@pytest.mark.asyncio
async def test_arun(fake_llm_with_tools, httpx_mock):
    llm, tools, fake_responses = await anext(fake_llm_with_tools)
    json_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.json"
    with open(json_path) as f:
        knowledge_graph_response = json.load(f)

    httpx_mock.add_response(
        url="http://fake_url",
        json=knowledge_graph_response,
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        agent = SimpleChatAgent(llm=llm, tools=tools, memory=memory)

        response = await agent.arun(
            thread_id="test", query="Call get_morpho with thalamus."
        )

        assert isinstance(response, AgentOutput)
        assert response.response == "Great answer"
        assert len(response.steps) == 1
        assert isinstance(response.steps[0], AgentStep)
        assert response.steps[0].tool_name == "get-morpho-tool"
        assert response.steps[0].arguments == {
            "brain_region_id": "http://api.brain-map.org/api/v2/data/Structure/549"
        }

        messages = memory.alist({"configurable": {"thread_id": "test"}})
        messages_list = [message async for message in messages]
        assert len(messages_list) == 5

        assert messages_list[-1].metadata["writes"]["__start__"]["messages"][
            0
        ] == HumanMessage(content="Call get_morpho with thalamus.")
        assert isinstance(
            messages_list[1].metadata["writes"]["tools"]["messages"][0], ToolMessage
        )
        assert (
            messages_list[0].metadata["writes"]["agent"]["messages"][0].content
            == "Great answer"
        )

        # The ids of the messages have to be unique for them to be added to the graph's state.
        for i, response in enumerate(fake_responses):
            response.id = str(i)

        llm.messages = iter(fake_responses)
        response = await agent.arun(
            thread_id="test", query="Call get_morpho with thalamus."
        )
        messages = memory.alist({"configurable": {"thread_id": "test"}})
        messages_list = [message async for message in messages]
        assert len(messages_list) == 10


@pytest.mark.asyncio
async def test_astream(fake_llm_with_tools, httpx_mock):
    llm, tools, fake_responses = await anext(fake_llm_with_tools)
    json_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.json"
    with open(json_path) as f:
        knowledge_graph_response = json.load(f)

    httpx_mock.add_response(
        url="http://fake_url",
        json=knowledge_graph_response,
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        agent = SimpleChatAgent(llm=llm, tools=tools, memory=memory)

        response = agent.astream(
            thread_id="test", query="Find morphologies in the thalamus"
        )

        msg_list = "".join([el async for el in response])
        assert (
            msg_list == "\n\n\nCalling tool : get-morpho-tool with arguments :"
            ' {"brain_region_id":"http://api.brain-map.org/api/v2/data/Structure/549"}\n\nGreat'
            " answer\n"
        )

        messages = memory.alist({"configurable": {"thread_id": "test"}})
        messages_list = [message async for message in messages]
        assert len(messages_list) == 5
        assert (
            messages_list[-1].metadata["writes"]["__start__"]["messages"]
            == "Find morphologies in the thalamus"
        )
        assert isinstance(
            messages_list[1].metadata["writes"]["tools"]["messages"][0], ToolMessage
        )
        assert (
            messages_list[0].metadata["writes"]["agent"]["messages"][0].content
            == "Great answer"
        )

        # The ids of the messages have to be unique for them to be added to the graph's state.
        for i, response in enumerate(fake_responses):
            response.id = str(i)
        llm.messages = iter(fake_responses)
        response = agent.astream(
            thread_id="test", query="Find morphologies in the thalamus please."
        )
        msg_list = "".join([el async for el in response])  # Needed to trigger streaming
        messages = memory.alist({"configurable": {"thread_id": "test"}})
        messages_list = [message async for message in messages]
        assert len(messages_list) == 10
