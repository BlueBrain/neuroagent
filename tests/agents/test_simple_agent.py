"""Testing agent."""

import json
from pathlib import Path

import pytest
from neuroagent.agents import AgentOutput, AgentStep, SimpleAgent


@pytest.mark.asyncio
async def test_simple_agent_arun(fake_llm_with_tools, httpx_mock):
    json_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.json"
    with open(json_path) as f:
        knowledge_graph_response = json.load(f)

    httpx_mock.add_response(
        url="http://fake_url",
        json=knowledge_graph_response,
    )

    llm, tools, _ = await anext(fake_llm_with_tools)
    simple_agent = SimpleAgent(llm=llm, tools=tools)

    response = await simple_agent.arun(query="Call get_morpho with thalamus.")
    assert isinstance(response, AgentOutput)
    assert response.response == "Great answer"
    assert len(response.steps) == 1
    assert isinstance(response.steps[0], AgentStep)
    assert response.steps[0].tool_name == "get-morpho-tool"
    assert response.steps[0].arguments == {
        "brain_region_id": "http://api.brain-map.org/api/v2/data/Structure/549"
    }


@pytest.mark.asyncio
async def test_simple_agent_astream(fake_llm_with_tools, httpx_mock):
    json_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.json"
    with open(json_path) as f:
        knowledge_graph_response = json.load(f)

    httpx_mock.add_response(
        url="http://fake_url",
        json=knowledge_graph_response,
    )

    llm, tools, _ = await anext(fake_llm_with_tools)
    simple_agent = SimpleAgent(llm=llm, tools=tools)

    response_chunks = simple_agent.astream("Call get_morpho with thalamus.")
    response = "".join([el async for el in response_chunks])

    assert (
        response == "\n\n\nCalling tool : get-morpho-tool with arguments :"
        ' {"brain_region_id":"http://api.brain-map.org/api/v2/data/Structure/549"}\n\nGreat'
        " answer\n"
    )
