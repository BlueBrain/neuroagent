from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from neuroagent.multi_agents.supervisor_multi_agent import AgentState
from src.neuroagent.multi_agents import SupervisorMultiAgent


def test_create_main_agent_initialization():
    mock_llm = Mock()
    bind_function_result = MagicMock()
    bind_function_result.__ror__.return_value = {}
    mock_llm.bind_functions.return_value = bind_function_result
    data = {"llm": mock_llm, "agents": [("agent1",)]}

    result = SupervisorMultiAgent.create_main_agent(data)
    assert "main_agent" in result
    assert "summarizer" in result


@pytest.mark.asyncio
async def test_agent_node():
    mock_message = HumanMessage(
        content="hello",
        name="test_agent",
    )

    async def mock_ainvoke(_):
        return {"messages": [mock_message]}

    agent_state = Mock()
    agent = Mock()
    agent.ainvoke = mock_ainvoke

    agent_node_test = await SupervisorMultiAgent.agent_node(
        agent_state, agent, "test_agent"
    )

    assert isinstance(agent_node_test, dict)
    assert "messages" in agent_node_test
    assert len(agent_node_test["messages"]) == 1
    assert agent_node_test["messages"][0].content == "hello"
    assert agent_node_test["messages"][0].name == "test_agent"


@pytest.mark.asyncio
async def test_summarizer_node(fake_llm_with_tools):
    fake_state = AgentState(
        messages=[
            HumanMessage(
                content="What is the airspeed velocity of an unladen swallow?"
            ),
            SystemMessage(content="11 m/s"),
        ]
    )

    mock_llm, _, _ = await anext(fake_llm_with_tools)
    agent = SupervisorMultiAgent(agents=[("agent1", [])], llm=mock_llm)

    mock_message = SystemMessage(
        content="hello",
        name="test_agent",
    )

    mock_summarizer = Mock()
    mock_summarizer.ainvoke = AsyncMock()
    mock_summarizer.ainvoke.return_value = mock_message
    agent.summarizer = mock_summarizer
    result = await agent.summarizer_node(fake_state)
    assert result["messages"][0].content == "hello"


@pytest.mark.asyncio
async def test_create_graph(fake_llm_with_tools):
    mock_llm, _, _ = await anext(fake_llm_with_tools)
    agent = SupervisorMultiAgent(agents=[("agent1", [])], llm=mock_llm)
    result = agent.create_graph()
    nodes = result.nodes
    assert "agent1" in nodes
    assert "Supervisor" in nodes
    assert "Summarizer" in nodes
