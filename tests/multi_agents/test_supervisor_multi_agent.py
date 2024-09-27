from unittest.mock import MagicMock, AsyncMock

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage, SystemMessage

import pytest

from neuroagent.multi_agents.supervisor_multi_agent import AgentState
from src.neuroagent.multi_agents import SupervisorMultiAgent


def test_create_main_agent():
    mock_llm = MagicMock()
    bind_function_result = MagicMock()
    bind_function_result.__ror__.return_value = {}
    mock_llm.bind_functions.return_value = bind_function_result
    data = {
        "llm": mock_llm,
        "agents": [("agent1",)]
    }
    from src.neuroagent.multi_agents import SupervisorMultiAgent
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
        return {
            "messages": [
                mock_message
            ]
        }

    agent_state = MagicMock()
    agent = MagicMock()
    agent.ainvoke = mock_ainvoke
    from src.neuroagent.multi_agents import SupervisorMultiAgent
    agent_node_test = await SupervisorMultiAgent.agent_node(agent_state, agent, "test_agent")

    assert isinstance(agent_node_test, dict)
    assert "messages" in agent_node_test
    assert len(agent_node_test["messages"]) == 1
    assert agent_node_test["messages"][0].content == "hello"
    assert agent_node_test["messages"][0].name == "test_agent"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.asyncio
async def test_summarizer_node():
    class FakeChatModel(GenericFakeChatModel):
        def bind_tools(self, functions: list):
            return self

        def bind_functions(self, **kwargs):
            return self


    fake_state = AgentState(messages=[HumanMessage(content="What is the airspeed velocity of an unladen swallow?"),
                                      SystemMessage(content="11 m/s")])

    mock_llm = FakeChatModel(messages=iter([]))
    agent = SupervisorMultiAgent(agents=[("agent1", [])], llm=mock_llm)

    mock_message = SystemMessage(
        content="hello",
        name="test_agent",
    )

    mock_summarizer = MagicMock()
    mock_summarizer.ainvoke = AsyncMock()
    mock_summarizer.ainvoke.return_value = mock_message
    agent.summarizer = mock_summarizer
    result = await agent.summarizer_node(fake_state)
    assert result["messages"][0].content == "hello"


def test_create_graph():
    class FakeChatModel(GenericFakeChatModel):
        def bind_tools(self, functions: list):
            return self

        def bind_functions(self, **kwargs):
            return self

    mock_llm = FakeChatModel(messages=iter([]))
    agent = SupervisorMultiAgent(agents=[("agent1", [])], llm=mock_llm)
    result = agent.create_graph()
    nodes = result.nodes
    assert "agent1" in nodes
    assert "Supervisor" in nodes
    assert "Summarizer" in nodes
