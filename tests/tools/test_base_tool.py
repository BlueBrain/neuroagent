from typing import Literal, Type

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import ToolException
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from neuroagent.tools.base_tool import BasicTool


class input_for_test(BaseModel):
    test_str: str
    test_int: int
    test_litteral: Literal["Allowed_1", "Allowed_2"] | None = None


class basic_tool_for_test(BasicTool):
    name: str = "basic_tool_for_test"
    description: str = "Dummy tool to test validation and tool errors."
    args_schema: Type[BaseModel] = input_for_test

    def _run(self, test_str, test_int):
        raise ToolException("fake tool error message", self.name)


def test_basic_tool_error_handling():
    response_list = [
        # test tool error.
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call00",
                    "name": "basic_tool_for_test",
                    "args": {
                        "test_str": "Hello",
                        "test_int": 1,
                    },
                },
            ],
        ),
        # test all possible validation error.
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call123",
                    "name": "basic_tool_for_test",
                    "args": {
                        "test_str": "3",
                        "test_int": 1,
                        "test_litteral": "Forbidden_value",
                    },
                },
                {
                    "id": "tool_call567",
                    "name": "basic_tool_for_test",
                    "args": {},
                },
                {
                    "id": "tool_call891",
                    "name": "basic_tool_for_test",
                    "args": {
                        "test_str": {"dummy": "test_dict"},
                        "test_int": "hello",
                    },
                },
            ],
        ),
        AIMessage(content="fake answer"),
    ]
    tool_list = [basic_tool_for_test()]

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_tools(self, functions: list):
            return self

    fake_llm = FakeFuntionChatModel(responses=response_list)

    fake_agent = create_react_agent(fake_llm, tool_list)

    response = fake_agent.invoke({"messages": [HumanMessage(content="fake_message")]})

    assert (
        response["messages"][2].content
        == '{"basic_tool_for_test": "fake tool error message"}'
    )
    assert (
        response["messages"][4].content
        == '[{"Validation error": "Wrong value: provided Forbidden_value for input'
        ' test_litteral. Try again and change this problematic input."}]'
    )
    assert (
        response["messages"][5].content
        == '[{"Validation error": "Missing input : test_str. Try again and add this'
        ' input."}, {"Validation error": "Missing input : test_int. Try again and'
        ' add this input."}]'
    )
    assert (
        response["messages"][6].content
        == '[{"Validation error": "test_str. Input should be a valid string"}, '
        '{"Validation error": "test_int. Input should be a valid integer, '
        'unable to parse string as an integer"}]'
    )
    assert response["messages"][7].content == "fake answer"
