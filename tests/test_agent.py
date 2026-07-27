"""
Tests for app.core.agent - create_agent function
"""

import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure app is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.agent import create_agent, get_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_env_api_key():
    """Set a fake DEEPSEEK_API_KEY in environment for the test."""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key-12345"}):
        yield


@pytest.fixture
def mock_dependencies(mock_env_api_key):
    """
    Patch all external dependencies so create_agent runs without real
    network / file-system calls.
    """
    with patch("app.core.agent.ChatOpenAI") as mock_chat, \
         patch("app.core.agent.ChatPromptTemplate") as mock_prompt, \
         patch("app.core.agent.MessagesPlaceholder") as mock_msgs, \
         patch("app.core.agent.create_react_agent") as mock_react, \
         patch("app.core.agent.Tool") as mock_tool, \
         patch("app.core.agent.SecretStr") as mock_secret, \
         patch("app.core.agent._profile_memory") as mock_profile, \
         patch("app.core.agent._checkpointer") as mock_checkpointer, \
         patch("app.core.agent.knowledge_search") as mock_know, \
         patch("app.core.agent.web_search") as mock_web, \
         patch("app.core.agent.calculator") as mock_calc:
        # ChatOpenAI returns a fake llm instance
        mock_chat.return_value = MagicMock(name="llm")
        # ChatPromptTemplate.from_messages returns a fake prompt
        mock_prompt.from_messages.return_value = MagicMock(name="prompt")
        # MessagesPlaceholder returns itself
        mock_msgs.return_value = MagicMock(name="placeholder")
        # create_react_agent returns a fake agent
        mock_react.return_value = MagicMock(name="agent")
        # Tool returns itself (constructor)
        mock_tool.side_effect = lambda **kw: MagicMock(name=f"tool_{kw.get('name','unknown')}")
        # SecretStr wraps the key
        mock_secret.side_effect = lambda key: key
        # User profile mock
        mock_profile.get_context_prompt.return_value = ""

        yield {
            "ChatOpenAI": mock_chat,
            "ChatPromptTemplate": mock_prompt,
            "MessagesPlaceholder": mock_msgs,
            "create_react_agent": mock_react,
            "Tool": mock_tool,
            "SecretStr": mock_secret,
            "profile_memory": mock_profile,
            "checkpointer": mock_checkpointer,
            "knowledge_search": mock_know,
            "web_search": mock_web,
            "calculator": mock_calc,
        }


# ---------------------------------------------------------------------------
# Tests: create_agent
# ---------------------------------------------------------------------------

class TestCreateAgent:
    """Test create_agent(user_id: str) -> agent"""

    def test_returns_agent_on_success(self, mock_dependencies):
        """正常流程：环境变量存在，所有依赖正常，返回 agent 对象"""
        agent = create_agent("user_001")
        assert agent is not None
        mock_dependencies["create_react_agent"].assert_called_once()

    def test_passes_correct_model_to_chat_openai(self, mock_dependencies):
        """验证 ChatOpenAI 使用正确的 model / base_url / temperature"""
        create_agent("user_001")
        call_kwargs = mock_dependencies["ChatOpenAI"].call_args.kwargs
        assert call_kwargs["model"] == "deepseek-chat"
        assert call_kwargs["base_url"] == "https://api.deepseek.com/v1"
        assert call_kwargs["temperature"] == 0.3

    def test_passes_api_key_via_secret_str(self, mock_dependencies):
        """验证 api_key 通过 SecretStr 包装后传入 ChatOpenAI"""
        create_agent("user_001")
        call_kwargs = mock_dependencies["ChatOpenAI"].call_args.kwargs
        # api_key should be the value returned by SecretStr mock
        assert "api_key" in call_kwargs

    def test_creates_three_tools(self, mock_dependencies):
        """验证创建了 3 个 Tool：knowledge_search, web_search, calculator"""
        create_agent("user_001")
        tool_calls = mock_dependencies["Tool"].call_args_list
        tool_names = [c.kwargs["name"] for c in tool_calls]
        assert tool_names == ["knowledge_search", "web_search", "calculator"]

    def test_tools_use_correct_functions(self, mock_dependencies):
        """验证每个 Tool 绑定了正确的 func"""
        create_agent("user_001")
        tool_calls = mock_dependencies["Tool"].call_args_list
        funcs = [c.kwargs["func"] for c in tool_calls]
        assert funcs[0] is mock_dependencies["knowledge_search"]
        assert funcs[1] is mock_dependencies["web_search"]
        assert funcs[2] is mock_dependencies["calculator"]

    def test_queries_user_profile_for_context(self, mock_dependencies):
        """验证调用了 _profile_memory.get_context_prompt(user_id)"""
        create_agent("user_xyz")
        mock_dependencies["profile_memory"].get_context_prompt.assert_called_once_with("user_xyz")

    def test_injects_user_context_into_system_prompt(self, mock_dependencies):
        """当用户画像有数据时，system prompt 包含用户信息"""
        mock_dependencies["profile_memory"].get_context_prompt.return_value = (
            "用户姓名：张三\n用户偏好：Python, 机器学习"
        )
        create_agent("user_001")
        from_messages_call = mock_dependencies["ChatPromptTemplate"].from_messages.call_args
        messages = from_messages_call[0][0]
        system_msg = messages[0][1]  # ("system", "...")
        assert "张三" in system_msg
        assert "Python" in system_msg

    def test_default_prompt_when_no_user_profile(self, mock_dependencies):
        """当用户画像为空时，system prompt 包含默认提示"""
        mock_dependencies["profile_memory"].get_context_prompt.return_value = ""
        create_agent("user_001")
        from_messages_call = mock_dependencies["ChatPromptTemplate"].from_messages.call_args
        messages = from_messages_call[0][0]
        system_msg = messages[0][1]
        assert "暂无用户信息" in system_msg

    def test_prompt_includes_messages_placeholder(self, mock_dependencies):
        """验证 prompt 包含 MessagesPlaceholder(messages)"""
        create_agent("user_001")
        from_messages_call = mock_dependencies["ChatPromptTemplate"].from_messages.call_args
        messages = from_messages_call[0][0]
        # The second element should be a MessagesPlaceholder
        assert len(messages) == 2
        # MessagesPlaceholder was called with variable_name="messages"
        mock_dependencies["MessagesPlaceholder"].assert_called_with(variable_name="messages")

    def test_create_react_agent_receives_correct_args(self, mock_dependencies):
        """验证 create_react_agent 收到正确的 model/tools/prompt/checkpointer"""
        create_agent("user_001")
        call_kwargs = mock_dependencies["create_react_agent"].call_args.kwargs
        assert call_kwargs["model"] is not None
        assert "tools" in call_kwargs
        assert "prompt" in call_kwargs
        assert call_kwargs["checkpointer"] is mock_dependencies["checkpointer"]

    def test_prompt_contains_tool_descriptions(self, mock_dependencies):
        """验证 system prompt 包含工具描述关键词"""
        create_agent("user_001")
        from_messages_call = mock_dependencies["ChatPromptTemplate"].from_messages.call_args
        messages = from_messages_call[0][0]
        system_msg = messages[0][1]
        assert "knowledge_search" in system_msg
        assert "web_search" in system_msg
        assert "calculator" in system_msg

    def test_prompt_contains_important_rules(self, mock_dependencies):
        """验证 system prompt 包含重要规则"""
        create_agent("user_001")
        from_messages_call = mock_dependencies["ChatPromptTemplate"].from_messages.call_args
        messages = from_messages_call[0][0]
        system_msg = messages[0][1]
        assert "必须" in system_msg
        assert "calculator" in system_msg
        assert "knowledge_search" in system_msg


# ---------------------------------------------------------------------------
# Tests: missing API key
# ---------------------------------------------------------------------------

class TestCreateAgentMissingApiKey:
    """Test create_agent when DEEPSEEK_API_KEY is missing"""

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_value_error_when_api_key_missing(self):
        """环境变量 DEEPSEEK_API_KEY 未设置时抛出 ValueError"""
        with pytest.raises(ValueError, match="请设置环境变量 DEEPSEEK_API_KEY"):
            create_agent("user_001")


# ---------------------------------------------------------------------------
# Tests: get_agent (singleton wrapper)
# ---------------------------------------------------------------------------

class TestGetAgent:
    """Test get_agent(user_id) which wraps create_agent with caching"""

    def test_get_agent_returns_agent(self, mock_dependencies):
        """get_agent 应该返回 agent 对象"""
        # Reset the module-level cache
        import app.core.agent as agent_mod
        agent_mod._agent_executor = None

        agent = get_agent("user_001")
        assert agent is not None

    def test_get_agent_caches_result(self, mock_dependencies):
        """get_agent 第二次调用时使用缓存，不再创建新 agent"""
        import app.core.agent as agent_mod
        agent_mod._agent_executor = None

        mock_dependencies["create_react_agent"].reset_mock()

        first = get_agent("user_001")
        second = get_agent("user_002")

        assert first is second
        # create_react_agent 只被调用一次（第一次创建时）
        assert mock_dependencies["create_react_agent"].call_count == 1
