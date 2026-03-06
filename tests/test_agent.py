"""Agent provider routing tests."""

from __future__ import annotations

import os
from unittest.mock import Mock, patch

from langchain_skills.agent import LangChainSkillsAgent, resolve_model_config


def test_resolve_model_config_defaults_to_anthropic_legacy_env():
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
            "ANTHROPIC_BASE_URL": "https://api.jiekou.ai/anthropic",
            "CLAUDE_MODEL": "claude-sonnet-4-5-20250929",
        },
        clear=True,
    ):
        config = resolve_model_config()

    assert config.provider == "anthropic"
    assert config.model == "claude-sonnet-4-5-20250929"
    assert config.api_key == "anthropic-token"
    assert config.base_url == "https://api.jiekou.ai/anthropic"
    assert config.supports_extended_thinking is True


def test_resolve_model_config_supports_openai_generic_env():
    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "openai",
            "MODEL_NAME": "gpt-5.4",
            "MODEL_API_KEY": "shared-token",
            "MODEL_BASE_URL": "https://api.jiekou.ai/openai",
        },
        clear=True,
    ):
        config = resolve_model_config()

    assert config.provider == "openai"
    assert config.model == "gpt-5.4"
    assert config.api_key == "shared-token"
    assert config.base_url == "https://api.jiekou.ai/openai"
    assert config.supports_extended_thinking is True


def test_resolve_model_config_supports_openai_provider_specific_env():
    with patch.dict(
        os.environ,
        {
            "OPENAI_MODEL": "gpt-5.4",
            "OPENAI_API_KEY": "openai-token",
            "OPENAI_BASE_URL": "https://api.jiekou.ai/openai",
        },
        clear=True,
    ):
        config = resolve_model_config()

    assert config.provider == "openai"
    assert config.model == "gpt-5.4"
    assert config.api_key == "openai-token"
    assert config.base_url == "https://api.jiekou.ai/openai"


def test_openai_agent_enables_reasoning_effort_kwargs():
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "openai",
            "MODEL_NAME": "gpt-5.4",
            "MODEL_API_KEY": "shared-token",
            "MODEL_BASE_URL": "https://api.jiekou.ai/openai",
        },
        clear=True,
    ), patch("langchain_skills.agent.SkillLoader", return_value=fake_loader), patch(
        "langchain_skills.agent.init_chat_model"
    ) as init_chat_model, patch("langchain_skills.agent.create_agent", return_value=object()):
        init_chat_model.return_value = object()
        agent = LangChainSkillsAgent(enable_thinking=True)

    kwargs = init_chat_model.call_args.kwargs
    assert agent.model_provider == "openai"
    assert agent.enable_thinking is True
    assert kwargs["model_provider"] == "openai"
    assert kwargs["api_key"] == "shared-token"
    assert kwargs["base_url"] == "https://api.jiekou.ai/openai/v1"
    assert kwargs["use_responses_api"] is True
    assert kwargs["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "thinking" not in kwargs


def test_anthropic_agent_keeps_thinking_kwargs():
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_NAME": "claude-sonnet-4-5-20250929",
            "MODEL_API_KEY": "anthropic-token",
            "MODEL_BASE_URL": "https://api.jiekou.ai/anthropic",
        },
        clear=True,
    ), patch("langchain_skills.agent.SkillLoader", return_value=fake_loader), patch(
        "langchain_skills.agent.init_chat_model"
    ) as init_chat_model, patch("langchain_skills.agent.create_agent", return_value=object()):
        init_chat_model.return_value = object()
        agent = LangChainSkillsAgent(enable_thinking=True, thinking_budget=2048)

    kwargs = init_chat_model.call_args.kwargs
    assert agent.model_provider == "anthropic"
    assert agent.enable_thinking is True
    assert kwargs["model_provider"] == "anthropic"
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert kwargs["api_key"] == "anthropic-token"
    assert kwargs["base_url"] == "https://api.jiekou.ai/anthropic"
