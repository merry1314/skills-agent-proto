"""Live E2E tests against real model providers.

These tests are opt-in because they make real network calls and consume tokens.
Enable them by setting `SKILLS_RUN_LIVE_TESTS=1`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from langchain_skills.agent import LangChainSkillsAgent


LIVE_TEST_PROMPT = "Return exactly the single word PING in uppercase."
LIVE_TOOL_PROMPT = "Use the bash tool to run pwd, then reply with the resulting absolute path only."


def _live_tests_enabled() -> bool:
    return os.getenv("SKILLS_RUN_LIVE_TESTS", "").lower() in {"1", "true", "yes"}


def _build_provider_env(provider: str) -> dict[str, str] | None:
    shared_key = (
        os.getenv("MODEL_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_AUTH_TOKEN")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_AUTH_TOKEN")
    )

    if provider == "anthropic":
        model = (
            os.getenv("MODEL_NAME")
            if os.getenv("MODEL_PROVIDER") == "anthropic"
            else None
        ) or os.getenv("ANTHROPIC_MODEL") or os.getenv("CLAUDE_MODEL")
        base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("MODEL_BASE_URL")
    else:
        model = (
            os.getenv("OPENAI_MODEL")
            or (
                os.getenv("MODEL_NAME")
                if os.getenv("MODEL_PROVIDER") == "openai"
                else None
            )
            or "gpt-5.4"
        )
        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.jiekou.ai/openai"

    if not shared_key or not model or not base_url:
        return None

    return {
        "MODEL_PROVIDER": provider,
        "MODEL_NAME": model,
        "MODEL_API_KEY": shared_key,
        "MODEL_BASE_URL": base_url,
    }


def _stream_text(events: Iterator[dict]) -> tuple[str, bool]:
    chunks: list[str] = []
    done_seen = False

    for event in events:
        if event["type"] == "text":
            chunks.append(event["content"])
        elif event["type"] == "done":
            done_seen = True

    return "".join(chunks).strip(), done_seen


@pytest.mark.skipif(not _live_tests_enabled(), reason="set SKILLS_RUN_LIVE_TESTS=1 to enable live E2E tests")
@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_live_agent_invoke_and_stream(provider: str):
    provider_env = _build_provider_env(provider)
    if provider_env is None:
        pytest.skip(f"missing live credentials for provider={provider}")

    previous = {key: os.environ.get(key) for key in provider_env}
    os.environ.update(provider_env)
    try:
        agent = LangChainSkillsAgent(skill_paths=[], enable_thinking=True)
        assert agent.enable_thinking is True

        invoke_result = agent.invoke(LIVE_TEST_PROMPT, thread_id=f"live-{provider}-invoke")
        invoke_text = agent.get_last_response(invoke_result).strip().upper()
        assert "PING" in invoke_text
        last_message = invoke_result["messages"][-1]
        usage_metadata = getattr(last_message, "usage_metadata", {}) or {}

        if provider == "openai":
            output_details = usage_metadata.get("output_token_details", {})
            assert output_details.get("reasoning", 0) > 0

        stream_events = list(agent.stream_events(LIVE_TEST_PROMPT, thread_id=f"live-{provider}-stream"))
        stream_text, done_seen = _stream_text(iter(stream_events))
        assert done_seen is True
        assert "PING" in stream_text.upper()
        if provider == "openai":
            thinking_events = [event for event in stream_events if event["type"] == "thinking"]
            assert any("reasoning enabled" in event["content"] for event in thinking_events)

        tool_events = list(
            agent.stream_events(LIVE_TOOL_PROMPT, thread_id=f"live-{provider}-tool")
        )
        tool_call_count = sum(1 for event in tool_events if event["type"] == "tool_call")
        tool_result_count = sum(1 for event in tool_events if event["type"] == "tool_result")
        tool_response = "".join(
            event.get("content", "") for event in tool_events if event["type"] == "text"
        ).strip()

        assert tool_call_count >= 1
        assert tool_result_count >= 1
        assert tool_response == os.getcwd()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
