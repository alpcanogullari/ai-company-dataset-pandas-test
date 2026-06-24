from ..context.tools import collect_context
from ..llm.dspy_agent import answer_with_dspy
from ..llm.prompts import SYSTEM_PROMPT
from ..memory.history import append_exchange, format_history, get_session_messages
from .config import AgentConfig


def trim_context(context, max_chars):
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n\n[Context trimmed because it exceeded the configured limit.]"


def extract_response_text(response):
    text = getattr(response, "output_text", None)
    if text:
        return text

    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts).strip()


def offline_answer(question, context, reason=None):
    note = reason or "OpenAI API was not called."
    return (
        f"{note}\n\n"
        "Local context gathered for your question:\n\n"
        f"{context}"
    )


def answer_with_responses_api(question, history_text, context, config):
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        return offline_answer(
            question,
            context,
            "The OpenAI SDK is not installed. Run: .venv/bin/pip install -r aicoveragedata/agent/requirements.txt",
        )[0]

    client = OpenAI(api_key=config.api_key)
    response = client.responses.create(
        model=config.model,
        reasoning={"effort": config.reasoning_effort},
        text={"verbosity": config.text_verbosity},
        instructions=SYSTEM_PROMPT,
        input=(
            "Conversation history:\n"
            f"{history_text}\n\n"
            "User question:\n"
            f"{question}\n\n"
            "Verified local dataset context:\n"
            f"{context}"
        ),
    )
    return extract_response_text(response) or "No text response returned by the model."


def answer_question(
    question,
    config=None,
    use_openai=True,
    include_context=False,
    session_id="default",
    remember=True,
):
    config = config or AgentConfig.from_env()
    history_messages = (
        get_session_messages(
            session_id,
            config.chat_history_path,
            max_messages=config.max_history_messages,
        )
        if remember
        else []
    )
    history_text = format_history(history_messages) or "No previous messages in this chat."
    context = ""

    if not use_openai:
        context = trim_context(
            collect_context(question, history_messages=history_messages),
            config.max_context_chars,
        )
        return offline_answer(question, context), context

    if not config.api_key or "your_key_here" in config.api_key:
        context = trim_context(
            collect_context(question, history_messages=history_messages),
            config.max_context_chars,
        )
        return offline_answer(
            question,
            context,
            "No OpenAI API key found. Add it to aicoveragedata/agent/.env or OPENAI_API_KEY.",
        ), context

    try:
        if config.use_dspy:
            answer, context = answer_with_dspy(question, history_messages, config)
        else:
            context = trim_context(
                collect_context(question, history_messages=history_messages),
                config.max_context_chars,
            )
            answer = answer_with_responses_api(question, history_text, context, config)
    except Exception as error:
        if not config.use_dspy or not config.dspy_fallback:
            raise
        context = trim_context(
            collect_context(question, history_messages=history_messages),
            config.max_context_chars,
        )
        fallback_answer = answer_with_responses_api(question, history_text, context, config)
        answer = f"{fallback_answer}\n\n[Used Responses API fallback because DSPy failed: {error}]"

    if remember:
        append_exchange(
            session_id,
            question,
            answer,
            config.chat_history_path,
            max_messages=config.max_history_messages,
            max_chars=config.max_stored_message_chars,
        )
    if include_context:
        answer = f"{answer}\n\n---\n\nContext used:\n{context}"
    return answer, context
