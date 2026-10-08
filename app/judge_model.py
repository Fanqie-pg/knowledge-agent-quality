"""Independent, role-specific Qwen judges; Agent credentials stay separate."""
import os
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from app.dataset_governance import ROOT
from app.resilient_model import RetryingChatModel
from app.tracing import emit


MODELS = {"rag": "qwen3.7-plus-2026-05-26", "security": "qwen3.8-max-0902"}


def judge_model_config(suite="rag"):
    if suite not in MODELS:
        raise ValueError("Unknown Judge suite")
    load_dotenv(ROOT / ".env")
    base = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    url = urlsplit(base)
    if url.scheme != "https" or not (url.hostname or "").endswith(".aliyuncs.com") or url.username or url.query or url.fragment:
        raise ValueError("QWEN_BASE_URL must be an official HTTPS Aliyun endpoint")
    return {"provider": "aliyun_dashscope", "suite": suite, "model": MODELS[suite],
            "api": "chat_completions", "base_url": base, "enable_thinking": True,
            "thinking_budget": 2048, "temperature": 0, "max_tokens": 4096,
            "json_strategy": "prompt_and_pydantic", "timeout": 120, "max_attempts": 2}


class QwenJudgeModel:
    def __init__(self, client, suite="rag"):
        self.client = client
        self.config = judge_model_config(suite)

    def invoke(self, messages):
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        inputs = [{"role": roles[message.type], "content": message.content} for message in messages]
        config = self.config
        response = self.client.chat.completions.create(
            model=config["model"], messages=inputs, temperature=config["temperature"],
            extra_body={"enable_thinking": config["enable_thinking"], "thinking_budget": config["thinking_budget"]},
            max_tokens=config["max_tokens"],
        )
        usage = response.usage
        metadata = {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens} if usage else None
        raw_usage = usage.model_dump() if usage else None
        choice = response.choices[0] if response.choices else None
        finish = choice.finish_reason if choice else None
        emit("judge_provider", "response", response_id=response.id, model=response.model,
             requested_model=config["model"], suite=config["suite"], finish_reason=finish,
             usage=metadata, raw_usage=raw_usage, enable_thinking=config["enable_thinking"], thinking_budget=config["thinking_budget"])
        if not choice or finish != "stop" or not choice.message.content:
            raise ValueError("Judge response incomplete or empty")
        # reasoning_content is not the answer. Parsers validate only final content.
        return AIMessage(content=choice.message.content, usage_metadata=metadata,
                         response_metadata={"id": response.id, "model_name": response.model,
                                            "suite": config["suite"], "finish_reason": finish})


def build_judge_model(suite="rag"):
    config = judge_model_config(suite)
    # QWENT_API_KEY is the existing project variable; retain it as an explicit alias.
    key = next((os.getenv(k) for k in ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "QWENT_API_KEY") if os.getenv(k)), None)
    if not key:
        raise RuntimeError("QWEN_API_KEY未配置（也支持DASHSCOPE_API_KEY、QWENT_API_KEY）")
    client = OpenAI(api_key=key, base_url=config["base_url"], timeout=config["timeout"], max_retries=0)
    model = RetryingChatModel(QwenJudgeModel(client, suite), max_attempts=config["max_attempts"],
                              retry_delay_seconds=1,
                              retryable_exceptions=(TimeoutError, ConnectionError, APITimeoutError,
                                                    APIConnectionError, RateLimitError, InternalServerError))
    if os.getenv("EVALUATION_MAX_CALLS"):
        from app.evaluation_records import CallBudget, attach_budget
        model = attach_budget(model, CallBudget(os.environ["EVALUATION_CALL_LEDGER"], int(os.environ["EVALUATION_MAX_CALLS"])))
    return model
