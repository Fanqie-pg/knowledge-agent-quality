"""Local-only DeepEval runtime, configured before importing its public API."""
import os

# Agent evaluation is authorized; publishing datasets/traces to another service
# is not part of this integration. Explicit models are supplied to every metric.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ["DEEPEVAL_DISABLE_LEGACY_KEYFILE"] = "1"
os.environ["DEEPEVAL_FILE_SYSTEM"] = "READ_ONLY"
os.environ["CONFIDENT_TRACE_SAMPLE_RATE"] = "0"
os.environ["CONFIDENT_API_KEY"] = ""

from deepeval import evaluate, assert_test  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig  # noqa: E402
from deepeval.metrics import BaseMetric, BaseConversationalMetric  # noqa: E402
from deepeval.test_case import LLMTestCase, ToolCall, Turn, ConversationalTestCase  # noqa: E402

FRAMEWORK_VERSION = "4.3.4"
PROTOCOL_VERSION = "1.1.0-native-json"


def evaluate_local(test_case, metrics):
    """Use the actual framework runner, without hidden cache hits or uploads."""
    return evaluate(test_cases=[test_case], metrics=metrics,
                    async_config=AsyncConfig(run_async=False, max_concurrent=1),
                    display_config=DisplayConfig(show_indicator=False, print_results=False, inspect_after_run=False),
                    cache_config=CacheConfig(write_cache=False, use_cache=False),
                    error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=False))
