import httpx
from openai import APIStatusError
import pytest

from app.evaluation_campaign import provider_error_details
from scripts.verify_quality_fixes import parallel_run


pytestmark = pytest.mark.unit


def test_generic_provider_code_still_retains_insufficient_balance_http_status():
    request = httpx.Request("POST", "https://api.example.test/chat/completions")
    response = httpx.Response(402, request=request)
    error = APIStatusError("Insufficient Balance", response=response,
                           body={"code": "invalid_request_error", "type": "insufficient_balance_error"})
    details = provider_error_details(error)
    assert details["error_http_status"] == 402
    assert details["provider_error_type"] == "insufficient_balance_error"


def test_permanent_http_errors_stop_additional_batches():
    called, saved = [], []
    def execute(item):
        called.append(item)
        return {"status": "error", "error_http_status": 402}
    parallel_run(list(range(12)), execute, saved.append, workers=4)
    assert len(called) == len(saved) == 4
