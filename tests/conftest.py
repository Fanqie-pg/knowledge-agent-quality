import pytest
import os
from dotenv import load_dotenv


def pytest_addoption(parser):
    parser.addoption("--trace-output", default=None, help="保存真实评测的业务事件Trace")
    parser.addoption("--strict-quality-gate", action="store_true", default=False,
                     help="要求完整真实模型门禁，关键测试不得跳过")
    parser.addoption(
        "--run-evaluation",
        action="store_true",
        default=False,
        help="运行会调用真实大模型的评测测试",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-evaluation"):
        if any("evaluation" in item.keywords for item in items):
            load_dotenv()
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise pytest.UsageError("真实模型评测缺少DEEPSEEK_API_KEY，不能以跳过冒充门禁通过")
        if config.getoption("--strict-quality-gate"):
            required = ("retrieval_quality_gate", "intent_router_evaluation", "rag_end_to_end", "security_quality_gate", "mcp_unified_agent")
            missing = [name for name in required if not any(name in item.nodeid for item in items)]
            if missing:
                raise pytest.UsageError("门禁用例缺失：" + ", ".join(missing))
        return

    if config.getoption("--strict-quality-gate"):
        raise pytest.UsageError("--strict-quality-gate需要--run-evaluation")

    skip_evaluation = pytest.mark.skip(
        reason="需要使用 --run-evaluation 才能运行真实大模型评测"
    )

    for item in items:
        if "evaluation" in item.keywords:
            item.add_marker(skip_evaluation)


def pytest_runtest_logreport(report):
    if report.skipped:
        _skipped.append(report.nodeid)


_skipped = []


@pytest.fixture
def seeded_business_repository(tmp_path):
    from app.business_repository import SQLiteBusinessRepository
    repository = SQLiteBusinessRepository(tmp_path / "business.sqlite3")
    repository.seed_demo()
    return repository


@pytest.fixture
def trusted_hr_context():
    from app.identity import load_user_context
    return load_user_context("hr-test")


@pytest.fixture(autouse=True)
def record_evaluation_trace(request):
    path = request.config.getoption("--trace-output")
    if not path or "evaluation" not in request.node.keywords:
        yield
        return
    from app.tracing import capture_trace
    from app.evaluation_records import append_jsonl
    with capture_trace() as events:
        try:
            yield
        finally:
            append_jsonl(path, {"nodeid": request.node.nodeid, "events": events})


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--strict-quality-gate") and (not session.testscollected or _skipped):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
