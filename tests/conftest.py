import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-evaluation",
        action="store_true",
        default=False,
        help="运行会调用真实大模型的评测测试",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-evaluation"):
        return

    skip_evaluation = pytest.mark.skip(
        reason="需要使用 --run-evaluation 才能运行真实大模型评测"
    )

    for item in items:
        if "evaluation" in item.keywords:
            item.add_marker(skip_evaluation)