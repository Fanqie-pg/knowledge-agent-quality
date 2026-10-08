"""Exercise the actual pytest hooks in a tiny isolated project, without LLM imports."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("has_key, message", [(True, "门禁用例缺失"), (False, "DEEPSEEK_API_KEY")])
def test_strict_gate_fails_before_execution_for_missing_prerequisites(tmp_path, has_key, message):
    original = Path(__file__).with_name("conftest.py")
    (tmp_path / "conftest.py").write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    evaluation: live test\n    quality_gate: gate\n", encoding="utf-8")
    (tmp_path / "test_subset.py").write_text("import pytest\npytestmark=[pytest.mark.evaluation,pytest.mark.quality_gate]\ndef test_live():\n    raise AssertionError('BODY_MUST_NOT_RUN')\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHON_DOTENV_DISABLED"] = "1"
    if has_key:
        env["DEEPSEEK_API_KEY"] = "unit-test-placeholder"
    else:
        env.pop("DEEPSEEK_API_KEY", None)
    result = subprocess.run([sys.executable, "-m", "pytest", "test_subset.py", "--run-evaluation", "--strict-quality-gate", "--collect-only", "-q"],
                            cwd=tmp_path, env=env, text=True, encoding="utf-8", capture_output=True, timeout=30)
    assert result.returncode == 4, result.stdout + result.stderr
    assert message in result.stdout + result.stderr
    assert "BODY_MUST_NOT_RUN" not in result.stdout + result.stderr


def test_default_collection_ignores_generated_report_projects(tmp_path):
    config = Path(__file__).resolve().parents[1] / "pytest.ini"
    (tmp_path / "pytest.ini").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "tests/test_good.py").write_text("def test_good(): pass\n", encoding="utf-8")
    (tmp_path / "reports/test_poison.py").write_text("raise AssertionError('SHOULD_NOT_COLLECT')\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=tmp_path, env=env,
                            text=True, encoding="utf-8", capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_good" in result.stdout and "test_poison" not in result.stdout
