from types import SimpleNamespace

from app.conversation_agent import ConversationTurnResult
from app.conversation_evaluation import grade_turn, load_cases


def test_story_dataset_has_valid_references_and_boundary_coverage():
    data = load_cases()
    assert len(data["cases"]) == 16
    assert sum(len(c["turns"]) for c in data["cases"]) == 42
    categories = {c["category"] for c in data["cases"]}
    assert {"dependent_steps", "slot_filling", "session_isolation", "identity_and_memory_injection",
            "durable_restart", "rag_to_tool_across_turns", "cross_user_and_hr"} <= categories


def test_backend_denial_is_not_counted_as_correct_model_decision():
    result = ConversationTurnResult("s", "t", "没有权限", decisions=[{"tool_calls": [{"name": "get_leave_balance"}]}],
                                    execution_records=[{"tool_name": "get_leave_balance", "invoked": True,
                                                        "status": "error", "business_authorized": False}])
    checks = grade_turn({"tools": []}, result, None)
    assert not checks["model_tool_choices"] and not checks["actual_tools"] and not checks["permission_boundary"]


def test_guessed_parallel_chain_fails_dependency_constraint():
    result = ConversationTurnResult("s", "t", "32", decisions=[{"tool_calls": [{"name": "calculate"}, {"name": "calculate"}]}],
                                    execution_records=[{"tool_name": "calculate", "invoked": True, "status": "success", "model_step": 1}] * 2)
    checks = grade_turn({"tools": ["calculate", "calculate"], "dependent_steps": True}, result, None)
    assert checks["model_tool_choices"] and not checks["dependent_model_steps"]


def test_cached_answer_must_not_consume_provider_call():
    checks = grade_turn({"tools": [], "cached": True}, ConversationTurnResult("s", "t", "96", cached=True),
                        None, new_provider_calls=1)
    assert not checks["cached_no_provider_call"]
