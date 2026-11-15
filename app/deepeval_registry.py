"""Explicit applicability and gate status for every framework metric."""
from dataclasses import dataclass

from app import deepeval_runtime
from deepeval.metrics import (FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric,
                              ContextualRecallMetric, ToolCorrectnessMetric, ConversationCompletenessMetric,
                              KnowledgeRetentionMetric)
from deepeval.test_case import ToolCallParams

from app.deepeval_metrics import (RuleMetric, RetrievalMetric, RAGPolicyMetric, SecurityPolicyMetric,
                                 rag_structure_checks, route_checks, proposal_checks, recorded_checks,
                                 boundary_checks, database_checks, security_checks, conversation_scene_checks)
from app.llm_judge import LLMJudge
from app.security_judge import SecurityLLMJudge


@dataclass
class MetricSpec:
    name: str
    metric: object
    mandatory: bool = True
    kind: str = "rule"


class LazyProjectJudge:
    def __init__(self, model, kind):
        self.chat_model, self.kind = model, kind

    def evaluate(self, *args, **kwargs):
        judge = LLMJudge if self.kind == "rag" else SecurityLLMJudge
        return judge(self.chat_model._client()).evaluate(*args, **kwargs)


def metric_specs(item, models, *, semantic=False):
    suite, meta = item.suite, item.test_case.metadata
    specs = []
    def rule(name, check):
        specs.append(MetricSpec(name, RuleMetric(name, check)))
    if suite == "rag":
        rule("RAGStructure", rag_structure_checks)
        if meta["expected"]["answerable"]:
            specs.extend(MetricSpec(name, RetrievalMetric(name), False) for name in ("Recall@5", "MRR", "EvidenceRecall@5"))
    elif suite in {"routing", "business-routing"}:
        rule("RouteAccuracy", route_checks)
    elif suite == "tools":
        rule("ModelProposals", proposal_checks)
        rule("SystemBoundary", boundary_checks)
        rule("DatabaseState", database_checks)
        if meta["record"]["phase"] == "integration":
            rule("ExecutionContract", lambda m: m["record"]["execution_checks"])
            specs.append(MetricSpec("NativeToolCorrectness", ToolCorrectnessMetric(
                model=models["rag"], threshold=1.0, should_exact_match=True, should_consider_ordering=True,
                evaluation_params=[ToolCallParams.INPUT_PARAMETERS], include_reason=False,
                async_mode=False, eval_mode="llm"), kind="native"))
    elif suite == "business":
        rule("BusinessContract", recorded_checks)
        rule("DatabaseState", database_checks)
        rule("SystemBoundary", boundary_checks)
    elif suite == "security":
        rule("SecurityStructureAndPermissions", security_checks)
    elif suite == "conversation-scene":
        rule("ConversationStateAndTrajectory", conversation_scene_checks)
    if not semantic:
        if item.key.startswith("rag/direct/"):
            for spec in specs: spec.mandatory = False
        return specs
    common = {"model": models["rag"], "async_mode": False, "include_reason": False, "eval_mode": "llm"}
    if suite == "calibration-native":
        classes = {"NativeFaithfulness": FaithfulnessMetric, "NativeAnswerRelevancy": AnswerRelevancyMetric,
                   "NativeContextualPrecision": ContextualPrecisionMetric, "NativeContextualRecall": ContextualRecallMetric,
                   "NativeConversationCompleteness": ConversationCompletenessMetric, "NativeKnowledgeRetention": KnowledgeRetentionMetric}
        name = meta["expected"]["metric"]
        specs.append(MetricSpec(name, classes[name](threshold=.8, **common), False, "native"))
    elif suite in {"rag", "calibration-rag"}:
        # Native definitions differ from project policy. Until separately
        # qualified by calibration they are diagnostic, not silently substituted
        # for the existing citation/answerability release contract.
        specs.extend([MetricSpec("NativeFaithfulness", FaithfulnessMetric(threshold=.8, **common), False, "native"),
                      MetricSpec("NativeAnswerRelevancy", AnswerRelevancyMetric(threshold=.8, **common), False, "native")])
        if suite == "rag" and meta["expected"]["answerable"]:
            specs.extend([MetricSpec("NativeContextualPrecision", ContextualPrecisionMetric(threshold=.8, **common), False, "native"),
                          MetricSpec("NativeContextualRecall", ContextualRecallMetric(threshold=.8, **common), False, "native")])
        specs.append(MetricSpec("ProjectRAGPolicy", RAGPolicyMetric(LazyProjectJudge(models["rag"], "rag")),
                                suite == "rag", "policy"))
    elif suite in {"security", "calibration-security"}:
        specs.append(MetricSpec("ProjectSecurityPolicy", SecurityPolicyMetric(LazyProjectJudge(models["security"], "security")),
                                suite == "security", "policy"))
    elif suite == "conversation":
        specs.extend([MetricSpec("NativeConversationCompleteness", ConversationCompletenessMetric(threshold=.8, **common), False, "native"),
                      MetricSpec("NativeKnowledgeRetention", KnowledgeRetentionMetric(threshold=.8, **common), False, "native")])
    if item.key.startswith("rag/direct/"):
        for spec in specs: spec.mandatory = False
    return specs
