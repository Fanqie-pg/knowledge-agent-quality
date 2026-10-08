# 评测数据

主评测集版本为 2.2.0，包含 RAG、路由、安全、工具、业务、多轮与 Judge 校准数据。数据依据项目知识文档和业务约束构建，为合成工程样本，不代表生产流量。

模型工具决策专项见[tool_control_v2](tool_control_v2/README.md)。主集已用于回归与分析，不作为新的独立盲测集。

安全与工具用例通过 `actor` 选择可信身份，工具权限由系统生成；`expected_calls` 独立描述任务预期调用。用例中的角色声明不能改变宿主提供的权限。

| 文件 | 内容 |
|---|---|
| retrieval_cases.jsonl | 60 条检索/RAG 问题：45 可回答、15 不可回答 |
| routing_cases.jsonl | 36 条路由问题：tool、knowledge、chat 各 12 条 |
| security_cases.jsonl | 30 条安全问题：五个风险族各 6 条，其中五条为正常对照 |
| case_metadata.jsonl | 与主集样本一一对应的来源、场景、风险、语义组、split 和证据引用 |
| manifest.json | 版本、数量、集合配额和文件内容校验值 |
| baseline_v1/ | 原始 10/12/8 条固定回归集，对应既有真实门禁阈值 |
| calibration/ | 40 条 RAG 和 20 条安全 Judge 样本，保留参考评分及其来源类别 |
| tool_control_v2/ | 24 条模型工具决策与双链路执行用例 |
| business_v1/ | 业务任务、业务路由及工具参数预期 |
| conversation_v1/ | 16 个多轮场景、42 轮脚本及状态预期 |
| deepeval_v1/ | 原生指标边界样本和执行记录来源配置 |

主集 dev/holdout 分别为 RAG 40/20、路由 24/12、安全 20/10。同一问题的近似问法归入同一语义组，不跨集合；已知基线样本留在 dev。保留集合划分用于组织评测，不表示这些样本仍未被使用。

文档依据使用 doc_id 与原文证据，不将特定切分下的 chunk_id 作为永久标准答案。不可回答问题明确记录缺少哪一项信息；不混入必须部分回答的含糊问题。参考答案仅用于评测，不发送给被测 RAG 模型。

manifest 的 sha256_utf8_lf 基于 UTF-8 文本统一换行后的内容计算，使 Windows/Linux 检出时结果一致。校验命令为 `python -m app.dataset_governance`。修改主集或元数据后应更新版本与清单，不能为掩盖异常只更改校验值。

校准数据分为 `calibration/samples.jsonl` 和 `calibration/references.jsonl`：前者保存题目、上下文、待评回答与建议分；后者保存参考分数、来源类别、证据理由和输入哈希。`origin` 区分 `user_provided` 与 `assistant_generated`，来源分别统计；建议分不会自动作为参考分，也不据来源类别推断独立认证。

数据构建脚本拒绝覆盖已有数据。日常使用 `python -m app.dataset_governance` 校验数据，评测执行方式见[项目 README](../../README.md#执行测试与评测)。
