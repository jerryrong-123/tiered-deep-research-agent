# Modifications / 改动说明

本项目基于 ApodexAI/FrontierAgent，原始来源提交为
`e4cb2397109c957a2f99c3a5d850b9fc479dbef6`，许可证保留为 Apache-2.0。

## 发布处理

- 公开发布副本的上游历史经过内容过滤；保留原作者、时间、提交拓扑和源码来源，过滤后的 commit ID 与原始 ID 不同。原始来源并未改写。
- `v2.3.2-dev30` 对应先提交的 V2.3.2 实现，随后提交 V2.6.1 + guard v8，最后提交文档并打 `v2.6.1-guard-v8`。
- 从所有待发布历史及两个快照中排除：`tools/golden/`、`benchmarks/frontier_search_bench/`、`benchmarks/frontierchallenge/site/`、`assets/apodex1.1_bench.png`、`docs/tool-result-truncation-ab.md`。这些内容含上游真实题目、参考答案、实验结果或展示数据；主解题运行代码不依赖它们。依赖这些外部基准的测试与适配器不能在缺少数据时用于复现原基准。
- 两个实现提交均将上游 `.github/workflows/` 移至 `.github/disabled-workflows/`，加 `.disabled` 后缀，避免触发原镜像推送、Pages、定时任务或依赖已排除数据的 CI。没有冒充新 CI 已通过。
- 两个实现提交均追加 `.gitignore`：忽略真实 `runtime.env`、`secrets/`、`outputs/`、`.research_receipts/` 与 `QUESTION*.json`。
- 文档提交新增本项目 README、NOTICE、此改动说明、空值运行变量模板；原上游 README 保留至 `docs/UPSTREAM_README.md`。上游说明中对已排除数据/图片的引用只用于追溯，不保证可用。
- 发布包此前已经将 launcher 绝对路径替换成占位符，同步 prepare 锚点，并合成化少量测试文本。这些包装处理与上述发布清理不应描述为历史实测源码逐字节未变。没有改变研究模型、预算或判分标准。
- README 的 200 题数据是项目作者另行确认的名称等价裁定后汇总，不是本包可独立重算的结果；不覆盖 DEV30 的原始冻结成绩。

## 原始源码快照相对上游的清单

下面保留发布包提供的逐文件清单与原始计数，描述的是清理前的两个源码快照。最终发布树的删减、CI 移动和新增文档应同时参考上节，不能把下面的“没有删除”误读为最终发布树没有删除。

### 包内自动清单

- 上游仓库：https://github.com/ApodexAI/FrontierAgent
- 对应的上游 commit：`e4cb2397109c957a2f99c3a5d850b9fc479dbef6`（2026-10-14 15:59:18 +0800，Merge pull request #65 from KennyMcSimpson/fix/download-response-filenames）
- 比较方法：逐文件计算 git blob 哈希，与该 commit 的文件树对比。
- 原始包的排除说明见其 `CHECKS_REPORT.md`（不随本仓库分发）；本次发布删减见上文[发布处理](#发布处理)。

## V2.3.2（`src_v232/`）

共 837 个文件：与上游相同 768，新增 57，修改 12，上游有而这里没有 0。

### 新增（57）

- `README_LOCAL.md`
- `local_profiles/PROGRESSIVE_RESEARCH_V1.md`
- `local_profiles/TIERED_RESEARCH_V1.md`
- `local_profiles/TIERED_RESEARCH_V2.md`
- `local_profiles/convergence_v1.yaml`
- `local_profiles/effort_high_v1.yaml`
- `local_profiles/integrated_v1.yaml`
- `local_profiles/progressive_research_v1.yaml`
- `local_profiles/shared_search_v1.yaml`
- `local_profiles/tiered_light_stage_v1.yaml`
- `local_profiles/tiered_light_stage_v2.yaml`
- `local_profiles/tiered_light_stage_v2_deep.yaml`
- `local_profiles/tiered_research_v1.yaml`
- `local_runner/PRICING_VERIFICATION.json`
- `local_runner/PROFILE.yaml`
- `local_runner/README.md`
- `local_runner/TEMPLATE_ORIGIN.json`
- `local_runner/__init__.py`
- `local_runner/answer_label_guard.py`
- `local_runner/budget_adapter.py`
- `local_runner/launch.ps1`
- `local_runner/prepare.py`
- `local_runner/run_native.py`
- `local_runner/sandbox_registry.py`
- `local_runner/task_supervisor.py`
- `local_runner/test_budget_adapter.py`
- `local_runner/test_linux_adapter.py`
- `local_runner/test_native.py`
- `local_runner/test_verified_pricing.py`
- `local_runner/tiered_router.py`
- `local_runner/tool_classification.py`
- `local_runner/trial_helpers.py`
- `plugins/tools/_archive_copy.py`
- `plugins/tools/_search_contract.py`
- `plugins/tools/_search_run_cache.py`
- `tests/test_convergence_control_v1.py`
- `tests/test_convergence_main_wiring_v1.py`
- `tests/test_convergence_native_lifecycle_v1.py`
- `tests/test_convergence_planning_v1.py`
- `tests/test_e2b_stdin_contract_v1.py`
- `tests/test_local_profiles_v1.py`
- `tests/test_local_runner_progressive_v1.py`
- `tests/test_native_search_reuse_v1.py`
- `tests/test_progressive_main_wiring_v1.py`
- `tests/test_progressive_native_wiring_v1.py`
- `tests/test_search_error_contract_v1.py`
- `tests/test_search_reuse_http_v1.py`
- `tests/test_search_run_cache_v1.py`
- `tests/test_tiered_research_v1.py`
- `tests/test_tiered_research_v2.py`
- `tests/test_tiered_research_v2_3.py`
- `tests/test_tiered_router.py`
- `workflows/agent_team/answer_settlement.py`
- `workflows/agent_team/convergence.py`
- `workflows/agent_team/convergence_cleanup.py`
- `workflows/agent_team/observers/convergence.py`
- `workflows/agent_team/progressive.py`

### 修改（12）

- `plugins/tools/_download_runner.py`
- `plugins/tools/_sandbox.py`
- `plugins/tools/assign_task.py`
- `plugins/tools/collect_reports.py`
- `plugins/tools/finalize_answer.py`
- `plugins/tools/web_fetch_aligned.py`
- `plugins/tools/web_search.py`
- `workflows/agent_team/nodes/main_agent.py`
- `workflows/agent_team/observers/auto_fan_in.py`
- `workflows/agent_team/observers/bare_text_finalize.py`
- `workflows/agent_team/observers/planning_gate.py`
- `workflows/agent_team/prompts.py`

### 上游有而这里没有（0）

- 无

## 最新版（`src_latest/`）

共 848 个文件：与上游相同 764，新增 68，修改 16，上游有而这里没有 0。

### 新增（68）

- `README_LOCAL.md`
- `local_profiles/PROGRESSIVE_RESEARCH_V1.md`
- `local_profiles/TIERED_RESEARCH_V1.md`
- `local_profiles/TIERED_RESEARCH_V2.md`
- `local_profiles/convergence_v1.yaml`
- `local_profiles/effort_high_v1.yaml`
- `local_profiles/graph_fallback_v1.yaml`
- `local_profiles/integrated_v1.yaml`
- `local_profiles/progressive_research_v1.yaml`
- `local_profiles/shared_search_v1.yaml`
- `local_profiles/tiered_light_stage_v1.yaml`
- `local_profiles/tiered_light_stage_v2.yaml`
- `local_profiles/tiered_light_stage_v2_deep.yaml`
- `local_profiles/tiered_research_v1.yaml`
- `local_runner/PRICING_VERIFICATION.json`
- `local_runner/PROFILE.yaml`
- `local_runner/README.md`
- `local_runner/TEMPLATE_ORIGIN.json`
- `local_runner/__init__.py`
- `local_runner/answer_label_guard.py`
- `local_runner/budget_adapter.py`
- `local_runner/launch.ps1`
- `local_runner/prepare.py`
- `local_runner/run_native.py`
- `local_runner/sandbox_registry.py`
- `local_runner/task_supervisor.py`
- `local_runner/test_budget_adapter.py`
- `local_runner/test_linux_adapter.py`
- `local_runner/test_native.py`
- `local_runner/test_verified_pricing.py`
- `local_runner/tiered_router.py`
- `local_runner/tool_classification.py`
- `local_runner/trial_helpers.py`
- `plugins/tools/_answer_leads.py`
- `plugins/tools/_archive_copy.py`
- `plugins/tools/_search_contract.py`
- `plugins/tools/_search_run_cache.py`
- `plugins/tools/calculate.py`
- `tests/test_convergence_control_v1.py`
- `tests/test_convergence_main_wiring_v1.py`
- `tests/test_convergence_native_lifecycle_v1.py`
- `tests/test_convergence_planning_v1.py`
- `tests/test_e2b_stdin_contract_v1.py`
- `tests/test_graph_fallback.py`
- `tests/test_graph_fallback_controller.py`
- `tests/test_graph_fallback_node.py`
- `tests/test_local_profiles_v1.py`
- `tests/test_local_runner_progressive_v1.py`
- `tests/test_native_search_reuse_v1.py`
- `tests/test_progressive_main_wiring_v1.py`
- `tests/test_progressive_native_wiring_v1.py`
- `tests/test_search_error_contract_v1.py`
- `tests/test_search_reuse_http_v1.py`
- `tests/test_search_run_cache_v1.py`
- `tests/test_tiered_research_v1.py`
- `tests/test_tiered_research_v2.py`
- `tests/test_tiered_research_v2_3.py`
- `tests/test_tiered_research_v2_4.py`
- `tests/test_tiered_router.py`
- `workflows/agent_team/answer_settlement.py`
- `workflows/agent_team/convergence.py`
- `workflows/agent_team/convergence_cleanup.py`
- `workflows/agent_team/graph_fallback.py`
- `workflows/agent_team/graph_fallback_controller.py`
- `workflows/agent_team/graph_fallback_node.py`
- `workflows/agent_team/graph_fallback_prompts.py`
- `workflows/agent_team/observers/convergence.py`
- `workflows/agent_team/progressive.py`

### 修改（16）

- `plugins/tools/__init__.py`
- `plugins/tools/_download_runner.py`
- `plugins/tools/_sandbox.py`
- `plugins/tools/assign_task.py`
- `plugins/tools/collect_reports.py`
- `plugins/tools/finalize_answer.py`
- `plugins/tools/meta.py`
- `plugins/tools/web_fetch_aligned.py`
- `plugins/tools/web_search.py`
- `tests/test_benchmark_kernel_bootstrap.py`
- `tests/test_tool_registry.py`
- `workflows/agent_team/nodes/main_agent.py`
- `workflows/agent_team/observers/auto_fan_in.py`
- `workflows/agent_team/observers/bare_text_finalize.py`
- `workflows/agent_team/observers/planning_gate.py`
- `workflows/agent_team/prompts.py`

### 上游有而这里没有（0）

- 无
