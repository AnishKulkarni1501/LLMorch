# MCPorch Benchmark Report

Generated from 30 normal paired tasks per system plus separate failure cases.

## Results

| Metric | Direct API baseline | MCPorch orchestrator |
|---|---:|---:|
| Task completion rate | 100.0% | 100.0% |
| Correct completion rate | 100.0% | 100.0% |
| Recovery success rate | N/A | N/A |
| End-to-end latency (mean) | 0.033913 s | 0.033845 s |
| End-to-end latency (median) | 0.030948 s | 0.030867 s |
| End-to-end latency (p95) | 0.062132 s | 0.062464 s |
| Tool invocations | 33 | 33 |
| Mean tool calls per task | 1.100 | 1.100 |
| Validation failure rate | N/A | 0.0% |
| Validation failures / checks | 0 / 33 | 0 / 33 |
| Model tokens | 0 | 0 |
| Model and tool cost / task | $0.000000 | $0.000000 |

## Injected transient-failure recovery

- Direct API: 30 / 30 recovered (100.0%).
- Orchestrator: 30 / 30 recovered (100.0%).

## Hard orchestration cases

These cases target pipeline problems rather than simple single-tool success. The direct baseline executes the same authored steps in order without validation or retry; MCPorch executes the same dependency structure with its DAG scheduler, output validation, and bounded retry.

| Case | Direct API: completed / expected behavior | MCPorch: completed / expected behavior | MCPorch recovery or propagation |
|---|---:|---:|---:|
| fanout_fanin | 100% / 100% | 100% / 100% | 0 recovered; propagation 0% |
| transient_timeout | 0% / 0% | 100% / 100% | 3 recovered; propagation 0% |
| invalid_then_valid | 100% / 0% | 100% / 100% | 3 recovered; propagation 0% |
| blocked_dependency | 0% / 100% | 0% / 100% | 0 recovered; propagation 100% |

## Parallel scheduling experiment

Two independent local operations: serial mean 0.062260 s; concurrency=4 mean 0.030903 s; speedup 2.015539x; time saved 0.031357 s.

## Interpretation and limits

This is a small deterministic benchmark with a simulated 0.025s API round trip per tool invocation. The direct API baseline is intentionally strong: it is given the correct tool and arguments, while MCPorch must plan and validate execution. This does not call a real remote provider and is not a provider SLA or billing measurement. Gemini is not called, so its planning tokens/cost are zero for this benchmark; local tool prices are set to $0. Validation failure rate is N/A if the run produced no invalid outputs.

### Methodology

- Completion means the workflow returned successfully. Correctness is checked separately against expected output fields, substrings, or a numeric tolerance.
- Failure recovery is measured both in the ordinary suite's separate transient-failure scenario and in the stress suite's explicit timeout/malformed-response cases.
- End-to-end time is wall-clock time from just before tool/workflow invocation through completion. For API deployment studies, repeat against the actual remote API and record provider latency and billing.
- Raw paired observations are in `benchmark-results.csv`; machine-readable details are in `benchmark-results.json`.
