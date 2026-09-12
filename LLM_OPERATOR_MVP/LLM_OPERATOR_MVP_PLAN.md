# LLM 离线邻域算子 MVP 计划

## 目标与边界

本阶段只验证：离线生成的 LLM 邻域算子，能否在当前动态 FJSP-AGV 上产生合法、可解码、可重复且相对 N1-N6 有增益的新染色体。候选算子不进入 Q-learning 动作空间，不实现在线 LLM、bandit、PPO、RAG、多 Agent 或固定人工 N7，也不改变 baseline 的解码、目标、动态事件及 IS/RS/CS 语义。

## 1. 当前调用链

静态 `run_qnsga2()` 的实际主链是：

`hybrid_population` → `variation` → 非支配排序/精英保留 → `state_of` 与 `select_action`（或随机动作）→ `apply_neighborhood(data, chromosome, action, rng)` → `decode_static` → `(makespan, machine_energy)` → 接受非劣化邻居 → 再次精英选择更新种群。

其中 Q-table 固定为 4×6，动作 `0..5` 对应 N1-N6。离线 MVP 不扩展这个动作集合。

动态 `execute_rescheduling()` 的 RS/CS 主链与此相似：初始化与约束 → `_decode_dynamic` → 遗传变异与环境选择 → 条件允许时选择 N1-N6 → `_constrain_chromosome` 保持策略边界 → `_decode_dynamic` → 更新动态种群。`matlab_observed` 模式下，非订单取消事件不会执行邻域阶段；该事实必须保留，不能借 MVP 改写。

## 2. N1-N6 实际接口

统一公开入口为：

```python
apply_neighborhood(data, chromosome, action, rng) -> Chromosome
```

- `data`: `ExperimentInput`，提供工序候选机器、AGV 数和速度档位等。
- `chromosome`: 不可变 `Chromosome`。
- `action`: 0 起始的 N1-N6 编号。
- `rng`: `random.Random`；全部随机性从这里取得。
- 返回值：新的 `Chromosome`，不计算最终目标。

实际操作范围：N1、N2 改 OS；N3、N4 改 MS；N5、N6 改 AS。N4/N6 会调用 `decode_static` 获取负载或等待信息；各算子保持两个速度段不变。

LLM 候选应采用：

```python
operator(data, chromosome, rng) -> Chromosome
```

理由：它与原入口的数据、染色体、随机源和返回类型完全兼容，只去掉属于 Q-learning 动作分派的 `action`。这样候选可由独立注册表调用，也能用一个很薄的适配器与 N1-N6 进行同预算比较，而无需修改原六动作 Q-table 或伪装成 N7。不增加 `state`，因为离线有效性测试不需要在线上下文选择。

## 3. Chromosome 五段结构与合法性检查

`Chromosome` 是冻结 dataclass，五段均为 0 起始整数元组：

1. `os`：工序顺序段，按工件编号重复出现；
2. `ms`：每道工序的候选机器局部索引；
3. `agv`：每道工序的 AGV 编号；
4. `empty_speed`：AGV 空载速度档位；
5. `loaded_speed`：AGV 负载速度档位。

`Chromosome.validate(instance, agv_count, speed_count)` 已检查：五段长度、OS 工件计数、MS 候选机器范围、AS 范围及两个速度段范围。候选验证直接调用它，不另写一套染色体规则。

## 4. 动态重调度实际入口

- `execute_is(data, chromosome, original_schedule, event)`：执行 IS，不优化染色体。
- `execute_rescheduling(data, original_chromosome, original_schedule, event, strategy, ...)`：执行 RS/CS 动态优化。
- `run_dynamic_batch(...)`：第 14 步批量入口；它先运行静态 QNSGA-II，选择初始染色体并 `decode_static`，再构造 `DynamicEvent`，随后运行 IS、RS、CS。
- `scripts/run_step14_dynamic.py`：命令行入口。

动态解码实际由 `dynamic.py` 内部 `_decode_dynamic(...)` 完成，策略约束由 `_constrain_chromosome(...)` 完成。MVP 只复用，不复制或修改其语义。

## 5. LLM 生成模块的最佳插入位置

最佳位置是 baseline 外的离线实验层，逻辑位置为：

`固定父代 Chromosome` → `候选算子或 N1-N6 适配器` → `Chromosome.validate()` → `原动态策略约束` → `原动态解码器` → `原目标/HV` → `记录结果`。

候选代码本身只允许定义三参数算子，不得调用 LLM、解码器、目标函数或修改全局状态。代码生成、代码抽取、元数据记录和异常隔离都属于离线准备/评测层，不进入 `qnsga2.py`、`neighborhoods.py` 或 `dynamic.py`。

## 6. 候选如何调用原解码器评价

每次试验使用相同实例、事件、父代和 seed，并为各算子分配相同的真实 decode 次数：

1. 用传入 `rng` 生成一个新 `Chromosome`；
2. 调用已有 `Chromosome.validate(...)`；
3. 按所选 RS/CS 策略调用已有约束逻辑；
4. 调用原 `_decode_dynamic(...)`，读取 `makespan` 与 `machine_energy`（当前代码中的 TEC 目标）；
5. 汇总候选前沿，与同条件 N1-N6 前沿统一归一化后调用现有 `hypervolume_2d`；
6. 记录可行率、decode count 与 runtime。

语法/导入、接口、合法性、解码依次执行。单个候选出现异常或达到受控超时后，仅记录失败并继续下一候选；超时隔离的实现方式需在编码阶段用最小测试确定，不能让批次退出。只有实际进入原解码器的调用才计入真实 decode budget。

## 7. MVP 最少新增文件

计划新增以下文件；实现阶段以测试先行为准，若一个文件能清楚承担职责则不再拆分：

- `llm_operator_mvp/operators.py`：3-5 个离线生成候选及只读注册表；对应论文机制为“LLM 离线生成新邻域”。
- `llm_operator_mvp/evaluation.py`：N1-N6 适配、分级验证、原动态解码调用、预算和异常记录；对应“候选评价接口”。
- `scripts/run_llm_operator_mvp.py`：单实例与 2 个实例×5 seed 的统一 CLI。
- `tests/test_llm_operator_mvp.py`：接口、合法性、解码和异常隔离的最小回归测试。
- `results/llm_operator_candidates.json`：每个候选的 `id/code/generation_operator/parents/model/seed/validation_result`。
- `results/llm_operator_mvp_results.csv`：逐实例、seed、算子的原始指标。
- `LLM_OPERATOR_MVP_REPORT.md`、`HANDOFF.md`：结论、限制与独立复现实例。
- `CURRENT_STATE.md`：当前阶段、已验证项、未完成项和下一条命令。

`llm_operator_mvp/__init__.py` 仅在 Python 导入确有需要时新增。除现有依赖外不增加第三方包。

建议首轮固定使用现有正式场景清单中的一种事件、两个实例和 5 个 seed；先用极小预算 smoke test，再运行正式 MVP。具体事件与场景只从现有 `formal_dynamic_scenarios()` 选择，不另造动态语义。

## 8. 无需修改的现有文件

以下文件只作为稳定依赖读取，MVP 无需修改：

- `python_baseline/dfjspt/qnsga2.py`
- `python_baseline/dfjspt/neighborhoods.py`
- `python_baseline/dfjspt/chromosome.py`
- `python_baseline/dfjspt/decoder.py`
- `python_baseline/dfjspt/dynamic.py`
- `python_baseline/dfjspt/dynamic_experiments.py`
- `python_baseline/dfjspt/metrics.py`
- `scripts/run_step14_dynamic.py`

因此原 N1-N6、Q-learning、静态/动态解码器、Makespan/TEC、事件逻辑和 IS/RS/CS 默认行为保持不变。

## EoH 参考边界

计划仅参考 `paper` 仓库分支 `p02-static-innovation-review` 中指定的三个文件，用于：

- `evolution.py`：i1、e1、e2、m1 的候选生成组织方式；
- `eoh.py`：LLM 调用、生成流程编排与候选生命周期；
- `problem.py`：代码抽取和候选评价接口边界。

结构分析阶段三个精确 GitHub 文件 URL 未返回内容；执行阶段随后通过 sparse checkout 只取得并读取了这三个文件。MVP 仅采用 i1/e1/e2/m1 的生成语义、代码抽取后再评价的分层方式，以及失败候选不进入种群的接口边界；没有移植 EoH 的 prompt 原文、并发框架、依赖或在线 LLM 客户端，也没有转读 EoH-S、LLaMEA、RACE-Sched 或 LLM4AD Next。

## 执行与验收顺序

1. 为候选接口和失败隔离写最小失败测试；
2. 实现三参数候选接口及 N1-N6 薄适配；
3. 生成并记录 3-5 个 i1/e1/e2/m1 候选；
4. 依次验证语法/导入、接口、`Chromosome.validate()`、原动态解码；
5. 跑单实例 smoke test；
6. 跑一种动态事件、两个实例、5 seed、相同 decode budget 的 MVP；
7. 输出 CSV、报告、交接说明并更新当前状态。

成功标准不是“复现论文”或“在线替代 Q-learning”，而是有充分原始记录判断：是否至少存在 1-2 个候选在多个实例/seed 上稳定、可重复，并相对 N1-N6 产生不同且有效的增益。
