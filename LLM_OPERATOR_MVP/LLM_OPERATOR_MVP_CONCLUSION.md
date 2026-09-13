# LLM 离线邻域算子 MVP 结论

## 一句话结论

LLM 能够离线生成适配当前动态 FJSP-AGV 染色体与原解码器的新邻域算子，但本轮 2 个实例、5 个 seed 的 MVP 尚未找到在多个实例和 seed 上稳定优于 N1-N6 的 1-2 个候选，因此可以继续离线迭代候选，暂不应进入“LLM 在线替代 Q-learning”的研究阶段。

## 实验事实

- 动态事件：订单取消，事件时间 50，取消工件 2；
- 实例：Mk02、Mk07；
- seed：11、22、33、44、55；
- 对比算子：N1-N6 与 4 个 LLM 候选；
- 统一预算：每个实例-seed-算子 20 次真实动态解码；
- 总计：2,000 次候选动态解码、100 行原始结果；
- 可行率：100%；
- 异常：0；
- 重复性：第二次完整运行除 runtime 外，所有字段与第一次一致；
- 测试：MVP 测试 4/4、项目权威 `tests/` 105/105 通过；
- baseline：指定的八个核心文件均未修改。

## 候选表现

四个候选均通过以下检查：

1. Python 语法与导入；
2. `operator(data, chromosome, rng) -> Chromosome` 接口；
3. `Chromosome.validate()`；
4. 原动态解码器解码。

表现最好的候选是 `llm_e1_order_agv`。它在 10 个实例-seed 组合中，相对当次最佳 N1-N6：

- Makespan 严格胜出 2/10 次；
- TEC 严格胜出 2/10 次；
- HV 严格胜出 2/10 次。

其平均 HV 为 0.8797，略高于 N1 的 0.8780；但平均最小 Makespan 为 275.349，弱于 N1 的 270.829；平均最小 TEC 为 3297.356，弱于 N1 的 3291.198。因此它是值得继续演化的候选种子，但不是稳定优于 N1-N6 的证据。

其余候选没有形成稳定信号：

- `llm_m1_transport_rewrite` 仅有 1/10 次 HV 胜出；
- `llm_i1_machine_speed` 和 `llm_e2_machine_transport` 未严格胜出当次最佳 N1-N6。

## 对研究问题的回答

本轮支持：

> LLM 可以离线生成结构不同、合法、可重复，且偶尔能超过 N1-N6 的动态 FJSP-AGV 邻域算子。

本轮不支持：

> 已经存在 1-2 个 LLM 算子，能在多个实例和 seed 上稳定优于 N1-N6。

因此 MVP 判定为：**接口与可行性通过，性能门槛未通过。**

## 下一步建议

只继续离线阶段：以 `llm_e1_order_agv` 为父代生成少量结构变体，扩大实例、动态事件和 decode 预算后重新筛选。在线 LLM selector、contextual bandit、PPO 和新 Q-learning 仍不应实现。

## GitHub 同步说明

全部代码、测试、候选记录、原始 CSV、PLAN、REPORT、HANDOFF、CURRENT_STATE 和本结论文档均同步到：

`https://github.com/tianyueni78-cyber/paper-explaination/tree/LLM/LLM_OPERATOR_MVP`

目标仓库原历史在当前环境中多次 fetch 超时，因此 `LLM` 被创建为只包含本次 MVP 工件的 orphan 分支。该处理没有影响本地实验、结果、重复性验证或测试，也没有覆盖目标仓库已有分支；影响仅是 `LLM` 分支不继承目标仓库默认分支的提交历史。
