# 当前状态

- 阶段：LLM 离线邻域算子第一阶段 MVP 已运行，等待最终验证与上传。
- 范围：订单取消；Mk02/Mk07；seed 11/22/33/44/55；N1-N6 + 4 个 LLM 候选；每算子每 case 20 decode。
- 已完成：PLAN、候选实现、四级验证、单实例 smoke、2×5 正式实验、CSV、REPORT、HANDOFF。
- 当前证据：2,000 次候选动态 decode，100% 可行、0 错误；性能稳定增益门槛未通过。
- 最有希望候选：`llm_e1_order_agv`，相对每 case 最佳 N1-N6 在 Makespan/TEC/HV 各胜出 2/10 次。
- baseline：八个指定核心文件均未修改。
- 待完成：复现实验一致性检查、完整测试、上传 `paper-explaination` 的 `LLM` 分支。
