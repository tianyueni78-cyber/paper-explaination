# LLM 邻域 MVP 交接

## 文件与论文机制

- `llm_operator_mvp/operators.py`：i1/e1/e2/m1 离线候选算子。
- `llm_operator_mvp/evaluation.py`：候选隔离验证、N1-N6 同预算动态评价。
- `scripts/run_llm_operator_mvp.py`：单实例与多 seed 终端入口。
- `tests/test_llm_operator_mvp.py`：接口、合法性、原解码、超时隔离与 CLI smoke 测试。
- `results/llm_operator_candidates.json`：候选代码与生成/验证记录。
- `results/llm_operator_mvp_results.csv`：100 行原始实验结果。

没有修改 QNSGA-II、N1-N6、Chromosome、静态/动态解码、动态事件、指标或 Step 14 入口。

## 运行命令

单实例 smoke test：

```powershell
python scripts/run_llm_operator_mvp.py --instance Mk02 --seed 11 --decode-budget 5 --output .codex_tmp/llm_operator_smoke.csv
```

完整 MVP（Mk02/Mk07 × 5 seed，默认每算子 20 decode）：

```powershell
python scripts/run_llm_operator_mvp.py --decode-budget 20
```

测试：

```powershell
python -m pytest tests/test_llm_operator_mvp.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider
```

## 阅读顺序

1. `LLM_OPERATOR_MVP_REPORT.md`；
2. `results/llm_operator_mvp_results.csv`；
3. `results/llm_operator_candidates.json`；
4. `llm_operator_mvp/operators.py`；
5. `llm_operator_mvp/evaluation.py`。

## 下一步边界

若继续第二轮离线筛选，应优先围绕 `llm_e1_order_agv` 生成变体，并扩大实例、事件和 decode 预算。不要在当前结果上实现在线 LLM selector；性能门槛尚未通过。
