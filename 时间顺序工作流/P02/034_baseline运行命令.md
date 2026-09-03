# baseline运行命令

## 结论

A0统一由`paper_static_baseline/scripts/run_a0.py`启动，必填参数为实例、种群、代数、随机种子和输出目录。运行器拒绝覆盖已有目录，并对成功、失败或中断写入状态记录。

验收命令为：

```powershell
python paper_static_baseline/scripts/verify_baseline.py
```
