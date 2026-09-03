# 完整baseline复现命令

## 结论

从Python-NEW仓库根目录执行：

```powershell
python paper_static_baseline/scripts/run_a0.py --instance Mk01 --population 10 --generations 1 --seed 1 --output paper_static_baseline/results/Mk01-seed-1
```

该命令用于快速验证完整调用链。正式实验只替换已批准的实例、种群、代数、种子和新输出目录，不改变A0算法入口。
