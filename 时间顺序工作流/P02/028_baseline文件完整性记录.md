# baseline文件完整性记录

## 结论

Python冻结包的`paper_static_baseline/evidence/manifest.sha256`逐文件记录SHA-256，覆盖代码、配置、数据、测试、README及证据文件，排除运行时缓存和清单自身。

后续如文件内容改变，重新计算值将不同，因此标签与SHA-256清单共同锁定A0内容。
