# baseline运行环境

## 结论

基线在Windows、CPython 3.13.5环境完成验证。路径可以位于含中文字符的目录；A0运行器主动把标准输出和错误输出设为UTF-8，避免Windows默认控制台编码影响记录。

运行不依赖MATLAB；MATLAB只作为确定性参考证据来源。
