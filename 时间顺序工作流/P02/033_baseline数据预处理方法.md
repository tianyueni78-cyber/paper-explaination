# baseline数据预处理方法

## 结论

Brandimarte文本按工件、工序、可选机器和加工时间原值解析；MATLAB一基编号转换为Python内部零基编号。资源JSON按原值读取为机器、AGV和运输参数。

不做归一化、插值、删样或随机增强。输出MATLAB对照染色体时恢复一基编号。
