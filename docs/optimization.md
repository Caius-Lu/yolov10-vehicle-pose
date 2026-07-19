# Optimization

优化顺序：先 one2one-only、向量化 decode、减少同步与拷贝，再做 channels/depth/卷积形式、特征层、输入分辨率和数值精度消融。每次只改一个主变量并做完整精度回归。Fast/Balanced/Accurate 只能从真实 Pareto 前沿产生。

