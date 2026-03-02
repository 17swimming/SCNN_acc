import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 读取CSV文件
df = pd.read_csv('../output/simulator_test_results_8.csv')

# 按dense列排序
df_sorted = df.sort_values(by='dense')

# 创建图形和子图
fig, ax1 = plt.subplots(figsize=(12, 6))

# 定义用于拟合的二次多项式函数
def poly_func(x, a, b, c):
    return a * x**2 + b * x + c

# 第一条线：绘制dense与speedup的曲线趋势线
# 拟合曲线
popt1, _ = curve_fit(poly_func, df_sorted['dense'], df_sorted['speedup'])
curve_y1 = poly_func(df_sorted['dense'], *popt1)

ax1.scatter(df_sorted['dense'], df_sorted['speedup'], alpha=0.6, color='blue', label='Speedup Data Points')  # 散点
line1, = ax1.plot(df_sorted['dense'], curve_y1, label='Speedup Trend Curve', color='blue', linewidth=2)
ax1.set_xlabel('Dense (Sparsity)')
ax1.set_ylabel('Speedup', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 设置标题
ax1.set_title('Relationship: Dense vs Speedup & Simulator Cycles Percentage')

# 创建第二个y轴用于绘制simulator_cycles的百分比
ax2 = ax1.twinx()

percentage = (df_sorted['simulator_cycles'] / df_sorted['theorical_cycles']) * 100

ax2.scatter(df_sorted['dense'], percentage, alpha=0.6, color='red', label='Simulator Cycles % Data Points')  # 散点
ax2.set_ylabel('Simulator Cycles Percentage (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 添加网格
ax1.grid(True, linestyle='--', alpha=0.6)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# 调整布局
plt.tight_layout()
# 如果需要保存图片，取消下面这行的注释
plt.savefig('../output/dense_speedup_analysis_8.png', dpi=300, bbox_inches='tight')
# 显示图表
plt.show()

