import numpy as np
import pandas as pd
from IPython import display
import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow.keras import models,layers

display.set_matplotlib_formats('svg')

dftrain_raw = pd.read_csv('../../data/titanic/train.csv')
dftest_raw = pd.read_csv('../../data/titanic/test.csv')
print(dftrain_raw.head(8))

# print(dftrain_raw.info())
# 幸存者分布情况
se = dftrain_raw['Survived'].value_counts()
ax = se.plot(kind='bar',
             figsize=(12, 8), fontsize=15, rot=0, color='#6666FF')
# 添加数值标签
for p in ax.patches:
    print(p.get_height(), p.get_x(), p.get_y())
    # 这两个语句看起来效果相同，区别可能是和选中的对象有关，plt.text应该是给当前活动对象添加标签
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() - 20),
                ha='center', va='bottom', fontsize=15, color='#FF9900')
    plt.text(p.get_x() + p.get_width() / 2, p.get_height() * 1.005, str(p.get_height()),
             ha='center', va='bottom', fontsize=15)
ax.set_ylabel('Counts', fontsize=15)
ax.set_xlabel('Survived', fontsize=15)
plt.show()

# 年龄分布情况，bins参数指定了分割的区间数
ax = dftrain_raw['Age'].plot(kind='hist', bins=20, color='#FF9900', alpha=0.66,
                             figsize=(12, 8), fontsize=15)
ax.set_ylabel('Frequency', fontsize=15)
ax.set_xlabel('Age', fontsize=15)
plt.show()

# Age和Survived的相关性
# 从图上可以看出，在年龄较小的区域，幸存者的比例明显更高，20-30岁区间，遇难者比例更高，30-40岁区间持平，大于40岁区间幸存者比例较低
# 这说明了什么？个人推测：20-30岁，血仍未冷，人性的光辉还在，愿意将生的机会优先留给孩子和老人，
# 30-40岁，血液已经冷了，人性的光辉已经消失，(这一行是github copilot自动生成的，我惊呆了，我的意思在下一行)
# 30-40岁，人性的自私开始显现，这个年龄段的人更加现实
# 40岁以上，受限于身体机能的原因，幸存率更低
ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
dftrain_raw.query('Survived == 1')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
dftrain_raw['Age'].plot(kind='kde', figsize=(12, 8), fontsize=15)
ax.legend(['Survived==0', 'Survived==1', 'All'], fontsize=12)
ax.set_ylabel('Density', fontsize=15)
ax.set_xlabel('Age', fontsize=15)
plt.show()



# 下面是各类画图尝试

# Allows plotting of one column versus another，比如说，某个人的年龄和性别，或者说，某个人的年龄和身高
dftrain_raw.plot(kind='scatter', x='Age', y='Survived', alpha=0.2)
plt.show()

dftrain_raw.iloc[:30].plot(kind='bar', alpha=1)
# plt.savefig('myplot1.svg', format='svg')
plt.show()

dftrain_raw.iloc[:10].plot(kind='line', alpha=1)
# plt.savefig('myplot1.svg', format='svg')
plt.show()

# 纵轴坐标Frequency
dftrain_raw.iloc[:10].plot(kind='hist', alpha=1)
# plt.savefig('myplot1.svg', format='svg')
plt.show()

dftrain_raw.iloc[:10].plot(kind='box')
# plt.savefig('myplot1.svg', format='svg')
plt.show()

# 类似于概览分布图，纵轴坐标Density
dftrain_raw.iloc[:10].plot(kind='kde')
# plt.savefig('myplot1.svg', format='svg')
plt.show()

# 各值累加区域图
dftrain_raw.iloc[:10].plot(kind='area')
# plt.savefig('myplot1.svg', format='svg')
plt.show()

# 饼图
dftrain_raw.iloc[:10].plot(kind='pie', y='Fare')
# plt.savefig('myplot1.svg', format='svg')
plt.show()

# 六边形分箱图，和scatter有些类似，这里看幸存和年龄两个维度的分布
dftrain_raw.plot(kind='hexbin', x='Age', y='Survived', gridsize=15, cmap=plt.cm.Blues)
plt.savefig('myplot1.svg', format='svg')
plt.show()

# 六边形分箱图可以看数据在两个维度的交叉分布
# gridsize不是指每个网格的大小，而是指网格总数(横向)
dftrain_raw.plot(kind='hexbin', x='Age', y='Fare', alpha=0.8, gridsize=40)
plt.savefig('myplot2.svg', format='svg')
plt.show()

# 某前端仓库，JavaScript，https://github.com/d3/d3-hexbin
# Hexagonal binning is useful for aggregating data into a coarser representation for display.
# For example, rather than rendering a scatterplot of tens of thousands of points,
# bin the points into a few hundred hexagons to show the distribution.
# Hexbins can support a color encoding, area encoding, or both.
