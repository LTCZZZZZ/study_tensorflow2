import time

import numpy as np
import pandas as pd
from IPython import display
import matplotlib.pyplot as plt

import tensorflow as tf

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


# 下面为正式的数据预处理
def preprocessing(dfdata):
    dfresult = pd.DataFrame()

    # 思考：为什么分类型的数据都要转换成one-hot编码？？
    # Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    # 添加“年龄是否缺失”作为辅助特征
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    # 重命名列
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return (dfresult)


x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values

print("x_train.shape =", x_train.shape)
print("x_test.shape =", x_test.shape)

# 使用Keras接口有以下3种方式构建模型：
# 使用Sequential按层顺序构建模型，使用函数式API构建任意结构模型，继承Model基类构建自定义模型。
tf.keras.backend.clear_session()
net = tf.keras.Sequential()
# 这里没加kernel_initializer参数
net.add(tf.keras.layers.Dense(20, activation='relu', input_shape=(15,)))  # input_shape可省略
net.add(tf.keras.layers.Dense(10, activation='relu', name='2222222'))
# 当activation获取不到时，会报错Please ensure this object is passed to the `custom_objects` argument
# 参见 https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object
net.add(tf.keras.layers.Dense(1, activation='sigmoid'))
print(net.summary())

# 训练模型通常有3种方法，内置fit方法，内置train_on_batch方法，以及自定义训练循环。
# 此处我们选择最常用也最简单的内置fit方法。
# 二分类问题选择二元交叉熵损失函数
# tf.keras.metrics中查看metrics参数
net.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC'])

history = net.fit(x_train, y_train,
                  batch_size=64,
                  epochs=30,
                  validation_split=0.2  # 分割一部分训练数据用于验证
                  )


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    # 第三个参数format_string参见sys_test项目figure1.py
    # plt.figure()  # 生成一个新figure
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()
    # time.sleep(1)


plot_metric(history, "loss")
plot_metric(history, "auc")
# plt.show()

# 在测试集上的loss和metrics
print(net.evaluate(x=x_test, y=y_test))  # 二分类问题下，auc可视为准确率的评判
# 在测试集上的准确率
result = net.predict_classes(x_test).reshape(-1) == y_test  # 这里降维，其实一般应该是y升维才对
print(result.mean())  # 和auc差得有点多，原因何在？

print(net.predict(x_test[0:10]))  # 预测概览
print(net.predict_classes(x_test[0:10]))  # 预测类别，此函数即将被废弃

# Keras方式保存
net.save('./data/keras_model.h5')
# 加载模型，看看执行结果是否相同
net = tf.keras.models.load_model('./data/keras_model.h5')
print(net.evaluate(x_test,y_test))

# 保存权重，该方式仅仅保存权重张量
net.save_weights('./data/tf_model_weights.ckpt',save_format = "tf")
# TensorFlow原生方式：保存模型结构与模型参数到文件，该方式保存的模型具有跨平台性便于部署
net.save('./data/tf_model_savedmodel', save_format="tf")
net = tf.keras.models.load_model('./data/keras_model.h5')
print(net.evaluate(x_test,y_test))
