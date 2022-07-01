# 学习率调度模块需要配合优化器使用，通过from jittor import lr_scheduler来获取该模块。
# 分段常数衰减(Piecewise Constant Decay)
import tensorflow as tf
import matplotlib.pyplot as plt

# 分段常数衰减 PiecewiseConstantDecay
decay = "PiecewiseConstantDecay"
boundaries=[20, 40, 60, 80]  # 以 0-20 20-40 40-60 60-80 80-inf 为分段
values=[1.0, 0.5, 0.25, 0.125, 0.0625]  # 各个分段学习率的值
piece_wise_constant_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                            boundaries=boundaries, values=values, name=None)
lr = []
for step in range(100):
    lr.append(piece_wise_constant_decay(step))

fig = plt.figure("1")
plt.plot(range(100), lr)
plt.savefig(f"{decay}.png")


# 逆时衰减(Inverse Time Decay)
decay = "Inverse Time Decay"
inverse_time_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
                     initial_learning_rate=1., decay_steps=1, decay_rate=0.1)
lr = []
for step in range(100):
    lr.append(inverse_time_decay(step))

fig = plt.figure("2")
plt.plot(range(100), lr)
plt.savefig(f"{decay}.png")


# 指数衰减(Exponential Decay)
decay = "Exponential Decay"
exponential_decay_staircase = tf.keras.optimizers.schedules.ExponentialDecay(
                             initial_learning_rate=1., decay_steps=100, decay_rate=0.96, staircase=True)
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1., decay_steps=100, decay_rate=0.96)
lr1 = []
lr2 = []
for step in range(1000):
    lr1.append(exponential_decay_staircase(step))
    lr2.append(exponential_decay(step))

fig = plt.figure("3")
plt.plot(range(1000), lr1)
plt.savefig(f"{decay}_staircase.png")

fig = plt.figure("4")
plt.plot(range(1000), lr2)
plt.savefig(f"{decay}.png")


# 余弦衰减(Cosine Decay)
decay = "Cosine Decay"
# decay_steps此处是学习率衰减到0所需要的step数
inverse_time_decay = tf.keras.optimizers.schedules.CosineDecay(
                     initial_learning_rate=1., decay_steps=1000)
lr = []
for step in range(1000):
    lr.append(inverse_time_decay(step))

fig = plt.figure("5")
plt.plot(range(1000), lr)
plt.savefig(f"{decay}.png")