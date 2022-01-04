import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./data/new_listings.csv', dtype={'id': str, 'host_id': str})
plt.rcParams['font.sans-serif'] = ['SimHei']
# 对表进行透视化
df_type = df.pivot_table(index="neighbourhood", columns="room_type", values="price", aggfunc="mean").fillna(0)
# print(df_type)
x_ = np.arange(len(df_type.index))
# 绘制条形图
plt.figure(figsize=(20, 8))
bar1 = plt.bar(x_ - 0.3, df_type["Entire home/apt"].values, width=0.3, label="Entire home/apt")
bar2 = plt.bar(x_, df_type["Private room"].values, width=0.3, label="Private room")
bar3 = plt.bar(x_ + 0.3, df_type["Shared room"].values, width=0.3, label="Shared room")
plt.xticks(x_, df_type.index, rotation=60, fontsize=15)
plt.legend(fontsize=15)
for bar in bar1:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "%d" % bar.get_height(), ha="center", va="bottom",
             size=13)
for bar in bar2:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "%d" % bar.get_height(), ha="center", va="bottom",
             size=13)
for bar in bar3:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "%d" % bar.get_height(), ha="center", va="bottom",
             size=13)
plt.show()
