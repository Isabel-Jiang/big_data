import re
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./data/new_listings.csv', dtype={'id': str, 'host_id': str})
plt.rcParams['font.sans-serif'] = ['SimHei']
# 获取每个地区平均价格，并排序
df_neighbourhood = df.groupby(by="neighbourhood")["price"].mean().sort_values(ascending=False)
# 绘图
plt.figure(figsize=(20, 8))
bar1 = plt.bar(df_neighbourhood.index, df_neighbourhood.values)
plt.xticks(rotation=45, size=15)
plt.yticks(size=15)
plt.title("各地区房价")
plt.xlabel("地区", size=15)
plt.ylabel("平均价格", size=15)
for i in bar1:
    plt.text(i.get_x() + i.get_width() / 2, i.get_height(), "%d" % int(i.get_height()), ha="center", va="bottom",
             fontsize=15)
plt.show()


