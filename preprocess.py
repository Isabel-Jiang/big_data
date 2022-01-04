import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

listings = pd.read_csv('./data/listings.csv', parse_dates=['last_review'],
                       dtype={'id': str, 'host_id': str})


# 处理名字异常
new_listings = listings.loc[~listings['name'].str.contains('测试|下架|np.NaN|不能租|下线', na=True)]


# 删除可出租夜晚为0的房源
new_listings = new_listings.loc[new_listings['availability_365'] != 0]

# 删除最小可租天数过多（超过300天）的房源，因为不符合短租的要求，很可能是异常的
new_listings = new_listings.loc[new_listings['minimum_nights'] < 365]

# print(new_listings.info())

# 下面处理价格。先占展示价格的详细分布。
print(new_listings['price'].describe(
    [.01, .05, .1, .2, .3, .5, .7, .8, .9, .92, .94, .95, .98, .99, .995, .996, .997, .998, .999]))
sns.distplot(new_listings['price'].sort_values(), bins=10)
plt.show()
new_listings = new_listings.loc[~(new_listings['price'] >= 10000) | (['price'] == 0)]

# nan处理
new_listings.fillna(value=0, inplace=True)
new_listings.drop(columns='last_review', inplace=True)
new_listings.drop(columns='neighbourhood_group', inplace=True)
new_listings.drop(columns='license', inplace=True)
new_listings.dropna(axis=0, how='any', inplace=True)

print(new_listings.info())

new_listings.to_csv('./data/new_listings.csv', sep=',', header=True, index=False)
