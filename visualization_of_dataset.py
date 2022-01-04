import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts.globals import GeoType
listings = pd.read_csv('./data/listings.csv', parse_dates=['last_review'],
                       dtype={'id': str, 'host_id': str})


# 查看前面的数据
print('***************************************************************')
print('查看前面的数据')
pd.set_option('display.max_columns', None)
print(listings.head())

# 对数据集进行简单的总体查看
print('***************************************************************')
print('查看数据的条数')
listings.info()
print('该数据集共有17列，分别为：房子id,房子名字，房东ID、房东姓名、所属行政区、经纬度、房间类型、价格、最小可租天数、评论数量、'
      '最后一次评论时间、每月评论占比、可出租房屋、每年可出租天数、资质')

# 查看空缺数据
print('***************************************************************')
print('查看空白数据')
print(listings.isnull().sum())
print('neighbourhood_group和license 全部是空的，review相关的数据有空的')
print()
print(listings[listings["number_of_reviews"] == 0].count())
print('最后评论时间和每月评论占比在评论数为0时为空值')

# 查看数据分布
print('***************************************************************')
print('查看数据分布')
subsets = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
           'availability_365']
fig, axes = plt.subplots(len(subsets), 1, figsize=(20, 10))
plt.subplots_adjust(hspace=1)
for i, subset in enumerate(subsets):
    sns.boxplot(x=listings[subset], y=None, ax=axes[i], whis=2, orient='h',
                meanline={"marker": "D", "markerfacecolor": "red"})
print(listings.describe())
plt.show()

# 查看地理位置分布
city = "北京"
g = Geo()
g.add_schema(maptype=city)
# 添加经纬度
for index, row in listings.iterrows():
    g.add_coordinate(row["name"], row["longitude"], row["latitude"])
# 添加价格
data_price = list(zip(listings["name"], listings["price"]))
g.add('', data_price, type_=GeoType.EFFECT_SCATTER, symbol_size=1)
g.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
# 设置不同价格点的颜色
pieces = [
    {'max': 1, 'label': '0以下', 'color': '#50A3BA'},
    {'min': 1, 'max': 100, 'label': '1-100', 'color': '#3700A4'},
    {'min': 101, 'max': 500, 'label': '101-500', 'color': '#81AE9F'},
    {'min': 501, 'max': 1000, 'label': '501-1000', 'color': '#E2C568'},
    {'min': 1001, 'max': 3000, 'label': '1001-3000', 'color': '#FCF84D'},
    {'min': 3001, 'max': 5000, 'label': '3001-5000', 'color': '#DD0200'},
    {'min': 5001, 'max': 10000, 'label': '5001-10000', 'color': '#DD675E'},
    {'min': 10001, 'label': '10000以上', 'color': '#D94E5D'}
]
g.set_global_opts(
    visualmap_opts=opts.VisualMapOpts(is_piecewise=True, pieces=pieces),
    title_opts=opts.TitleOpts(title="{}-短租房分布".format(city)),
)
g.render()
