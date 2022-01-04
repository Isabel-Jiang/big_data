from jieba import analyse
import re
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./data/new_listings.csv', dtype={'id': str, 'host_id': str})
plt.rcParams['font.sans-serif'] = ['SimHei']

huairou = pd.Series()
changping = pd.Series()
yanqing = pd.Series()
pinggu = pd.Series()
miyun = pd.Series()
mentougou = pd.Series()
dongcheng = pd.Series()
shunyi = pd.Series()
fangshan = pd.Series()
xicheng = pd.Series()
fengtai = pd.Series()
haidian = pd.Series()
chaoyang = pd.Series()
tongzhou = pd.Series()
for index, row in df.iterrows():
    neighbour = row["neighbourhood"]
    name = row["name"]
    if "怀柔区" in neighbour:
        huairou = huairou.append(pd.Series(analyse.extract_tags(name)))
    elif "昌平区" in neighbour:
        changping = changping.append(pd.Series(analyse.extract_tags(name)))
    elif "延庆区" in neighbour:
        yanqing = yanqing.append(pd.Series(analyse.extract_tags(name)))
    elif "平谷区" in neighbour:
        pinggu = pinggu.append(pd.Series(analyse.extract_tags(name)))
    elif "密云县" in neighbour:
        miyun = miyun.append(pd.Series(analyse.extract_tags(name)))
    elif "门头沟" in neighbour:
        mentougou = mentougou.append(pd.Series(analyse.extract_tags(name)))
    elif "东城区" in neighbour:
        dongcheng = dongcheng.append(pd.Series(analyse.extract_tags(name)))
    elif "顺义区" in neighbour:
        shunyi = shunyi.append(pd.Series(analyse.extract_tags(name)))
    elif "房山区" in neighbour:
        fangshan = fangshan.append(pd.Series(analyse.extract_tags(str(name))))
    elif "西城区" in neighbour:
        xicheng = xicheng.append(pd.Series(analyse.extract_tags(name)))
    elif "丰台区" in neighbour:
        fengtai = fengtai.append(pd.Series(analyse.extract_tags(name)))
    elif "海淀区" in neighbour:
        haidian = haidian.append(pd.Series(analyse.extract_tags(name)))
    elif "朝阳区" in neighbour:
        chaoyang = chaoyang.append(pd.Series(analyse.extract_tags(name)))
    elif "通州区" in neighbour:
        tongzhou = tongzhou.append(pd.Series(analyse.extract_tags(name)))

# 合并所有Series
df_word = pd.concat([huairou.value_counts(), changping.value_counts(), yanqing.value_counts(), pinggu.value_counts(),
                     miyun.value_counts(), mentougou.value_counts(), dongcheng.value_counts(), shunyi.value_counts(),
                     fangshan.value_counts(), xicheng.value_counts(), fengtai.value_counts(), haidian.value_counts(),
                     chaoyang.value_counts(), tongzhou.value_counts()], axis=1)
# nan值设置为0
df_word = df_word.fillna(0)
# 设置新的列名
columns_name = ["怀柔区", "昌平区", "延庆区", "平谷区", "密云区", "门头沟", "东城区", "顺义区", "房山区", "西城区", "丰台区", "海淀区", "朝阳区", "通州区"]
df_word.columns = columns_name
# 所有区出现数排名前十的关键词
df_wordtop10 = pd.DataFrame(index=range(10))
for i in columns_name:
    df_wordtop10[i] = pd.Series(df_word[i].sort_values(ascending=False).index)

print(df_wordtop10)
