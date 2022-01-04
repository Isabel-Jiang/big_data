import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('./data/new_listings.csv', dtype={'id': str, 'host_id': str})


# 根据价格分类，使得每个类别房间数大致相同
def price_level(price):
    if price <= 100:
        return 1
    elif price <= 500:
        return 2
    elif price <= 1000:
        return 3
    elif price <= 3000:
        return 4
    elif price <= 5000:
        return 5
    elif price <= 10000:
        return 6
    else:
        return 7


# 把价格转成类别
df["price"] = df["price"].apply(lambda x: price_level(x))
y = df[["price"]]

# 处理输入数据的格式
# 把neighborhood列转成只有中文
df['neighbourhood'] = df['neighbourhood'].map(lambda x: x.split('/')[0].strip())
# 类别dummy
neigh_dummy = pd.get_dummies(df['neighbourhood'])
df = pd.concat([df, neigh_dummy], axis=1)
df.drop(columns=['neighbourhood'], inplace=True)
room_dummy = pd.get_dummies(df['room_type'])
df = pd.concat([df, room_dummy], axis=1)
x = df.drop(columns=['id', 'name', 'host_id', 'host_name', 'room_type', 'price'])
# 归一化
x = (x - x.min()) / (x.max() - x.min())
x = x.values
y = y.values
# 分割训练测试
x_train, x_test, y_train, y_test = train_test_split(x, y)

# KNN
KNN = KNeighborsClassifier(n_neighbors=7)
KNN.fit(x_train, y_train)
print('KNN训练集', KNN.score(x_train, y_train))
print('KNN测试集', KNN.score(x_test, y_test))

# XGBoost
XGB = xgb.XGBClassifier(early_stopping_rounds=200, max_depth=7)
XGB.fit(x_train, y_train)
print('XGB训练集', XGB.score(x_train, y_train))
print('XGB测试集', XGB.score(x_test, y_test))

# BP神经网络
BP = MLPClassifier(hidden_layer_sizes=50, learning_rate_init=5e-4, learning_rate='adaptive', early_stopping=True)
BP.fit(x_train, y_train)
print('BP神经网络训练集', BP.score(x_train, y_train))
print('BP神经网络测试集', BP.score(x_test, y_test))