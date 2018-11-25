import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time


titanic = pd.read_csv("titanic.csv")

# 分析数据集信息
print("*" * 30 + " info " + "*" * 30)
print(titanic.info())
print(titanic.head())

# `survived`字段表示是否生存，以此作为预测目标
y = titanic['Survived']
print("*" * 30 + " y " + "*" * 30)
print(y.head())

# 取其中三个特征做分析演示，分别是：
# pclass，1-一等舱，2-二等舱，3-三等舱
# age年龄
# sex性别
x = titanic[['Pclass', 'Age', 'Sex']]
print("*" * 30 + " x " + "*" * 30)
print(x.info())
print(x.head())

# age字段存在缺失，用均值填充
age_mean = x['Age'].mean()
print("*" * 30 + " age_mean " + "*" * 30)
print(age_mean)
x['Age'].fillna(age_mean, inplace=True)
print("*" * 30 + " 处理age缺失值后 " + "*" * 30)
print(x.info())

# 特征抽取 - onehot编码
# 为了方便使用字典特征抽取，构造字典列表
x_dict_list = x.to_dict(orient='records')
print("*" * 30 + " train_dict " + "*" * 30)
print(pd.Series(x_dict_list[:5]))

dict_vec = DictVectorizer(sparse=False)
x = dict_vec.fit_transform(x_dict_list)
print("*" * 30 + " onehot编码 " + "*" * 30)
print(dict_vec.get_feature_names())
print(x[:5])

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# 决策树分类器
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)

print("*" * 30 + " 准确率 " + "*" * 30)
print(dec_tree.score(x_test, y_test))

# 随机森林分类器
# n_jobs: -1表示设置为核心数量
rf = RandomForestClassifier(n_jobs=-1)

# 网格搜索
# n_estimators: 决策树数目
# max_depth: 树最大深度
param = {
    "n_estimators": [120, 200, 300, 500, 800, 1200],
    "max_depth": [5, 8, 15, 25, 30]
}
# 2折交叉验证
search = GridSearchCV(rf, param_grid=param, cv=2)
print("*" * 30 + " 超参数网格搜索 " + "*" * 30)

start_time = time.time()
search.fit(x_train, y_train)
print("耗时：{}".format(time.time() - start_time))
print("最优参数：{}".format(search.best_params_))

print("*" * 30 + " 准确率 " + "*" * 30)
print(search.score(x_test, y_test))






