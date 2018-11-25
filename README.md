# sklearn-DecisionTree
决策树、随机森林

用泰坦尼克号事件的数据集练习一下决策树和随机森林的API。

# 分析数据集信息
先读入数据集，看看有哪些特征：
```
import pandas as pd

titanic = pd.read_csv("titanic.csv")

# 分析数据集信息
print("*" * 30 + " info " + "*" * 30)
print(titanic.info())
print(titanic.head())
```

输出：
```
****************************** info ******************************
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
None
   PassengerId  Survived  Pclass    ...        Fare Cabin  Embarked
0            1         0       3    ...      7.2500   NaN         S
1            2         1       1    ...     71.2833   C85         C
2            3         1       3    ...      7.9250   NaN         S
3            4         1       1    ...     53.1000  C123         S
4            5         0       3    ...      8.0500   NaN         S

[5 rows x 12 columns]
```

`survived`字段表示是否生存，我们以此作为预测目标。
```
y = titanic['survived']
print(y.head())
```

输出：
```
0    0
1    1
2    1
3    1
4    0
Name: survived, dtype: int64
```

我们取其中三个特征做分析演示，分别是：
- pclass：1-一等舱，2-二等舱，3-三等舱
- age年龄
- sex性别

```
x = titanic[['pclass', 'age', 'sex']]
print(x.info())
print(x.head())
```

输出：
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 3 columns):
Pclass    891 non-null int64
Age       714 non-null float64
Sex       891 non-null object
dtypes: float64(1), int64(1), object(1)
memory usage: 21.0+ KB
None
   Pclass   Age     Sex
0       3  22.0    male
1       1  38.0  female
2       3  26.0  female
3       1  35.0  female
4       3  35.0    male
```

# 缺失值处理

age字段存在缺失，用均值填充：
```
age_mean = x['age'].mean()
print("*" * 30 + " age_mean " + "*" * 30)
print(age_mean)
x['age'].fillna(age_mean, inplace=True)
print("*" * 30 + " 处理age缺失值后 " + "*" * 30)
print(x.info())
```

输出：
```
****************************** age_mean ******************************
29.69911764705882
****************************** 处理age缺失值后 ******************************
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 3 columns):
Pclass    891 non-null int64
Age       891 non-null float64
Sex       891 non-null object
dtypes: float64(1), int64(1), object(1)
memory usage: 21.0+ KB
```

# 特征抽取 - onehot编码

为了方便使用字典特征抽取，构造字典列表：
```
x_dict_list = x.to_dict(orient='records')
print("*" * 30 + " train_dict " + "*" * 30)
print(pd.Series(x_dict_list[:5]))

dict_vec = DictVectorizer(sparse=False)
x = dict_vec.fit_transform(x_dict_list)
print("*" * 30 + " onehot编码 " + "*" * 30)
print(dict_vec.get_feature_names())
print(x[:5])
```

```
****************************** train_dict ******************************
0      {'Pclass': 3, 'Age': 22.0, 'Sex': 'male'}
1    {'Pclass': 1, 'Age': 38.0, 'Sex': 'female'}
2    {'Pclass': 3, 'Age': 26.0, 'Sex': 'female'}
3    {'Pclass': 1, 'Age': 35.0, 'Sex': 'female'}
4      {'Pclass': 3, 'Age': 35.0, 'Sex': 'male'}
dtype: object
****************************** onehot编码 ******************************
['Age', 'Pclass', 'Sex=female', 'Sex=male']
[[22.  3.  0.  1.]
 [38.  1.  1.  0.]
 [26.  3.  1.  0.]
 [35.  1.  1.  0.]
 [35.  3.  0.  1.]]
```

# 划分训练集和测试集

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
```

# 决策树分类器

```
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)

print("*" * 30 + " 准确率 " + "*" * 30)
print(dec_tree.score(x_test, y_test))
```

输出：
```
****************************** 准确率 ******************************
0.7892376681614349
```

# 随机森林分类器
- n_jobs: -1表示设置为核心数量
- n_estimators: 决策树数目
- max_depth: 树最大深度

同时使用网格搜索最优超参数：
```
rf = RandomForestClassifier(n_jobs=-1)
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
```

输出：
```
****************************** 超参数网格搜索 ******************************
耗时：66.85670185089111
最优参数：{'max_depth': 5, 'n_estimators': 120}
****************************** 准确率 ******************************
0.7847533632286996
```
最优的参数是`{'max_depth': 5, 'n_estimators': 120}`。
在我的2015款MacBookPro上，仅2折的交叉验证就跑了66秒。- -|
