import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer, LabelBinarizer, Normalizer, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit
pd.options.display.max_columns = None
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb


def check_row_nan_count(df_var, ds_type):
    row_null = df_var.isnull().sum(axis=1)
    row_null_df = pd.DataFrame(df_var["ID"])
    row_null_df["行缺失值数量"] = row_null

    total_columns = df_var.shape[1]
    row_null_df["行缺失值比例"] = round(row_null_df["行缺失值数量"]/total_columns, 2)
    row_null_df_groupby = row_null_df["行缺失值数量"].value_counts()
    data = {
        "行缺失值数量": row_null_df_groupby.index,
        "缺失数量对应的条数": row_null_df_groupby.values
    }
    row_null_df_groupby = pd.DataFrame(data)
    row_null_df_groupby["缺失数量对应的占比"] = round(row_null_df_groupby["缺失数量对应的条数"] / df_var.shape[0], 3)
    row_null_df_groupby = row_null_df_groupby.sort_values(by="行缺失值数量" , ascending=False)
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 6))

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x="行缺失值数量", y="缺失数量对应的占比", data=row_null_df_groupby,
                label="行缺失值数量", color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 24), ylabel="",
           xlabel="{}-不同缺失值数量的占比条数".format(ds_type))
    sns.despine(left=True, bottom=True)
    return row_null_df


# 检查列缺失值，返回缺失值百分比的字典
def check_nan(df_var, show=True):
    print(df_var.shape)
    nan_result = df_var.isnull().sum(axis=0)
    col_name_list = df_var.columns.values
    result_dict = {k: v for k, v in zip(col_name_list, list(nan_result))}

    total = df_var.shape[0]

    nan_dict = dict()

    print("含有缺失值的列为：")
    for rd in result_dict.items():
        if rd[1] != 0:
            if show:
                print("{}: {}%".format(rd[0], round((rd[1] / total) * 100, 2)))
            nan_dict[rd[0]] = round((rd[1] / total) * 100, 2)

    return nan_dict


# 箱线图 连续类型特征（多选），枚举类型特征（单选）
def draw_box(df, value_fields=None, y_col=None, have_y=False):
    for value_field in value_fields:
        # data = pd.concat([df[value_field], df[y_col]], axis=1)
        plt.subplots(figsize=(15, 12))
        if have_y:
            fig = sns.boxplot(x=df[y_col], y=value_field, data=df)
        else:
            fig = sns.boxplot(x=None, y=value_field, data=df)
        plt.xlabel(y_col, fontdict={'weight': 'normal', 'size': 24})
        plt.ylabel(value_field, fontdict={'weight': 'normal', 'size': 24})
        plt.yticks(size=18)
        plt.xticks(size=18)
        fig.axis()
        plt.show()


# 填补缺失值 mean median most_frequent constant
def imputer_nan(df, axis=None, cols=None, missing_values="NaN", strategy='mean', fill_value=0):
    if strategy == "constant":
        if cols:
            part_a = df.drop(cols, axis=axis)
            part_b = df[cols]
            try:
                fill_value = float(fill_value)
            except TypeError:
                pass

            part_b = part_b.fillna(fill_value)
            # 合并
            df = pd.concat([part_a, part_b], axis=axis)
        else:
            df = df.fillna(fill_value)
    else:
        imp = Imputer(missing_values=missing_values, strategy=strategy)
        try:
            if cols:
                part_a = df.drop(cols, axis=axis)
                part_b = df[cols]
                if "object" in str(part_b.dtypes):
                    print("is object")
                part_b_name_list = part_b.columns.values
                part_b = imp.fit_transform(part_b)
                part_b = pd.DataFrame(part_b, columns=part_b_name_list)
                # 合并
                df = pd.concat([part_a, part_b], axis=axis)
            else:
                col_name_list = df.columns.values
                df = imp.fit_transform(df)
                df = pd.DataFrame(df, columns=col_name_list)
        except ValueError as e:
            print("部分列中存在多种类型的数据，请先转哑变量后再填充")
            raise e

    return df


# xgboost分类
def xgboost_classifier(x, y, max_depth=3, learning_rate=0.1, n_estimators=100, min_child_weight=1, gamma=0,
                       subsample=1, colsample_bytree=1, scale_pos_weight=1, random_state=27, reg_alpha=0, reg_lambda=1):
    if max_depth:
        max_depth = int(max_depth)
        if max_depth < 0:
            max_depth = None
    else:
        max_depth = None

    learning_rate = float(learning_rate)
    n_estimators = int(n_estimators)
    min_child_weight = int(min_child_weight)
    gamma = float(gamma)
    subsample = float(subsample)
    colsample_bytree = float(colsample_bytree)
    scale_pos_weight = int(scale_pos_weight)

    random_state = int(random_state)

    # 拟合XGBoost模型
    model = xgboost.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                                  min_child_weight=min_child_weight, gamma=gamma,
                                  subsample=subsample, colsample_bytree=colsample_bytree,
                                  scale_pos_weight=scale_pos_weight, random_state=random_state, reg_alpha=reg_alpha,
                                  reg_lambda=reg_lambda)

    model.fit(x, y)
    return model


# 划分训练集和测试集
def split_train_test(df, random_state=33, test_size=0.25):
    random_state = int(random_state)
    test_size = float(test_size)
    train, test = train_test_split(df, random_state=random_state, test_size=test_size)
    return train, test


def split_column(df, y="y"):
    try:
        X = df.drop(y, axis=1)
    except KeyError:
        raise KeyError("请在拆分列的参数中选择数据中有的字段")
    y = pd.DataFrame(df[y], columns=[y])
    return X, y


# 删除列
def delete_column(df, target=None, inverse=False):
    # 默认值
    if target is None:
        target = []
    # 所有列
    col_name_list = list(df.columns.values)
    # 两种情况
    if inverse:
        for t in target:
            col_name_list.remove(t)
        df = df.drop(col_name_list, axis=1)
    else:
        df = df.drop(target, axis=1)
    return df


# 处理时间格式数据
# 00:00 ==> 3800
# NaN   ==> 0
def tran_date(df_var, field_name):
    date_value = []
    for dlt in df_var[field_name]:
        if not isinstance(dlt, float):
            a, b = dlt.split(":")
            value = float(a) * 60 + float(b)
            date_value.append(3800 - value)
        else:
            date_value.append(0)
    df_var[field_name] = date_value
    return df_var


# 转换为one-hot编码
def OnehotEncoding(df, columns=None):
    if not columns:
        col_name_list = df.columns.values

        obj_list = []
        for index, d in enumerate(df.dtypes):
            if d == "object":
                obj_list.append(col_name_list[index])

        columns = obj_list

    data = df[columns]
    # 实例化OnehotEncoder
    enc = OneHotEncoder(categories="auto")
    # 生成目标特征列One_hot编码
    data_encoded = enc.fit_transform(data).toarray()
    # 生成新的列名
    new_columns = list(enc.get_feature_names())
    for i, column_name in enumerate(new_columns):
        df[column_name] = data_encoded[:, i]
    return df


# 自动分箱分析（卡方分箱，需要标签值，需要选定分成几个箱子）（展示分段）
def Chi_merge(df, x_col=None, y_col=None, k=6, rate=0.4):
    if not x_col:
        x_col = list(df.columns.values)

    pinf = float('inf')  # 正无穷大
    ninf = float('-inf')  # 负无穷大
    df_json = []

    bin_dict = dict()
    for col in x_col:
        print(col)
        col_result_dict = {}
        # 读取数据
        data = pd.concat([df[col], df[y_col]], axis=1)

        # 数据统计[feature1, class1, class2, class3]
        df_count = data.groupby(col)[y_col].value_counts().unstack().reset_index()
        df_count.fillna(0, inplace=True)

        # 去掉一些行，用于加快计算速度
        while df_count.shape[0] > 120:
            drop_row_index = []
            threshold = (data.shape[0] / df_count.shape[0]) * rate
            for i in range(df_count.shape[0] - 1):
                diff = 0
                for j in range(df_count.shape[1] - 1):
                    diff = abs(df_count.iloc[i, j + 1] - df_count.iloc[i + 1, j + 1]) + diff
                if diff < threshold:
                    drop_row_index.append(i + 1)
            df_count.drop(drop_row_index, inplace=True)
            len_df_count = df_count.shape[0]
            df_count.index = [i for i in range(len_df_count)]
            # print(len_df_count)

        df_count.index.name = 'index'
        n_class = df_count.shape[1] - 1
        num_interval = df_count.shape[0]
        print("num_interval: ", num_interval)
        max_col = pinf

        # 计算卡方值，合并删除，直到行数为k
        while (num_interval > k):
            chi_values = []
            drop_index = []
            for i in range(num_interval - 1):  # 制作表格，行列分别求和，用于计算卡方值
                data_chi = df_count.iloc[i:i + 2, 1:].copy().reset_index(drop=True)
                data_chi['sum'] = data_chi.apply(lambda x: x.sum(), axis=1)
                data_chi.loc[2] = data_chi.apply(lambda x: x.sum())
                for index in range(2):
                    for j in range(n_class):  # 计算卡方值
                        data_chi.iloc[index, j] = (data_chi.iloc[index, j] -
                                                   data_chi.iloc[index, -1] * data_chi.iloc[2, j] / data_chi.iloc[
                                                       2, -1]) ** 2 / \
                                                  (data_chi.iloc[index, -1] * data_chi.iloc[2, j] / data_chi.iloc[
                                                      2, -1])
                chi_value = sum(data_chi.iloc[0:-1, 0:-1].sum())
                chi_values.append(chi_value)
            min_chi = min(chi_values)  # 最小卡方值
            for i in range(num_interval - 2, -1, -1):  # 合并最小卡方值的行
                if chi_values[i] == min_chi:
                    df_count.iloc[i, 1:] = df_count.iloc[i, 1:] + df_count.iloc[i + 1, 1:]
                    drop_index.append(i + 1)
            df_count.drop(drop_index, inplace=True)  # 丢弃最小卡方值的行
            num_interval = df_count.shape[0]
            df_count.index = [i for i in range(num_interval)]
            df_count.iloc[0, 0] = ninf

        binning_list = list(df_count[col])
        binning_list.remove(binning_list[0])
        print(binning_list)

        bin_dict[col] = binning_list
    return bin_dict


def train_and_valid(train_x, train_y, valid_x, valid_y, n_estimators=88, max_depth=8, gamma=0, learning_rate=0.1,
                    reg_alpha=0, reg_lambda=1, random_state=27):
    # xgboost分类
    xg_model = xgboost_classifier(train_x, train_y, max_depth=max_depth, learning_rate=learning_rate,
                   n_estimators=n_estimators, min_child_weight=1, gamma=gamma,
                   subsample=1, colsample_bytree=1, scale_pos_weight=1,
                   random_state=random_state, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
    y_pred = xg_model.predict(valid_x)

    accuracy_score_result = metrics.accuracy_score(valid_y, y_pred)
    precision_score_result = metrics.precision_score(valid_y, y_pred, average='macro')
    recall_score_result = metrics.recall_score(valid_y, y_pred, average='macro')
    f1_score_result = metrics.f1_score(valid_y, y_pred, average='macro')

    #     print("accuracy_score_result: {}".format(accuracy_score_result))
    #     print("precision_score_result: {}".format(precision_score_result))
    #     print("recall_score_result: {}".format(recall_score_result))
    #     print("f1_score_result: {}".format(f1_score_result))

    return xg_model


# 数据类别
def check_columns_classifier(df, x_list):
    col_name_list = x_list
    classifier_list = []
    for i, name in enumerate(col_name_list):
        name_sum = df[name].value_counts().shape[0]
        classifier_list.append(name_sum)

    result_dict = {k: v for k, v in zip(col_name_list, classifier_list)}
    for rd in result_dict.items():
        print(rd)
    return None

# 类别分析(分析分类的占比的分布)
def check_classifier(df, col_name="Y"):
    result = df[col_name].value_counts()
    cols = result.index
    result_dict = {k: round(v / result.shape[0], 2) for k, v in zip(cols, list(result))}

    kinds = []
    nums = []
    for rd in result_dict.items():
        kinds.append(rd[0])
        nums.append(rd[1])

    plt.figure(figsize=(7, 8))
    plt.pie(nums, labels=kinds, autopct="%3.1f%%", startangle=60)
    plt.title("标签值的比例", fontsize=20)
    plt.legend(fontsize=12)
    plt.show()
    return None


