import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats.stats as stats
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

def make_M1target(df, column_name):
    df['m1'] = df.apply(lambda x: 1 if x[column_name] > 30 else 0, axis=1)
    return df


def make_var_list(df):
    var_list = []
    for i in df.columns:
        if i.find('var') == 0:
            var_list.append(i)
    return var_list


def calIV(df, var, target):
    eps = 0.00001
    gbi = pd.crosstab(var, target) + eps
    gb = target.value_counts() + eps
    gbri = gbi / gb
    gbri['woe'] = np.log(gbri[1] / gbri[0])
    gbri['iv'] = (gbri[1] - gbri[0]) * gbri['woe']
    return gbri['iv'].sum()


def mono_bin(Y, X, n=20):
    X2 = X.fillna(np.median(X))
    r = 0
    while np.abs(r) < 1:  # 自动分割阈值
        d1 = pd.DataFrame({"X": X2, "Y": Y, "Bucket": pd.qcut(X2, n, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)  # pearsonr,spearmanr,kendalltau
        n = n - 1
    d3 = pd.DataFrame(d2.min().X, columns=['min_' + X.name])
    d3['min_'] = d2.min().X
    d3['max_'] = d2.max().X
    d3[Y.name] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['m1_rate'] = round(d2.mean().Y, 4)
    return d3


def var_classification(df, var_list):
    var_classification_dict = {}
    classification_error_dict = {}

    for i in var_list:
        if df[i].drop_duplicates(keep='last').count() == 1:
            classification_error_dict[i] = 'classification error:variable distinct count 1'
        elif df[i].dtype == 'O' and df[i].drop_duplicates(keep='last').count() > 20:
            classification_error_dict[i] = 'classification error:too many classified variable distinct count '
        elif df[i].dtype == 'O' and df[i].drop_duplicates(keep='last').count() <= 20:
            var_classification_dict[i] = 'classified'
        elif df[i].dtype != 'O' and df[i].drop_duplicates(keep='last').count() <= 5:
            var_classification_dict[i] = 'classified'
        else:
            var_classification_dict[i] = 'continuous'
    return var_classification_dict, classification_error_dict


def variable_analyzing(df, var_classification_dict):
    variable_list = []
    analyzing_error_dict = {}

    for i in var_classification_dict:
        print('Now:', i)

        # 缺失率
        df_reshape = df[['m1', i]]
        missing_count = len(df_reshape['m1']) - df_reshape[i].count()
        missing_rate = round(1 - (df_reshape[i].count() / len(df_reshape['m1'])), 4)
        missing_df = df_reshape[df_reshape[i].isnull().values == True]
        if len(missing_df['m1']) > 0:
            missing_m1_rate = round(sum(missing_df['m1']) / len(missing_df['m1']), 4)
        else:
            missing_m1_rate = '-'

        # 去除缺失值
        df_reshape = df_reshape.dropna(axis=0).reset_index()
        variable = df_reshape[i]

        if var_classification_dict[i] == 'continuous':
            variable_describe = str(round(variable.min(), 2)) + '|' + str(round(variable.max(), 2)) + '|' + str(
                round(variable.mean(), 2)) + '|' + str(round(variable.median(), 2))
        else:
            variable_describe = '-'

        try:
            if var_classification_dict[i] == 'continuous':
                # 方差检验
                table = sm.stats.anova_lm(ols('m1 ~ variable', data=df_reshape).fit())
                # 笔数逾期分组
                bins = mono_bin(df_reshape['m1'], variable)
                bins_json = bins.loc[:, ['min_', 'max_', 'm1_rate', 'm1', 'total']].to_json(orient='records')
                # 笔数IV计算
                bins_IV_list = list(bins['max_'])
                bins_IV_list.insert(0, -9999)
                bins_IV = calIV(df_reshape, pd.cut(variable, bins_IV_list, right='True'), df_reshape['m1'])
                # 入LIST
                variable_list.append(
                    [i, 'continuous', table['PR(>F)'].iloc[0], variable_describe, missing_count, missing_rate,
                     missing_m1_rate, bins_IV, bins_json])

            elif var_classification_dict[i] == 'classified':
                # 卡方检验
                t1 = pd.crosstab(df_reshape['m1'], variable)
                g, p, dof, expctd = chi2_contingency(t1)
                # 笔数逾期分组
                t2 = df_reshape['m1'].groupby(variable).agg({'count', 'sum', 'mean'}).reset_index()
                t2.columns = (['group_name', 'm1_total', 'm1_rate', 'm1'])
                bins_json = t2.to_json(orient='records', force_ascii=False)
                # 笔数IV计算
                bins_IV = calIV(df_reshape, variable, df_reshape['m1'])
                # 入LIST
                variable_list.append(
                    [i, 'classified', p, variable_describe, missing_count, missing_rate, missing_m1_rate, bins_IV,
                     bins_json])

        except Exception as e:
            analyzing_error_dict[i] = e

    return variable_list, analyzing_error_dict


def sort_report_list(variable_list):
    variable_list.sort(key=lambda x: x[7], reverse=True)
    result_dataFrame = pd.DataFrame(variable_list,
                                    columns=['variable', 'variable_type', 'P_value', 'min|max|mean|median',
                                             'missing_count', 'missing_rate', 'missing_m1_rate', 'IV', 'group_bins'])
    return result_dataFrame


def corr_heatmap(df, result_dataFrame, limit_IV, figsave_path):

    corr_var_list = list(result_dataFrame['variable'][result_dataFrame['IV'] >= limit_IV])
    if len(corr_var_list) >= 2:

        corr_df = df[corr_var_list]
        corr = corr_df.corr()

        xticks = list(corr.index)
        yticks = list(corr.index)
        fig = plt.figure(figsize=(len(xticks) * 1.8, len(xticks) * 1.2))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        ax1 = fig.add_subplot(1, 1, 1)
        sns.heatmap(corr, annot=True, cmap=cmap, ax=ax1, linewidths=0.1,
                    annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
        ax1.set_xticklabels(xticks, rotation=45, fontsize=10)
        ax1.set_yticklabels(yticks, rotation=0, fontsize=10)

        fig.savefig(figsave_path)

    else:
        print('corr_heatmap_failed')


def main_loop(file_path,result_output_path):
    print('read file....')
    df = pd.read_excel(file_path)
    df = make_M1target(df, 'overdue_days_correct')

    print('variable classification....')
    var_list = make_var_list(df)
    var_classification_dict, classification_error_dict = var_classification(df, var_list)

    print('variable analyzing....')
    variable_list, analyzing_error_dict = variable_analyzing(df, var_classification_dict)
    result_dataFrame = sort_report_list(variable_list)

    print('output....')
    result_dataFrame.to_excel(result_output_path + '\\' + 'result.xlsx')
    error_dict = dict(classification_error_dict, **analyzing_error_dict)
    print('classification_error_dict:', error_dict)

    print('draw corr heatmap....')
    corr_heatmap(df, result_dataFrame, 0.02, result_output_path + '\\' + 'corr_heatmap.png')

    print('done!')


if __name__ == '__main__':

    file_path = r'C:\Users\jizeyuan\Desktop\20200804_晋松网络信息技术有限公司_输出\20200804_晋松网络信息技术有限公司_输出\763评分\IOS_安卓\763评分_只保留IOS安卓.xlsx'
    result_output_path = r'C:\Users\jizeyuan\Desktop\20200804_晋松网络信息技术有限公司_输出\20200804_晋松网络信息技术有限公司_输出\763评分\IOS_安卓'

    main_loop(file_path, result_output_path)
