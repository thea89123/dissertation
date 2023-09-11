import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from matplotlib.ticker import FuncFormatter
from sklearn.inspection import partial_dependence


def descriptive_statistics(x, y, save_path=None):
    """
    This function takes in a DataFrame x and an array y, and returns a DataFrame containing descriptive statistics for each variable.
    """
    data = pd.concat([x, pd.Series(y, name='LST')], axis=1)
    desc = data.describe()
    if save_path:
        desc.to_excel(save_path)
    return desc

def autocorrelation_and_vif_analysis(x, y, save_path=None):
    """
    This function takes in a DataFrame x and an array y, performs autocorrelation and VIF analysis, and returns the results.
    """
    data = pd.concat([x, pd.Series(y, name='y')], axis=1)
    
    # Autocorrelation
    autocorr = data.apply(lambda col: col.autocorr(), axis=0)
    
    # Plotting the heatmap of autocorrelation
    plt.figure(figsize=(15, 9))
    sns.heatmap(autocorr.to_frame(), annot=True, cmap='coolwarm')
    plt.title('Autocorrelation')
    
    # VIF analysis
    X = add_constant(data)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    
    # Plotting the VIF
    # fig, ax = plt.subplots()
    # ax.bar(vif.index, vif)
    # ax.set_title('Variance Inflation Factor (VIF)')
    # ax.set_ylabel('VIF')
    
    result = {
        'autocorrelation': autocorr.to_dict(),
        'vif': vif.to_dict()
    }
    result = pd.DataFrame(result)
    
    if save_path:
        result.to_excel(save_path)
    
    return result

def pearson_correlation_heatmap(x, y, save_path=None):
    """
    This function takes in a DataFrame data, and plots a heatmap of the Pearson correlation coefficients.
    """
    data = pd.concat([x, pd.Series(y, name='LST')], axis=1)
    corr = data.corr()
    
    # Plotting the heatmap of Pearson correlation coefficients
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={'fontsize': 10})

    
    plt.title('Pearson Correlation Coefficients')
    
    if save_path:
        plt.savefig(save_path, dpi=500)
    
    plt.show()
    


def boosted_regression_tree(x, y, save_path=None, save_path1=None, save_path2=None):
    # 拟合模型
    model = GradientBoostingRegressor()
    model.fit(x, y)
    
    # 预测
    y_pred = model.predict(x)
    
    # 计算r2和rmse
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # 保存
    x['LST'] = y
    x['LSTP'] = y_pred
    if save_path:
        x.to_excel(save_path)
    
    # 绘制特征重要性图
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(12, 9))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), x.columns[sorted_idx])
    plt.xlabel('Importance (%)')
    plt.title('Feature Importance')
    
    # 在条形图上添加百分比标签
    for i, v in enumerate(feature_importance[sorted_idx]):
        plt.text(v + 0.01, i - 0.25, f'{v*100:.1f}%', color='black', fontweight='bold')
    
    # 设置x轴刻度为百分比形式
    def to_percent(x, position):
        return f'{x*100:.0f}%'
    
    formatter = FuncFormatter(to_percent)
    plt.gca().xaxis.set_major_formatter(formatter)
    
    if save_path1:
        plt.savefig(save_path1, dpi=500)
    
    plt.show()
    

    for i in range(len(x.columns[:-2])):
        result = partial_dependence(model, x, [i], percentiles=(0, 1))
        pdp = result['average']
        axes = result['values']
        plt.plot(axes[0], pdp.reshape(-1))
        plt.plot(np.linspace(0, axes[0].max(), 1000), [0]*1000, color='red', linestyle='dashed')
        plt.xlabel(x.columns[i])
        if save_path2:
            plt.savefig(save_path2+x.columns[i]+'.png')
        plt.show()
    
    return r2, rmse

def plot_density(dfs, column, save_path=None):
    # 对于每个DataFrame，绘制指定列的密度函数图
    for i, df in dfs.items():
        
        sns.kdeplot(data=df[column], shade=True, label=i)
    
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=500)
    
    plt.show()
    
if __name__ == '__main__':
    
    print('-'*30, 'start', '-'*30)
    r2_, rmse_, scale = [], [], []
    data_file = os.listdir('DATA')
    dfs = {}
    
    mapping = pd.read_excel(r'mapping.xlsx', header=None)
    mapping = mapping.iloc[1, 1:].tolist()


    for data_name in data_file:
        data_path = os.path.join('DATA', data_name)
        
        _data_name = data_name.split('_')[1].split('.')[0]
        data = pd.read_excel(data_path)
        
        x = data.iloc[:, 3:17]
        x.columns = mapping
        y = data['LST']
        
        dfs[_data_name] = x
        
        descriptive_statistics(x, y, f'TABLE//{_data_name}_descriptive_statistics.xlsx')
        autocorrelation_and_vif_analysis(x, y, f'TABLE//{_data_name}_autocorr_vif.xlsx')
        pearson_correlation_heatmap(x, y, f'PLOT//{_data_name}_corr.png')
        r2, rmse = boosted_regression_tree(x, y, f"DATA//{_data_name}_pred.xlsx", 
                                           f"PLOT//{_data_name}_feature_importance.png",
                                           f"PLOT//{_data_name}_", )
        r2_.append(r2)
        rmse_.append(rmse)
        scale.append(_data_name)
    
    for i in x.columns:
        plot_density(dfs, i, f'PLOT//{i}_distribution.png')
    
    Performance = pd.DataFrame({'scale': scale,
                                'r2': r2_,
                                'rmse': rmse_
                                })
    Performance.to_excel('TABLE//Performance.xlsx')
    
    print('-'*30, 'end', '-'*30)


