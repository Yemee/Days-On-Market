import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import statsmodels.api as sm

def classifier_metrics(ytest, yhat):
    '''returns dictionary of classifier metrics'''
    metrics = [accuracy_score, precision_score, recall_score,
               f1_score, brier_score_loss, roc_auc_score, confusion_matrix]
    names = ['acc', 'prec', 'rec', 'f1', 'brier', 'roc_auc', 'confusion_matrix']       
    d = {}
    for name, metric in zip(names, metrics):
        m = metric(y_test, y_hat)
        d[name] = m
        if metric == confusion_matrix:
            d['specificity'] = m[0][0]/(m[0][0] + m[0][1])
    return d

def plot_seasonal_decomposition(axs, series, sd):
        axs[0].plot(series.index, series)
        axs[0].set_title("Raw Series")
        axs[0].set_xticks(series.index)
        axs[0].set_xticklabels(series.index, rotation=90)
        axs[1].plot(series.index, sd.trend)
        axs[1].set_title("Trend Component $T_t$")
        axs[1].set_xticks(series.index)
        axs[1].set_xticklabels(series.index, rotation=90)
        axs[2].plot(series.index, sd.seasonal)
        axs[2].set_title("Seasonal Component $S_t$")
        axs[2].set_xticks(series.index)
        axs[2].set_xticklabels(series.index, rotation=90)
        axs[3].plot(series.index, sd.resid)
        axs[3].set_title("Residual Component $R_t$")
        axs[3].set_xticks(series.index)
        axs[3].set_xticklabels(series.index, rotation=90)

if __name__=="__main__":
    # load data and create train & validation sets
    data =  pd.read_csv('../data/train.csv')
    
    df = data[['quick', 'ratio', 'is_realtor', 'is_warm', 'is_corp_owner']]
    y = df.pop('quick')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2,
                                                        random_state=42)
    
    # Classifier models (3)
    # 1) sklearn logmod
    logmod = LogisticRegression()
    logmod.fit(X_train, y_train)
    y_prob = logmod.predict_proba(X_test)[:,1]
    y_hat = y_prob>0.5
    class_df = pd.DataFrame.from_dict(classifier_metrics(y_test, y_hat),
                                      orient='index', columns=['logmod'])
    
    # 2) statsmodels logmod and summary report
    logm = sm.Logit(y_train,sm.add_constant(X_train.astype(float))).fit()
    print(logm.summary())
    sm_pred_proba = logm.predict(sm.add_constant(X_test.astype(float)))
    y_hat = sm_pred_proba>0.5
    class_df['sm_logm'] = pd.DataFrame.from_dict(classifier_metrics(y_test, y_hat), orient='index')
    
    # 3) gradient boost + feature importances + permutation importances
    gb = GradientBoostingClassifier(n_estimators=100,                                                                 learning_rate=1.0,
                                    max_depth=1)
    gb.fit(X_train, y_train)
    y_prob = gb.predict_proba(X_test)[:,1]
    y_hat = y_prob>0.5
    class_df['gb'] = pd.DataFrame.from_dict(classifier_metrics(y_test, y_hat),                                                 orient='index')
    print(class_df)
    
    # feature importances
    sorted_idx = np.argsort(gb.feature_importances_)
    y_ticks = np.arange(0, len(X.columns))
    fig, ax = plt.subplots()
    ax.barh(y_ticks, gb.feature_importances_[sorted_idx])
    ax.set_yticklabels(X.columns[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title("gb fi")
    fig.tight_layout()
    fig.savefig('../images/classifier_feature_importances.png')
    plt.show()
    
    # permutation importances
    result = permutation_importance(gb, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[perm_sorted_idx].T,
               vert=False, labels=X_test.columns[perm_sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    fig.savefig('../images/classifier_permutation_importances.png')
    plt.show()
    
    
    # Time Series Decomposition to deseason dtp and detrend dtc
    # monthly window
    df_ts = data[['dtp', 'pcd_mo_year']]
    df_ts = df_ts.groupby('pcd_mo_year').mean()
    decomp_dtp = sm.tsa.seasonal_decompose(df_ts['dtp'], period=12)
    df_ts['dtp_trend'] = decomp_dtp.trend
    df_ts['dtp_seasonal'] = decomp_dtp.seasonal
    df_ts['dtp_resid'] = decomp_dtp.resid
    fig, axs = plt.subplots(4, figsize=(14, 8))
    plot_seasonal_decomposition(axs, df_ts.dtp, decomp_dtp)
    fig.tight_layout()
    fig.savefig('../images/dtp_mean_decomposition.png')
    plt.show()
    
    df_ts_dtc = data[['dtc', 'pcd_mo_year']]
    df_ts_dtc = df_ts_dtc.groupby('pcd_mo_year').mean()
    decomp_dtc = sm.tsa.seasonal_decompose(df_ts_dtc['dtc'], period=12)
    df_ts['dtc_trend'] = decomp_dtc.trend
    df_ts['dtc_seasonal'] = decomp_dtc.seasonal
    df_ts['dtc_resid'] = decomp_dtc.resid
    print(df_ts.head())
    fig, axs = plt.subplots(4, figsize=(14, 8))
    plot_seasonal_decomposition(axs, df_ts_dtc.dtc, decomp_dtc)
    fig.tight_layout()
    fig.savefig('../images/dtc_mean_decomposition.png')
    plt.show()
    
    # Add decompositon data to regression data
    df_reg = data[['dtp', 'dtc', 'ratio', 'is_realtor', 'is_warm', 'is_corp_owner', 'with_cash', 'pcd_mo_year']]
    df_reg = df_reg.set_index('pcd_mo_year').join(df_ts, rsuffix='ts')
#     df_reg = df_reg.dropna()
    print(df_reg.head())
    
    # dtp regression with GLM
    df_dtp = df_reg[['dtp', 'ratio', 'is_realtor', 'is_warm', 'is_corp_owner', 'dtp_seasonal']]
    df_dtp['sans_season'] = df_dtp.dtp - df_dtp.dtp_seasonal
    # model without seasonal piece, then add back in. Exclude is_warm b/c same information
    df_dtp = df_dtp[['sans_season', 'ratio', 'is_realtor', 'is_corp_owner']]
    y = df_dtp.pop('sans_season')
    print(y.min(), y.max(), y.mean(), y.var())
    plt.hist(y, bins = 50)
    plt.show()
    X = df_dtp
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2,
                                                        random_state=42)
#     dtp_mod = sm.GLM(y_train, sm.add_constant(X_train.astype(float)), sm.families.NegativeBinomial(alpha=8)).fit()
#     print(dtp_mod.summary())

#     # g_yhat = (g_mod.predict(X_test))**2 + s_test
#     g_yhat = g_mod.predict(X_test) + s_test
#     rmse = np.sqrt(mean_squared_error(dtp_test, g_yhat))
#     mae = mean_absolute_error(dtp_test, g_yhat)
#     print(f'rmse: {rmse}')
#     print(f'mae: {mae}')