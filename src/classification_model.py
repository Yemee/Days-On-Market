import numpy as np
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


def quad_form(a,b,c):
    '''from (1-p)/'''
    z1 = (-b + np.sqrt((b**2)-(4*a*c)))/(2*a)
    z2 = (-b - np.sqrt((b**2)-(4*a*c)))/(2*a)
    return z1, z2

def log_likelihood_poisson(lam, data):
    poisson = stats.poisson(lam)
    likelihoods = poisson.pmf(data)
    log_likelihoods = np.log(likelihoods)
    log_likelihood_sum = np.sum(log_likelihoods)
    return(log_likelihood_sum)

def pois_mle(ax, start, stop, data, clr='yellowgreen'):
    poisson_log_likelihood = []
    search_space = np.linspace(start,stop,100)
    data=data
    for lam in search_space:
        poisson_log_likelihood.append(log_likelihood_poisson(lam, data))
    max_idx = np.argmax(poisson_log_likelihood) 
    mle = search_space[max_idx]
    highest_log_pois = poisson_log_likelihood[max_idx]
#     fig, ax = plt.subplots(1, 1)
    ax.plot(search_space, poisson_log_likelihood, color=clr)
    ax.axhline(highest_log_pois, ls = '--',c='m', label = 'Derivative = 0')
    ax.set_title('MLE $\lambda$: {}'.format(round(search_space[max_idx],4)))
    ax.set_xlabel("parameter")
    ax.set_ylabel("log likelihood")
    ax.legend();
    return search_space[max_idx]

def log_likelihood_geom(p,data):
    geom = stats.geom(p)
    likelihoods = geom.pmf(data)
    log_likelihoods = np.log(likelihoods)
    log_likelihood_sum = np.sum(log_likelihoods)
    return log_likelihood_sum

def geom_mle(ax, start, stop, data):
    geom_log_likelihood = []
    search_space = np.linspace(start,stop,100)
    data=data
    for p in search_space:
        geom_log_likelihood.append(log_likelihood_geom(p, data))
    max_idx = np.argmax(geom_log_likelihood) 
    mle = search_space[max_idx]
    highest_log_geom =geom_log_likelihood[max_idx]
    ax.plot(search_space, geom_log_likelihood, color='yellowgreen')
    ax.axhline(highest_log_geom, ls = '--',c='m', label = 'Derivative = 0')
    ax.set_title('MLE p: {}'.format(round(search_space[max_idx],4)))
    ax.set_xlabel("parameter")
    ax.set_ylabel("log likelihood")
    ax.legend();
    return search_space[max_idx]

def log_likelihood_binom(p, data):
    binom = stats.binom(n=77, p=p)
    likelihoods = binom.pmf(data)
    log_likelihoods = np.log(likelihoods)
    log_likelihood_sum = np.sum(log_likelihoods)
    return(log_likelihood_sum)

def binom_mle(ax, start, stop, data, clr='violet'):
    binom_log_likelihood = []
    search_space = np.linspace(0,1, 100)
    data=data
    for p in search_space:
        binom_log_likelihood.append(log_likelihood_binom(p, data))
    max_idx = np.argmax(binom_log_likelihood) 
    mle = search_space[max_idx]
    highest_log_binom = binom_log_likelihood[max_idx]
#     fig, ax = plt.subplots(1, 1)
    ax.plot(search_space, binom_log_likelihood, color=clr)
    ax.axhline(highest_log_binom, ls = '--',c='m', label = 'Derivative = 0')
    ax.set_title('MLE p: {}'.format(round(search_space[max_idx],4)))
    ax.set_xlabel("parameter")
    ax.set_ylabel("log likelihood")
    ax.legend();
    return search_space[max_idx]


class LogMod():
    def __init__(self, df, params={'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01}):

        self.df = df
        # self.y = self.df.pop('quick')
        # self.X = self.df
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        self.params = params

    def cv(self):
        df = self.df.copy()
        y = df.pop('quick')
        X = df
        clf = LogisticRegressionCV(cv=5, random_state=42).fit(X, y)
        print(f'5-fold CV logistic regression = {clf.score(X, y)}')
        return clf.score(X, y)

    def metric_func(self, y_pred, y_test, method):
        metrics = [accuracy_score, precision_score, recall_score,
                f1_score, brier_score_loss, roc_auc_score, confusion_matrix]
        d1 = {}
        for metric in metrics:
            m = metric(y_pred, y_test)
            d1[metric]=m
        out_df = pd.DataFrame.from_dict(d1, orient='index')
        out_df['metric'] = ['acc', 'prec', 'rec', 'f1', 'brier', 'roc_auc', 'confusion_matrix']
        out_df.set_index('metric', inplace=True)
        out_df[method] = out_df.values
        out_df.pop(0)
        return out_df

    def res(self):
        df = self.df.copy()
        y = df.pop('quick')
        X = df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        clf = LogisticRegression().fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:,1]
        print(self.metric_func(y_hat, y_test, 'logistic_regression'))
        return self.metric_func(y_hat, y_test, 'logistic_regression'), y_hat, y_prob

    def L1_reg(self):
        df = self.df.copy()
        y = df.pop('quick')
        X = df
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        l1_ratio = 0.5 #L1 weight in elastic net regularization
        clf_l1_LR = LogisticRegression(penalty='l1', tol=0.01, solver='saga').fit(X_train, y_train)
        clf_l2_LR = LogisticRegression(penalty='l2', tol=0.01, solver='saga').fit(X_train, y_train)
        clf_en_LR = LogisticRegression(penalty='elasticnet', solver='saga',
                                   l1_ratio=l1_ratio, tol=0.01).fit(X_train, y_train)
        print(f'L1 = {clf_l1_LR.score(X_test, y_test)}')
        print(f'L2 = {clf_l2_LR.score(X_test, y_test)}')
        print(f'EN = {clf_en_LR.score(X_test, y_test)}')
        y_hat = clf_en_LR.predict(X_test)
        print(self.metric_func(y_hat, y_test, 'Elastic Net'))
        return self.metric_func(y_hat, y_test, 'Elastic Net'), y_hat, clf_en_LR.predict_proba(X_test)
        
    def sm(self):
        df = self.df.copy()
        y = df.pop('quick')
        X = df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        logmod = sm.Logit(y_train, sm.add_constant(X_train.astype(float))).fit()
        print(logmod.summary())
        sm_pred_proba = logmod.predict(sm.add_constant(X_test.astype(float)))
        sm_pred = sm_pred_proba>0.5
        print(self.metric_func(sm_pred, y_test, 'sm logmod'))
        return self.metric_func(sm_pred, y_test, 'sm logmod'), sm_pred, sm_pred_proba

    def gb(self):
        df = self.df.copy()
        y = df.pop('quick')
        X = df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        gb = GradientBoostingClassifier(**self.params).fit(X_train, y_train)
        y_hat = gb.predict(X_test)
        y_prob = gb.predict_proba(X_test)[:,1]
        print(self.metric_func(y_hat, y_test, 'gb'))
        return self.metric_func(y_hat, y_test, 'gb'), y_hat, y_prob

    def cv_gb(self):
        df = self.df.copy()
        y = df.pop('quick')
        X = df
        gb = GradientBoostingClassifier(**self.params)
        res_gb = cross_validate(gb, X, y, cv=5)
        # res_gb2 = cross_validate(gb, self.X, self.y, cv=5))
        # print(f"gb cv mae results: {res_gb['test_score']}")
        # print(f"gb cv rmse results: {res_gb2['test_score']}")
        # print(f"mean gb mae cv: {res_gb['test_score'].mean()}")
        # print(f"mean gb rmse cv: {res_gb2['test_score'].mean()}")
        # return res_gb['test_score'].mean(), res_gb2['test_score'].mean()
        print(f"5-fold CV gb = {res_gb['test_score'].mean()}")
        return res_gb['test_score'].mean()

    def grid_search_best(self, param_grid):
        df = self.df.copy()
        y = df.pop('quick')
        X = df
        gb = GradientBoostingClassifier()
        grid_search = GridSearchCV(estimator = gb, param_grid = param_grid,
                          cv = 5, n_jobs = -1, verbose = 2)
        grid_search.fit(X, y)
        print(grid_search.best_params_)
        print(grid_search.cv_results_['mean_test_score'])
        return grid_search.best_params_
    
    def results(self):
        a = self.res()[0]
        b = self.L1_reg()[0]
        c = self.sm()[0]
        d = self.gb()[0]
        return a.join(b).join(c).join(d)
    
    def holdout(self, holdout_df):
        df = self.df.copy()
        y_train = df.pop('quick')
        X = df
        y_test = holdout_df.pop('quick')
        X_test = holdout_df
        X_train = np.array(X)
        X_test = np.array(X_test)
        logmod = sm.Logit(y_train, sm.add_constant(X_train.astype(float))).fit()
        print(logmod.summary())
        sm_pred_proba = logmod.predict(sm.add_constant(X_test.astype(float)))
        sm_pred = sm_pred_proba>0.5
        print(self.metric_func(sm_pred, y_test, 'sm logmod'))
        return self.metric_func(sm_pred, y_test, 'sm logmod'), sm_pred, sm_pred_proba
 

if __name__=="__main__":

    T = pd.read_csv('../data/train.csv')
    H = pd.read_csv('../data/holdout.csv')

    T['broken_stick'] = [br_stick(x, 5) for x in T.pcd_month]
    H['broken_stick'] = [br_stick(x, 5) for x in H.pcd_month]

    T['seasonality'] = T.broken_stick
    T['ratio'] = T.ratio*100
    dtp_df = T[['quick', 'ratio', 'is_realtor',  
                  'period_turn', 'seasonality', 'with_cash', 'is_corp_owner']]

    H['seasonality'] = H.broken_stick
    H['ratio'] = H.ratio*100
    dtpH_df = H[['quick', 'ratio', 'is_realtor',  
                  'period_turn', 'is_warm', 'seasonality', 'is_corp_owner']]

     
    LogMod(dtp_df).cv()
    LogMod(dtp_df).cv_gb()
    LogMod(dtp_df).L1_reg()
    LogMod(dtp_df).res()
    LogMod(dtp_df).sm()
    LogMod(dtp_df).gb()
    print(LogMod(dtp_df).results())

    # get best params
    param_grid = {
        'max_depth': [3,4],
        'min_samples_split': [2,5],
        'n_estimators': [100,500]
        }
    LogMod(dtp_df).grid_search_best(param_grid)

    # redo with best params
    params={'n_estimators': 100,
          'max_depth': 3,
          'min_samples_split': 5,
          'learning_rate': 0.01}
    LogMod(dtp_df, params=params).gb()



