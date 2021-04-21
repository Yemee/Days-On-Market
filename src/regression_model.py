import numpy as np
import pandas as pd
import sklearn.utils
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import plot_partial_dependence

import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 25})
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['figure.figsize'] = (20,8)
plt.rcParams['savefig.dpi'] = 300


def mae_rmse(yhat, ytest):
    mae = mean_absolute_error(ytest, yhat)
    rmse = np.sqrt(mean_squared_error(ytest, yhat))
    print(f'the MAE is {mae}')
    print(f'the RMSE is {rmse}')
    return mae, rmse



class NB_GB():
    def __init__(self, df, target, k=5, alpha=9, params={'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}):

        self.df = df
        self.copy_df = df.copy()
        self.target = target
        self.k = k
        self.alpha = alpha

        self.y = self.df.pop(self.target)
        self.X = self.df

        self.mean = np.mean(self.y) 
        self.median = np.median(self.y) 

        self.params = params

    def cv_gb(self):
        gb = GradientBoostingRegressor(**self.params)
        res_gb = cross_validate(gb, self.X, self.y, cv=5, scoring=('neg_mean_absolute_error'))
        res_gb2 = cross_validate(gb, self.X, self.y, cv=5, scoring=('neg_root_mean_squared_error'))
        print(f"gb cv mae results: {res_gb['test_score']}")
        print(f"gb cv rmse results: {res_gb2['test_score']}")
        print(f"mean gb mae cv: {res_gb['test_score'].mean()}")
        print(f"mean gb rmse cv: {res_gb2['test_score'].mean()}")
        return res_gb['test_score'].mean(), res_gb2['test_score'].mean()
    
    def k_fold(self):
        temp = sklearn.utils.shuffle(self.copy_df)
        m = int(self.df.shape[0]/self.k)
        d = {}
        for num in range(self.k):
            d[num] = temp.iloc[num:num+m,:]
        return d

    def one_fold(self, i):
        split_dict = self.k_fold()
        k_lst = list(range(len(split_dict)))
        tst = split_dict[k_lst.pop(i)]
        d={}
        d ={num: split_dict[num] for num in k_lst}
        trn = pd.concat(d.values())
        a = trn.copy()
        b = tst.copy()
        y_train = a.pop(self.target)
        y_test = b.pop(self.target)
        x_train = np.array(a)
        x_test = np.array(b)
        nb_dtp = sm.GLM(y_train, sm.add_constant(x_train.astype(float)), family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()
        mae_nb = mean_absolute_error(y_test,nb_dtp.predict(sm.add_constant(x_test.astype(float))))
        rmse_nb = np.sqrt(mean_squared_error(y_test, nb_dtp.predict(sm.add_constant(x_test.astype(float)))))
        return mae_nb, rmse_nb
    
    def cv_nb(self):
        maes = []
        rmses = []
        for i in range(self.k):
            m, r = self.one_fold(i)
            maes.append(m)
            rmses.append(r)
        print(f"nb mae cv results: {maes}")
        print(f"nb rmse cv results: {rmses}")
        print(f"mean nb mae cv: {np.mean(maes)}")
        print(f"mean nb rmse cv: {np.mean(rmses)}")
        return np.mean(maes), np.mean(rmses)
    

    def res(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        X_train = np.array(X_train)

        nb_dtp = sm.GLM(y_train, sm.add_constant(X_train.astype(float)), family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()
        print(nb_dtp.summary())
        gb_dtp = GradientBoostingRegressor(**self.params).fit(X_train, y_train)

        mae_nb = mean_absolute_error(y_test,nb_dtp.predict(sm.add_constant(X_test.astype(float))))
        rmse_nb = np.sqrt(mean_squared_error(y_test, nb_dtp.predict(sm.add_constant(X_test.astype(float)))))

        mae_gb = mean_absolute_error(y_test,gb_dtp.predict(X_test))
        rmse_gb = np.sqrt(mean_squared_error(y_test, gb_dtp.predict(X_test)))

        mae_mean = mean_absolute_error(y_test, [self.mean]*len(y_test))
        rmse_mean = np.sqrt(mean_squared_error(y_test, [self.mean]*len(y_test)))          
        
        mae_median = mean_absolute_error(y_test, [self.median]*len(y_test))
        rmse_median = np.sqrt(mean_squared_error(y_test, [self.median]*len(y_test)))

        print("NB: The mean absolute error (MAE) on test set: {:.4f}".format(mae_nb))
        print("NB: The root mean squared error (RMSE) on test set: {:.4f}".format(rmse_nb))
        print("GB: The mean absolute error (MAE) on test set: {:.4f}".format(mae_gb))
        print("GB: The root mean squared error (RMSE) on test set: {:.4f}".format(rmse_gb))
        print("MEAN: The mean absolute error (MAE) on test set: {:.4f}".format(mae_mean))
        print("MEAN: The root mean squared error (RMSE) on test set: {:.4f}".format(rmse_mean))
        print("MEDIAN: The mean absolute error (MAE) on test set: {:.4f}".format(mae_median))
        print("MEDIAN: The root mean squared error (RMSE) on test set: {:.4f}".format(rmse_median))
        
        return nb_dtp.predict(sm.add_constant(X_test.astype(float))), gb_dtp.predict(X_test)

    def gb(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        X_train = np.array(X_train)
        gb_dtp = GradientBoostingRegressor(**self.params).fit(X_train, y_train)   
        return gb_dtp

    def feat_import(self, ax):
        gb_dtp = self.gb()
        sorted_idx = np.argsort(gb_dtp.feature_importances_)
        y_ticks = np.arange(0, len(self.X.columns))
        ax.barh(y_ticks, gb_dtp.feature_importances_[sorted_idx], color='dodgerblue')
        ax.set_yticklabels(self.X.columns[sorted_idx])
        ax.set_yticks(y_ticks)
        ax.set_title('Gradient Boost Feature Importance')

    def perm_import(self, ax):  
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        X_train = np.array(X_train)
        gb_dtp = GradientBoostingRegressor(**self.params).fit(X_train, y_train)
        result = permutation_importance(gb_dtp, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
        perm_sorted_idx = result.importances_mean.argsort()
        ax.boxplot(result.importances[perm_sorted_idx].T,
                vert=False, labels=X_test.columns[perm_sorted_idx])
        ax.set_title('Gradient Boost Bootstrapped Permutation Test')
        ax.axvline(x=0, linestyle=':', color='r')
    
    def plot_dev(self, ax):
        pass

    def partial_dep(self, ax):
        pass

    def choose_alpha(self, ax):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        X_train = np.array(X_train)

        alphas = np.arange(0.01,3)
        maes = []
        devs = []
        for a in alphas:
            nb_dtp = sm.GLM(y_train, sm.add_constant(X_train.astype(float)), family=sm.families.NegativeBinomial(alpha=a)).fit()
            maes.append(mean_absolute_error(y_test,nb_dtp.predict(sm.add_constant(X_test.astype(float)))))
            devs.append(nb_dtp.deviance)
        ax.plot(alphas, maes, color='red', linestyle='--')
        ax.set_ylabel('mean absolute error', color='red')
        ax.set_xlabel('alpha dispersion parameters')
        ax2 = ax.twinx()
        ax2.plot(alphas, devs, color='blue', linestyle='--')
        ax2.set_ylabel('deviation', color='blue')
        ax.set_title('Choose Alpha')
#         ax.axvline(x=1, color='k', label='alpha = 1', linestyle=':')
        ax.legend(fontsize=20, loc='center')
    
    def grid_search_best(self, param_grid):
        gb = GradientBoostingRegressor()
        grid_search = GridSearchCV(estimator = gb, param_grid = param_grid,
                          cv = 5, n_jobs = -1, verbose = 2, scoring = ('neg_mean_absolute_error'))
        grid_search.fit(self.X, self.y)
        print(grid_search.best_params_)
        print(grid_search.cv_results_['mean_test_score'])
        return grid_search.best_params_

class HoldOut():
    def __init__(self, T, H, target, alpha = 9, params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}):
        self.T = T
        self.H = H
        self.target = target
        self.alpha = alpha

        self.y_train = self.T.pop(self.target)
        self.X_train = self.T
        self.y_test = self.H.pop(self.target)
        self.X_test = self.H

        self.mean = np.mean(self.y_train) 
        self.median = np.median(self.y_train) 

        self.params = params

    def res(self):
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        nb_dtp = sm.GLM(y_train, sm.add_constant(X_train.astype(float)), family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()
        print(nb_dtp.summary())
        gb_dtp = GradientBoostingRegressor(**self.params).fit(X_train, y_train)

        mae_nb = mean_absolute_error(y_test,nb_dtp.predict(sm.add_constant(X_test.astype(float))))
        rmse_nb = np.sqrt(mean_squared_error(y_test, nb_dtp.predict(sm.add_constant(X_test.astype(float)))))

        mae_gb = mean_absolute_error(y_test,gb_dtp.predict(X_test))
        rmse_gb = np.sqrt(mean_squared_error(y_test, gb_dtp.predict(X_test)))

        mae_mean = mean_absolute_error(y_test, [self.mean]*len(y_test))
        rmse_mean = np.sqrt(mean_squared_error(y_test, [self.mean]*len(y_test)))          
        
        mae_median = mean_absolute_error(y_test, [self.median]*len(y_test))
        rmse_median = np.sqrt(mean_squared_error(y_test, [self.median]*len(y_test)))

        print("NB: The mean absolute error (MAE) on test set: {:.4f}".format(mae_nb))
        print("NB: The root mean squared error (RMSE) on test set: {:.4f}".format(rmse_nb))
        print("GB: The mean absolute error (MAE) on test set: {:.4f}".format(mae_gb))
        print("GB: The root mean squared error (RMSE) on test set: {:.4f}".format(rmse_gb))
        print("MEAN: The mean absolute error (MAE) on test set: {:.4f}".format(mae_mean))
        print("MEAN: The root mean squared error (RMSE) on test set: {:.4f}".format(rmse_mean))
        print("MEDIAN: The mean absolute error (MAE) on test set: {:.4f}".format(mae_median))
        print("MEDIAN: The root mean squared error (RMSE) on test set: {:.4f}".format(rmse_median))

        yhat_nb = nb_dtp.predict(sm.add_constant(X_test.astype(float)))
        yhat_gb = gb_dtp.predict(X_test)
        return yhat_nb, yhat_gb
    
    def gb(self):
#         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        X_train = np.array(self.X_train)
        gb_dtp = GradientBoostingRegressor(**self.params).fit(X_train, self.y_train)   
        return gb_dtp
    
    def feat_import(self, ax):
        gb_dtp = self.gb()
        sorted_idx = np.argsort(gb_dtp.feature_importances_)
        y_ticks = np.arange(0, len(self.X_train.columns))
        ax.barh(y_ticks, gb_dtp.feature_importances_[sorted_idx], color='dodgerblue')
        ax.set_yticklabels(self.X_train.columns[sorted_idx])
        ax.set_yticks(y_ticks)
        ax.set_title('Gradient Boost Feature Importance')

    def perm_import(self, ax):  
#         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)
        X_train = np.array(self.X_train)
        gb_dtp = GradientBoostingRegressor(**self.params).fit(X_train, self.y_train)
        result = permutation_importance(gb_dtp, self.X_test, self.y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
        perm_sorted_idx = result.importances_mean.argsort()
        ax.boxplot(result.importances[perm_sorted_idx].T,
                vert=False, labels=self.X_test.columns[perm_sorted_idx])
        ax.set_title('Gradient Boost Bootstrapped Permutation Test')
        ax.axvline(x=0, linestyle=':', color='r')
        
    def part_dep(self):
        X_train = np.array(self.X_train)
        gb_dtp = GradientBoostingRegressor(**self.params).fit(X_train, self.y_train)
        plot_partial_dependence(gb_dtp, X_train, [0], line_kw={"color": "dodgerblue"})
        plt.gca()
        plt.gcf()

if __name__=="__main__":

    T = pd.read_csv('../data/train.csv')
    H = pd.read_csv('../data/holdout.csv')

    T['broken_stick'] = [br_stick(x, 5) for x in T.pcd_month]
    H['broken_stick'] = [br_stick(x, 5) for x in H.pcd_month]

    T['seasonality'] = T.broken_stick
    T['ratio'] = T.ratio*100
    dtp_df = T[['dtp', 'ratio', 'seasonality']]

    H['seasonality'] = H.broken_stick
    H['ratio'] = H.ratio*100
    dtpH_df = H[['dtp', 'ratio', 'seasonality']]
    
#     nb_gb = NB_GB(dtp_df, 'dtp', alpha=9)
    # print(nb_gb.cv_nb())
    # print(nb_gb.cv_gb())
    # nb_gb.res()
     
    # fig, ax = plt.subplots()
    # nb_gb.choose_alpha(ax)
    # plt.show()

    # fig, ax = plt.subplots()
    # nb_gb.feat_import(ax)
    # plt.show()

    # fig, ax = plt.subplots()
    # nb_gb.perm_import(ax)
    # plt.show()

    ho = HoldOut(dtp_df, dtpH_df, 'dtp')
    ho.res()
    ho.part_dep()
    plt.show()
#     param_grid = {
#         'max_depth': [3,4,5],
#         'min_samples_split': [5,15],
#         'n_estimators': [100,500]
#         }
#     nb_gb.grid_search_best(param_grid)
     