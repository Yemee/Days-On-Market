import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import statsmodels.api as sm 
import pickle

def metric_func(y_pred, y_test, model):
    metrics = [accuracy_score, precision_score, recall_score,
               f1_score, brier_score_loss, roc_auc_score, confusion_matrix]
    d1 = {}
    for metric in metrics:
        m = metric(y_pred, y_test)
        d1[metric]=m
    out_df = pd.DataFrame.from_dict(d1, orient='index')
    out_df['metric'] = ['acc', 'prec', 'rec', 'f1', 'brier', 'roc_auc', 'confusion_matrix']
    out_df.set_index('metric', inplace=True)
    out_df[model] = out_df.values
    out_df.pop(0)
    return out_df

def results_by_thresh(y_proba, y_test, thresh_list):
    y_pred = y_proba>=thresh_list[0]
    df = metric_func(y_pred, y_test, thresh_list[0])
    for thresh in thresh_list[1:]:
        y_pred = y_proba>=thresh
        df[thresh] = metric_func(y_pred, y_test, thresh)
    return df

class CompareClassifiers():
    features = ['quick', 'ratio', 'is_warm', 'is_realtor', 'is_corp_owner']
#                 
# #     features = ['quick', 'ratio', 'fin_sqft', 'inventory',
#                 'n_photos', 'walkscore',
#                 'has_fireplace', 'has_bsmt', 'one_story',
#                 'is_realtor', 'with_cash', 'is_corp_owner', 'is_warm']

    # num_dat = ['fin_sqft', 'inventory', 
    #            'n_baths', 'n_beds', 'n_fireplaces', 'n_photos',
    #            'orig_price','close_price', 'lot_sqft', 'psf']

    # cat_dat = ['financing', 'ownership', 'levels',
    #            'garage_size' ,'garage_0', 'garage_1', 'garage_2',
    #            'period','period_turn', 'period_midmod', 'period_modern', 
    #            'photo', 'photo_small', 'photo_medium', 'photo_large']
    
    names = ['Logistic Regression', "Nearest Neighbors", 
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", 'Gradient Boost',
             "Naive Bayes", "QDA"]

    metrics = [accuracy_score, precision_score, recall_score,
               f1_score, brier_score_loss, roc_auc_score, confusion_matrix]
   

    classifiers = [
        LogisticRegression(),
#         SVC(kernel='linear', C=0.025),
#         SVC(gamma=2, C=1),
        KNeighborsClassifier(21),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    def __init__(self, df):
        self.df = df[self.features]

    def dict_to_sorted_dict(self, d):
        key_list = []
        val_list = []
        for k, v in d.items():
            key_list.append(k)
            val_list.append(v)
        zipped = list(zip(key_list, val_list))
        sorted_zip = sorted(zipped, key=lambda x: x[1])[::-1]
        sorted_d = OrderedDict()
        for tup in sorted_zip:
            sorted_d[tup[0]]=tup[1]
        return sorted_d

    def classify_func(self, target, test_set='val', holdout=None):
        df = self.df
        y = df.pop(target)
        X = df
        if test_set=='holdout':
            holdout = holdout[self.features]
            y_holdout = holdout.pop(target)
            X_holdout = holdout
            X_train, X_test, y_train, y_test= X, X_holdout, y, y_holdout
        elif test_set=='val':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, 
                                                                random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        d1 = {}
        for name, clf in zip(self.names, self.classifiers):
#             sc = StandardScaler()
#             X_train = sc.fit_transform(X_train)
#             X_test = sc.transform(X_test)
            clf.fit(X_train, y_train)
#             y_hat = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:,1]
            y_hat = y_prob >= .5
            d1[name]=[]
            for metric in self.metrics:
                m = metric(y_test, y_hat)
                d1[name].append(m)
                if metric == confusion_matrix:
                    spec = m[0][0]/(m[0][0] + m[0][1])
                    d1[name].append(spec)
            out_df = pd.DataFrame.from_dict(d1, orient='columns')
            out_df['metric'] = ['acc', 'prec', 'rec', 'f1', 'brier', 'roc_auc', 'confusion_matrix', 'specificity']
            out_df.set_index('metric', inplace=True)
        return out_df
        
if __name__=="__main__":

    train = pd.read_csv('../data/train.csv')
    train['walkscore'] = train.walkscore/100
    holdout = pd.read_csv('../data/holdout.csv')
    print('TRAINING SCORES:')
    print(CompareClassifiers(train).classify_func('quick', test_set='train'))
    print('VALIDATION SCORES:')
    print(CompareClassifiers(train).classify_func('quick', test_set='val'))
    print('HOLDOUT SCORES:')
    print(CompareClassifiers(train).classify_func('quick', test_set='holdout', holdout=holdout))

    


