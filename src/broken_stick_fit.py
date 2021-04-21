import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import statsmodels.api as sm
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
colors = ['black', 'slategrey', 'm','tab:orange', 'yellowgreen', 'violet', 'thistle', 'plum']

import warnings
warnings.filterwarnings('ignore')

# eyeball broken stick not used
def br_stick(x, br,a=1, b=1, c=1, d=1, e=.15):
    return a*(br-x) * b*(x<br) + c*(br-x)**2 * d*(x<br) + x + e*x**2+6

# OLS broken stick
def rsq_for_b(b, df):
    '''get an rsqd value for this specific brokenstick regression'''
    df['lhs_x'] = [b-x if x < b else 0 for x in df.pcd_month]
    df['rhs_x'] = [0 if x < b else x-b for x in df.pcd_month]
    df['lhs_xsq'] = [(b-x)**2 if x < b else 0 for x in df.pcd_month]
    df['rhs_xsq'] = [0 if x < b else (x-b)**2 for x in df.pcd_month]
    X = df[['dtp', 'lhs_x', 'rhs_x', 'lhs_xsq', 'rhs_xsq']]
    y = X.pop('dtp')
    lmod = sm.OLS(y,sm.add_constant(X.astype(float))).fit()
    return lmod.rsquared

def rsq_best_b(df, start=1, stop=12, n=1000):
    rsqs = []
    bs = []
    for b in np.linspace(start, stop, n):
        rsqs.append(rsq_for_b(b,df))
        bs.append(b)
    rsqs = np.array(rsqs)
    bs = np.array(bs)
    rsq, b = rsqs.max(), bs[rsqs.argmax()]
    return rsq, b

def broken_stick_feature(b, df):
    '''get an rsqd value for this specific brokenstick regression'''
    df['lhs_x'] = [b-x if x < b else 0 for x in df.pcd_month]
    df['rhs_x'] = [0 if x < b else x-b for x in df.pcd_month]
    df['lhs_xsq'] = [(b-x)**2 if x < b else 0 for x in df.pcd_month]
    df['rhs_xsq'] = [0 if x < b else (x-b)**2 for x in df.pcd_month]
    X = df[['dtp', 'lhs_x', 'rhs_x', 'lhs_xsq', 'rhs_xsq']]
    y = X.pop('dtp')
    lmod = sm.OLS(y,sm.add_constant(X.astype(float))).fit()
    c, lhs_x, rhs_x, lhs_xsq, rhs_xsq = lmod.params
    x = np.arange(1,64)
    y = c + lhs_x*df.lhs_x + rhs_x*df.rhs_x + lhs_xsq*df.lhs_xsq + rhs_xsq*df.rhs_xsq
    print(lmod.params, lmod.rsquared)
    return y



if __name__=="__main__":

    all_the_data_train = pd.read_csv('../data/all_the_data_train.csv')
    all_the_data_holdout = pd.read_csv('../data/all_the_data_holdout.csv')
    all_df = pd.concat([all_the_data_train, all_the_data_holdout])

    df = all_df[['dtp', 'dtc', 'pcd_mo_year', 'pcd_month']]
    df = df[df.pcd_mo_year<'2020-03-01']
    df = df.groupby('pcd_mo_year').mean()

    rsq, b = rsq_best_b(df)
    y = broken_stick_feature(b, df)
    seasonality_by_month = y.unique()

    fig, ax = plt.subplots(figsize=(20,8))
    ax.plot(df.index, df.dtp, label='average DTP/mo', color='olivedrab', marker='o')
    ax.plot(np.arange(1,64), y, label = 'seasonality feature', color='dodgerblue', linestyle=':', lw=3)
    ax.axvline(x=b, label=f'knot at {b:2.2f}, Rsq:{rsq:2.2f}', 
               color='k', linestyle='--', linewidth=3)
    ax.set_xticks(np.arange(1,64,12))
    ax.set_xticklabels([ '2015',  '2016',  '2017',  '2018',  '2019',  '2020'])
    ax.set_title('DTP Seasonality')
    ax.set_xlabel('date', fontsize=20)
    ax.set_ylabel('days', fontsize=20)
    ax.set_ylim(10,52)
    ax.legend(fontsize=20, loc="upper center");
     

    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(np.arange(1,13), seasonality_by_month, 
            color='dodgerblue', label='seasonality values', 
            marker='o',linewidth=3)
    ax.scatter(np.arange(1,13), seasonality_by_month, 
            color='dodgerblue', s=300)
    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels([ 'Jan',  'Feb',  'Mar',  'Apr',  'May',  'Jun', 'Jul',  'Aug',  'Sep',  'Oct',  'Nov',  'Dec'])
    ax.set_ylabel('days')
    ax.set_title('DTP Seasonality')
    ax.legend(loc='upper center');
     