import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from datetime import datetime

def convert_to_date(df, col, keep_col):
    '''drops nans for entire df'''
    df.dropna(inplace=True)
    d = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
    df['month'] = [dt.split('-')[0] for dt in df[col]]
    df['n_month'] = [d[mo] for mo in df.month]
    df['year'] = [str(int(dt.split('-')[1])+2000) for dt in df[col]]
    df[col+'_mo_year'] = df['n_month']+'-'+df['year'] 
    df['mo_yr'] = pd.to_datetime(df[col+'_mo_year'])#.dt.to_period('M')
    df = df[[keep_col, 'mo_yr']]
    df = df[(df.mo_yr>'2014-12') & (df.mo_yr<'2021-01')]
    df.set_index('mo_yr', inplace=True)
    return df

if __name__=="__main__":

    active = pd.read_csv('../data/active.csv')
    new = pd.read_csv('../data/new.csv')
    pending = pd.read_csv('../data/pending.csv')
    closed = pd.read_csv('../data/closed.csv')
    showings = pd.read_csv('../data/showings.csv')
    train = pd.read_csv('../data/train.csv')

    active = convert_to_date(active, 'date', 'active')
    new = convert_to_date(new, 'date', 'new')
    pending = convert_to_date(pending, 'date', 'pending')
    closed = convert_to_date(closed, 'date', 'closed')
    showings = convert_to_date(showings, 'date', 'showings')
    activity = active.join(new).join(pending).join(closed).join(showings)

    df = train[['dtp', 'dtc', 'pcd_mo_year']]
    df = df.groupby('pcd_mo_year').mean()
    activity = df.join(activity)
    activity.to_csv('../data/activity.csv')

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(activity.index[:-1], activity.dtp[:-1], label = 'average days to purchase contract', color = 'm')
    ax.plot(activity.index[:-1], activity.dtc[:-1], label = 'average days to close', color='yellowgreen')
    ax.axvline(x='2020-03-01', linestyle='--', label = 'Covid-19: March 2020', lw=5)
    ax.set_ylim(10,60)
    ax.set_ylabel('days', fontsize=20)
    ax.set_xlabel('date', fontsize=20)
    ax.set_title('Covid-19 Effect', fontsize=25)
    ax.legend(fontsize=15, loc='upper center', facecolor='whitesmoke');
    fig.savefig('../images/covid_effect.png')

    fig, ax = plt.subplots(figsize=(12,8))
    ax2 = ax.twinx()
    ax.plot(activity.index, activity.showings, color='g')
    ax2.plot(activity.index, activity.active, color='b')
    ax.axvline(x = '2020-03-01', color='r', ls='--', label = 'March 2020')
    ax.scatter('2019-01-01', 6.6, color='r', s=250, label = 'January 2019')
    ax.scatter('2020-08-01', 7.75, color='r', s=250, label = 'August 2020')
    ax.set_xlabel('date', fontsize=20)
    ax.set_ylabel('average showings per listing', fontsize=25, color='g')
    ax2.set_ylabel('number of active listings', fontsize=25, color='b')
    ax.set_title('Supply & Demand Comparison', fontsize=25)
    ax.legend(fontsize=20, loc='upper center');
    fig.savefig('../images/supply_and_demand.png')

    plt.show()