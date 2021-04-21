import math
import numpy as np
import pandas as pd 
from collections import OrderedDict
import scipy.stats as stats
import matplotlib.pyplot as plt

class EDA():
    num_dat = ['dom', 'dtp', 'dtc', 'ratio', 'fin_sqft', 'inventory', 
               'n_baths', 'n_beds', 'n_fireplaces', 'n_photos',
               'orig_price','close_price', 'lot_sqft', 'walkscore', 'psf', 
               'em', 'ba_comp', 'garage_spaces', 'year_built']

    bool_dat = ['quick', 'back_on_market', 'is_vacant',
               'has_fireplace', 'has_bsmt', 'one_story',
               'is_realtor', 'offer_var_comp', 'has_virtual_tour', 'has_showing_instr',
               'with_cash', 'is_corp_owner', 'is_warm']

    cat_dat = ['financing', 'ownership', 'levels',
               'garage_size' ,'garage_0', 'garage_1', 'garage_2',
               'period','period_turn', 'period_midmod', 'period_modern', 
               'photo', 'photo_small', 'photo_medium', 'photo_large',
               'zip']
                
    nlp_dat = ['private', 'public', 'showing_instructions', 'all', 'low_all']
    geo_dat = ['lat', 'long', 'zip']
    ts_dat = ['pcd_mo_year', 'pcd_day_mo_year', 'cd_day_mo_year', 'cd_mo_year']

    def __init__(self, df):
        self.df = df

    def class_num_bars(self, ax, col):
        '''means bar chart with pval for classes'''
        complete_class = self.df
        d = {}
        d['quick'] = complete_class[complete_class.quick==True][col].mean()
        d['slow'] = complete_class[complete_class.quick==False][col].mean()
        stdT = complete_class[complete_class.quick==True][col].std()
        stdF = complete_class[complete_class.quick==False][col].std()
        tstat, pval = stats.ttest_ind(complete_class[complete_class.quick==True][col],
                                    complete_class[complete_class.quick==False][col], 
                                    equal_var = False)
        ax.bar(d.keys(), d.values(), yerr=[stdT, stdF])
        ax.set_title(f'{col}, p-val: {round(pval,2)}', fontsize=25)
        ax.text(0,0,f'    mean: {round(complete_class[complete_class.quick==True][col].mean() ,2)}', 
                    rotation=90, fontsize=20)
        ax.text(1,0,f'    mean: {round(complete_class[complete_class.quick==False][col].mean() ,2)}', 
                    rotation=90, fontsize=20)
    
    def class_cat_bars(self, ax, col, cls=True):
        df = self.df
        x = df[df.quick==cls][col].value_counts()
        ax.bar(x.index, x.values/sum(x))
        ax.set_xticklabels(x.index, rotation=90)
        if cls==True:
            ax.set_title(f'quick: {col}',fontsize=25)
        else:
            ax.set_title(f'slow: {col}',fontsize=25)
        for i in range(len(x)):
            ax.text(i,0,f'     {round((x.values/sum(x))[i]*100,2)}%', rotation=90)


    def plot_nans_func_vert(self, ax, asc=True):
        df = self.df
        d = {}
        for col in df.columns:
            d[col] = df[col].isna().sum()
        nans = pd.Series(d.values(), index=d.keys()).sort_values(ascending=asc)
        y_pos = np.arange(len(nans))
        ax.barh(y_pos, nans.values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(nans.index, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('NaNs', fontsize=20)
        ax.set_title('NaNs per Feature', fontsize=25)
        xmin = nans.values.min()-0.2*df.shape[0]
        ax.set_xlim(xmin, df.shape[0])
        ax.text(xmin,-1, f'    {df.shape[0]} total datapoints', fontsize=20)
        for i in range(df.shape[1]):
            ax.text(xmin, y_pos[i], f'{nans.values[i]}', fontsize=11)

    def percentile(self, col, interval=.95, sided='two', prnt=False):
        if sided=='lower':
            l = self.df[col].min()
            h = self.df[col].sort_values().iloc[math.ceil(self.df.shape[0]*(interval))]
        elif sided=='upper':
            l = self.df[col].sort_values().iloc[math.ceil(self.df.shape[0]*(1-interval))]
            h = self.df[col].max()
        else:
            l = self.df[col].sort_values().iloc[math.ceil(self.df.shape[0]*(1-interval)/2)]
            h = self.df[col].sort_values().iloc[math.ceil(self.df.shape[0]*(interval+(1-interval)/2))]
        if prnt==True:
            print(f'{int(interval*100)}% btwn ({l}, {h})')
        return l, h, interval

    def legend_function(self, ax, col, vert=True):
        df = self.df
        l, h, interval = self.percentile(col)
        Q1, Q3, iqr = self.percentile(col, interval=.50)
        if vert==True:
            func=ax.axvline
        else:
            func=ax.axhline
        func(self.df[col].mean(), 
             label=f'mean: {round(self.df[col].mean(),2)}, std: {round(self.df[col].std(),2)}')
        func(l, linestyle='--', 
                   label=f'{int(interval*100)}% btwn ({round(l,2)}, {round(h,2)})')
        func(h, linestyle='--')
        func(self.df[col].median(), linestyle=':',
                    label=f'IQR: {round(Q1,2)}, {round(self.df[col].median(),2)}, {round(Q3,2)}')
        func(Q1, linestyle=':')
        func(Q3, linestyle=':')
        func(linestyle='', label=f'min: {round(self.df[col].min(),2)}, max: {round(self.df[col].max(),2)}')
        func(linestyle='', label=f'sample size: {self.df.shape[0]}')

    def text_plot(self, ax, col, text='', fontsize=25, loc=(0,0)):
        '''use a plot space for text'''
        df = self.df
        ax.text(loc[0], loc[1], text, fontsize=fontsize)
        ax.tick_params(
        grid_color='white',
        axis='both',          # changes apply to the x-axis
        which='both',         # both major and minor ticks are affected
        bottom=False,         # ticks along the bottom edge are off
        top=False,            # ticks along the top edge are off
        right=False,
        left=False,
        labelleft=False,
        labelbottom=False)
        self.legend_function(ax, col)
        ax.legend(loc=loc, facecolor='lightgrey')

    def histogram(self,  ax, col, bins=150, color='slategrey', alpha=1, density=True, leg=False):
        df = self.df
        ax.hist(df[col], bins=bins, color=color, alpha=alpha, density=density)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel('density/mass/counts')
        ax.set_xlim(df[col].min(), df[col].max())
        self.legend_function(ax, col)
        if leg==True:
            ax.legend(loc=loc, facecolor='lightgrey')

    def jittered(self, ax, col, color='slategrey', alpha=.2, leg=False):
        df = self.df
        n=df.shape[0]
        y=df[col]
        ybar=df[col].mean()
        ax.scatter(y, np.repeat(0,n)+np.random.normal(0,0.1,n), color=color, alpha=alpha)
        ax.set_xlabel(f'{col}')
        ax.set_xlim(df[col].min(), df[col].max())
        ax.set_title(f'{col} jittered data')
        ax.set_yticks([],[])
        self.legend_function(ax, col)
        if leg==True:
            ax.legend(loc=loc, facecolor='lightgrey')

    def sorted_plot(self, ax, col, color='slategrey', alpha=.2, leg=False):
        df = self.df
        ax.scatter(np.arange(df.shape[0]), df[col].sort_values(), color=color, alpha=alpha)
        self.legend_function(ax, col, vert=False)
        ax.set_ylabel(f'{col}')
        ax.set_xlabel('index')
        ax.set_title(f'{col} sorted data')
        ax.set_xlim(df[col].min(), df[col].max())
        if leg==True:
            ax.legend(loc=loc, facecolor='lightgrey')

    def target_on_col(self, ax, col, target, color='slategrey', alpha=.5, loc='best'):
        df = self.df
        rho = round(df[col].corr(df[target]),2)
        ax.scatter(df[col], df[target], color=color, alpha=alpha, label=f'corr: {rho}')
        ax.set_title(f'{target} on {col}')
        ax.set_ylabel(target)
        ax.set_xlabel(col)
        ax.legend(loc=loc, facecolor='lightgrey')

    def correlation_function(self):
        df = self.df
        return df.corr()


    # categoric functions
    def perc_by_level(self, col):
        df = self.df
        level_list = df[col].unique()
        n = df.shape[0]
        d = {}
        for level in level_list:
            d[level] = round(len(df.loc[:,col][df.loc[:,col]==level])/n*100,2)
        return self.dict_to_sorted_dict(d)

    def dict_to_sorted_dict(self, d, which_col=1, asc=-1):
        key_list = []
        val_list = []
        for k, v in d.items():
            key_list.append(k)
            val_list.append(v)
        zipped = list(zip(key_list, val_list))
        sorted_zip = sorted(zipped, key=lambda x: x[which_col])[::asc]
        sorted_d = OrderedDict()
        for tup in sorted_zip:
            sorted_d[tup[0]]=tup[1]
        return sorted_d

    def slice_by_level(self, col):
        '''separated df in subdataframes by levels'''
        df = self.df
        d = {}
        for level in df[col].unique():
            d[level] = df[df[col]==level]
        return d

    def target_by_level(self, col, target, thresh = 0.0):
        '''sorts dataframe by level and return a dictionary of the 
        target values for that level if that level is greater than thresh % of all data'''
        df = self.df
        d = {}
        temp = self.slice_by_level(col)
        for level in df[col].unique():
            if self.perc_by_level(col)[level] > thresh:
                d[level] = temp[level][target]
        return self.dict_to_sorted_dict(d, which_col=0, asc=1)

    def box_by_level(self, ax, col, target, thresh=0.0, ylim=None, color='slategrey'):
        '''plot boxplots of feature levels on target with percentages'''
        df = self.df
        d = self.target_by_level(col, target)
        ax.set_xticklabels(d.keys(), rotation=90, color='k', fontsize=20)
        if ylim != None:
            ax.set_ylim(ylim)
        bplot= ax.boxplot(d.values(),
                            notch=True,  # notch shape
                            vert=True,  # vertical box alignment
                            patch_artist=True)  # fill with color 
        cols = [color]*len(d)
        for b in bplot:
            for patch, color in zip(bplot['boxes'], cols):
                patch.set_facecolor(color)
        zip_perc = self.dict_to_sorted_dict(self.perc_by_level(col), which_col=0, asc=1)
        i = 0
        samples = []
        for k, v in zip_perc.items():
            if v > 0.0:
                i+=1
                ax.text(i+.1, 65, f'mean: {round(df[df[col]==k][target].mean(),2)}', rotation=90)
                ax.text(i-.2, -5, f'median: {round(df[df[col]==k][target].median(),2)}', rotation=0)
                ax.text(i-.25, -5, f'{v}% of {target}' , rotation=0)
                samples.append(df[df[col]==k][target])
#         ax.set_title(f'{col}, pval = {round(stats.ttest_ind(samples[0], samples[1], equal_var=False)[1],2)}')

    def bar_plot(self, ax, col, target, width=.5, color='slategrey', rotation=0):
        df = self.df
        x = df[col].unique()
        means = []
        stds = []
        percs = []
        for lev in x:
            means.append(df[df[col]==lev][target].mean())
            stds.append(df[df[col]==lev][target].std())
            percs.append(df[df[col]==lev].shape[0]/df.shape[0])
        labs = df[col].unique().astype(str)
        ax.bar(labs, means, yerr=stds, align='center')
        ax.set_ylim(0,80)
        plt.xticks(labs, rotation=45)
        for i in range(len(x)):
            ax.text(labs[i], 72, f'{round(percs[i]*100,0)}% of data', rotation=rotation)
            ax.text(labs[i], 66, f'mean: {round(means[i],2)}', rotation=rotation)
            ax.text(labs[i], 60, f'std: {round(stds[i],2)}', rotation=rotation)  
        ax.set_title(col+' by '+target)

    def log_hist(self, ax, col):
        df = self.df
        y = df[col]+.99
        ax.hist(np.log(y), bins=50)
    
    def describe_func(self):
        df = self.df
        return df[self.num_dat].describe()

if __name__=="__main__":

    train = pd.read_csv('../data/train.csv')
    train.set_index('Listing Id', inplace=True)

    train = pd.read_csv('../data/train.csv')
    holdout = pd.read_csv('../data/holdout.csv')
    all_df = pd.concat([train, holdout])

    covid_train = pd.read_csv('../data/covid_train.csv')
    covid_holdout = pd.read_csv('../data/covid_holdout.csv')
    all_covid = pd.concat([covid_train, covid_holdout])

    complete = pd.concate(all_df, all_covid)
    complete.set_index('Listing Id', inplace=True)

#     quick vs slow class numeric EDA
    for col in EDA(complete).num_dat:
        fig, ax = plt.subplots()
        EDA(complete).class_num_bars(ax, col)
        temp_str = f'../images/class_{col}.png'
        fig.savefig(temp_str)
        fig.tight_layout()
        plt.show()

    # quick vs slow class categoric EDA
    for col in EDA(complete).cat_dat:
        fig, ax = plt.subplots(1,2, figsize = (15, 5))
        EDA(complete).class_cat_bars(ax[0],col)
        EDA(complete).class_cat_bars(ax[1],col, cls=False)
        temp_str = f'../images/class_cat_{col}.png'
        fig.savefig(temp_str)
        fig.tight_layout()
        plt.show()

    # explore numeric data
    for col in EDA(complete).num_dat:
        fig, ax = plt.subplots(2,3, figsize=(20,5))
        EDA(complete).text_plot(ax[0][0], col)
        EDA(complete).target_on_col(ax[0][1], col=col, target='dtp')
        EDA(complete).target_on_col(ax[0][2], col=col, target='dtc')
        EDA(complete).histogram(ax[1][0], col)
        EDA(complete).jittered(ax[1][1], col)
        EDA(complete).sorted_plot(ax[1][2], col)
        temp_str = f'../images/{col}.png'
        fig.savefig(temp_str)
        fig.tight_layout()
        # plt.show()

    # explore categoric data
    for col in EDA(complete).bool_dat+EDA(complete).cat_dat:
        fig, ax = plt.subplots(1, 2, figsize=(20,5))
        EDA(complete).box_by_level(ax[0], col, 'dtp')
        EDA(complete).bar_plot(ax[1], col, 'dtp')
        fig.tight_layout()
        temp_str = f'../images/{col}_dtp.png'
        fig.savefig(temp_str)
        # plt.show()

    for col in EDA(complete).bool_dat+EDA(complete).cat_dat:
        fig, ax = plt.subplots(1, 2, figsize=(20,5))
        EDA(complete).box_by_level(ax[0], col, 'dtc')
        EDA(complete).bar_plot(ax[1], col, 'dtc')
        fig.tight_layout()
        temp_str = f'../images/{col}_dtc.png'
        fig.savefig(temp_str)
        # plt.show()

    # look at log transform
    fig, ax = plt.subplots()
    EDA(complete).log_hist(ax, 'dtp')
    fig.tight_layout()
    # plt.show()

    # look at summary stats for numeric data
    print(EDA(complete).describe_func())


    