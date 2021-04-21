import numpy as np
import pandas as pd
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'lightgrey'

class HypTest():
    def __init__(self, col_list, name_list, alpha=.05, alternative='two-sided'):

        self.col_list = col_list
        self.name_list = name_list
        self.alpha = alpha
        self.alpha_2 = self.alpha/2
        self.alternative = alternative
        self.Bonferroni = self.alpha/56

        self.A = col_list[0]
        self.B = col_list[1]
        self.nA = len(col_list[0])
        self.nB = len(col_list[1])
        self.xbarA = np.mean(col_list[0])
        self.xbarB = np.mean(col_list[1])
        self.medA = np.median(col_list[0])
        self.medB = np.median(col_list[1])
        self.sdA = np.std(col_list[0])
        self.sdB = np.std(col_list[1])
        
        # standard error = standard deviation of means sampling distribution (CLT)
        self.seA = self.sdA/np.sqrt(self.nA)
        self.seB = self.sdB/np.sqrt(self.nB)
        # pooled standard deviation for difference of means
        self.pooled_sd = np.sqrt(self.sdA**2/self.nA + self.sdB**2/self.nB)
        
        # test statistic
        self.diff = self.xbarA-self.xbarB  #raw difference
        self.test_stat = self.diff/self.pooled_sd
        self.z = self.test_stat #want to be able to call it z
        
        # standard deviations for effect measures
        self.cohens_sd = np.sqrt((self.sdA**2 + self.sdB**2)/2)
        self.glass_sd = self.pooled_sd
        self.hedges_sd = np.sqrt(((self.nA-1)*self.sdA**2 + (self.nB-1)*self.sdB**2)/(self.nB+self.nA-2))
        # effect metrics
        self.effect_cohens_d = self.diff/self.cohens_sd
        self.effect_glass_delta = self.diff/self.pooled_sd
        self.effect_hedges_g = self.diff/self.hedges_sd
        
        # critical value called rr and cv
        self.rr_l = stats.norm(0, self.pooled_sd).ppf(self.alpha_2)
        self.rr_h = stats.norm(0, self.pooled_sd).ppf((1-self.alpha)+self.alpha_2)
        self.rr_lt = stats.norm(0, self.pooled_sd).ppf(self.alpha)
        self.rr_gt = stats.norm(0, self.pooled_sd).ppf(1-self.alpha)
        # ^^these are equivalent, and redundant, but helpful :/
        self.cv_l = stats.norm().ppf(self.alpha_2)*self.pooled_sd
        self.cv_r = stats.norm().ppf(1-self.alpha_2)*self.pooled_sd
        self.cv_lt = stats.norm().ppf(self.alpha)*self.pooled_sd
        self.cv_gt = stats.norm().ppf(1-self.alpha)*self.pooled_sd

        if self.alternative=='two_sided':
            self.power = stats.norm(self.diff, self.pooled_sd).cdf(self.rr_l)
        else:
            self.power = 1 - stats.norm(self.diff, self.pooled_sd).cdf(abs(self.rr_gt))
 
        # colors for visualizations
        self.colA = 'yellowgreen'
        self.colB = 'lightblue'
        self.col_null = 'yellowgreen'
        self.col_alt = 'lightblue'
        self.col_rr = 'm'
        self.col_power = 'slategrey'

        self.pval_1sided = (1 - stats.norm(0, self.pooled_sd).cdf(abs(self.diff)))
        self.pval_2sided = (1 - stats.norm(0, self.pooled_sd).cdf(abs(self.diff)))*2

    def p_value(self, equal_var=False):
        return stats.ttest_ind(self.col_list[0], 
                               self.col_list[1], 
                               equal_var=equal_var, 
                               alternative=self.alternative)
    
    def p_value_by_hand(self):
        pv = 1 - stats.norm(0, self.pooled_sd).cdf(abs(self.diff))
        if self.alternative=='two-sided':
            return pv*2
        else:
            return pv

    def descriptive_stats(self, num_list, perc = .95):
        '''input: list of numeric data
        output: dictionary of descriptive statistics'''
        d = {}
        
        d['size'] = len(num_list)
        
        d['mean'] = np.mean(num_list)
        d['sd'] = np.std(num_list)
        
        d['min'] = np.min(num_list)
        d['1Q'] = np.percentile(num_list, 25)
        d['median'] = np.median(num_list)
        d['3Q'] = np.percentile(num_list, 75)
        d['max'] = np.max(num_list)
        
        d['min_mode'] = stats.mode(num_list)[0][0]
        
        d[f'{int(perc*100)}% quantile low'] = np.percentile(num_list, 100*(1-perc)/2) 
        d[f'{int(perc*100)}% quantile high'] = np.percentile(num_list, 100*(1-((1-perc)/2)))
        
        d['IQR'] = np.percentile(num_list, 75) - np.percentile(num_list, 25)
        d['1Q - 1.5*IQR'] = np.percentile(num_list, 25) - 1.5*(np.percentile(num_list, 75) - np.percentile(num_list, 25))
        d['3Q + 1.5*IQR'] = np.percentile(num_list, 75) + 1.5*(np.percentile(num_list, 75) - np.percentile(num_list, 25))
        num_arr = np.array(num_list)
        d['n_outliers_low?'] = len(num_arr[num_arr<d['1Q - 1.5*IQR']])
        d['n_outliers_high?'] = len(num_arr[num_arr>d['3Q + 1.5*IQR']])
        d['%_outiers_low'] = len(num_arr[num_arr<d['1Q - 1.5*IQR']])/len(num_list)
        d['%_outiers_high'] = len(num_arr[num_arr>d['3Q + 1.5*IQR']])/len(num_list)
        return d

    def descriptive_stats_df(self, perc=.95):
        '''input: list of numeric lists, names of numeric lists
        output: dataframe of descriptive statistics, stats for each list = column'''
        arrs = self.col_list
        cols = self.name_list
        d = self.descriptive_stats(arrs[0], perc=perc)
        df = pd.DataFrame(index = d.keys(), data = d.values(), columns=[cols[0]])
        for i, arr in enumerate(arrs[1:]):
            temp = self.descriptive_stats(arr)
            df_temp = pd.DataFrame(index = temp.keys(), data = temp.values(), columns=[cols[i+1]])
            df=df.join(df_temp)
        return df
    
    def bonferroni_and_the_tests(self):
        '''input: list of samples, list_of_sample_names, alpha(default is 0.05)
           output: dataframe of pvalues, bonferroni corrected alpha'''
        n = len(self.col_list)
        arr = np.zeros((n,n))
        for i, sample1 in enumerate(self.col_list):
            for j, sample2 in enumerate(self.col_list):
                arr[i,j] = stats.ttest_ind(sample1,sample2, equal_var=False, alternative=self.alternative)[1]
        df = pd.DataFrame(arr, columns=self.name_list, index=self.name_list)
        return df
    

    def inferential_stats(self, num_list, ci=.95):
        '''input: list of numeric data
        output: dictionary of inferential statistics'''
        d = {}
        n = len(num_list)
        d['size'] = n

        xbar = np.mean(num_list)
        d['mean'] = xbar

        se = np.std(num_list)/np.sqrt(len(num_list))
        d['se'] = se
        
        d[f'{int(ci*100)}% CI low'] = stats.norm(xbar, se).ppf(1-ci) 
        d[f'{int(ci*100)}% CI high'] = stats.norm(xbar,se).ppf(ci) 
        return d

    def inferential_stats_df(self, ci=.95):
        '''input: list of numeric lists, names of numeric lists
        output: dataframe of inferential statistics, stats for each list = column'''
        arrs = self.col_list
        cols = self.name_list
        d = self.inferential_stats(arrs[0], ci=ci)
        df = pd.DataFrame(index = d.keys(), data = d.values(), columns=[cols[0]])
        for i, arr in enumerate(arrs[1:]):
            temp = self.inferential_stats(arr)
            df_temp = pd.DataFrame(index = temp.keys(), data = temp.values(), columns=[cols[i+1]])
            df=df.join(df_temp)
        return df

    def plot_mean_sd_all(self, ax):
        ax.bar(np.arange(len(self.col_list)), 
               [np.mean(x) for x in self.col_list], 
               yerr=[np.std(x) for x in self.col_list], 
               color=[self.colA, self.colB]+[self.colA]*(len(self.col_list)-2))
        ax.set_xticks(np.arange(len(self.col_list)))
        ax.set_xticklabels(self.name_list, fontsize=20)
        ax.set_ylabel('days to purchase contract', fontsize=20)
        ax.set_title('Mean and Standard Deviation', fontsize=25)
        ax.axhline(y=np.mean(self.xbarB), linestyle=':', linewidth=3, color='m', label='covid mean')
        for i, s in enumerate(self.col_list):
            ax.text(i-.375,np.mean(s)/2, f'mean:  {np.mean(s):2.2f}', fontsize=15)
            ax.text(i-.15,np.mean(s)+1, f' sd: {np.std(s):2.2f}', fontsize=15, rotation=90)
            ax.scatter(i,np.mean(s), color = 'k')
        ax.legend(loc='upper left', facecolor='lightgrey')
        
    def plot_mean_sd(self, ax):
        ax.bar([0,1], 
               [self.xbarA, self.xbarB], 
               yerr=[self.sdA, self.sdB], 
               color=[self.colA, self.colB])
        ax.set_xticks([0,1])
        ax.set_xticklabels([self.name_list[0], self.name_list[1]], fontsize=20)
        ax.set_ylabel('days to purchase contract', fontsize=20)
        ax.set_title('Mean and Standard Deviation Pre-Covid & During Covid', fontsize=25)
        ax.text(0+.05,self.xbarA/2, f'mean = {self.xbarA:2.2f}', fontsize=15)
        ax.text(1+.05,self.xbarB/2 , f'mean = {self.xbarB:2.2f}', fontsize=15)
        ax.text(0-.05,self.xbarA+1, f' sd = {self.sdA:2.2f}', fontsize=15, rotation=90)
        ax.text(1-.05,self.xbarB+1, f' sd = {self.sdB:2.2f}', fontsize=15, rotation=90)  
        ax.scatter(0,self.xbarA, color = 'k')
        ax.scatter(1,self.xbarB, color = 'k')
     
    
    def box_plot_all(self, ax):
        bplot = ax.boxplot(self.col_list[::-1], 
                           notch=True, 
                           vert=False, 
                           patch_artist=True)
        cols = [self.colA]*(len(self.col_list)-2)+[self.colB, self.colA]
        for b in bplot:
            for patch, color in zip(bplot['boxes'], cols):
                patch.set_facecolor(color)
        ax.set_title('Median on Notched Box Plot')
        ax.set_xlabel('days to purchase contract', fontsize=20)
        ax.set_yticklabels(self.name_list[::-1], fontsize=20)
        ax.axvline(x=self.medB, linestyle=':', linewidth=3, color='m', label='covid median')
        for i, s in enumerate(self.col_list[::-1]):
            ax.text(np.percentile(s,51)+5, i+.94, f'{np.median(s):2.2f}', fontsize=15)
        ax.legend()
    
    def box_plot(self, ax):
        bplot = ax.boxplot([self.B, self.A], 
                           notch=True, 
                           vert=False, 
                           patch_artist=True)
        cols = [self.colB, self.colA]
        for b in bplot:
            for patch, color in zip(bplot['boxes'], cols):
                patch.set_facecolor(color)
        ax.set_title('Median Notched Box Plot')
        ax.set_xlabel('days to purchase contract', fontsize=20)
        ax.set_yticklabels([self.name_list[1], self.name_list[0]], fontsize=20)
        
    def mode_line(self, ax):
        x = np.arange(len(self.col_list))
        y = [stats.mode(s)[0][0] for s in self.col_list]
        ax.plot(x, y, color=self.colA)
        ax.scatter(x,y, color=self.colA, label = 'modes', s=500)
        ax.scatter(x[1],y[1],color='slateblue', s=700)
        ax.scatter(x[1],y[1],color=self.colB, label='covid mode', s=700, marker='*')
        for a, b in zip(x,y):
            ax.text(a,b-1,f'{int(b)}', fontsize=25)
        ax.set_title('Modes')
        ax.set_ylim(0,6)
        ax.set_ylabel('days', fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.name_list, fontsize=20)
        ax.legend(fontsize=20, facecolor='lightgrey')
        
    def sample_size(self, ax):
        ax.bar(np.arange(len(self.col_list[1:-1])), 
               [len(x) for x in self.col_list[1:-1]],  
               color=[self.colB]+[self.colA]*(len(self.col_list)-1))
        ax.set_xticks(np.arange(len(self.col_list[1:-1])))
        ax.set_xticklabels(self.name_list[1:-1], fontsize=20)
        ax.set_ylabel('number of data points', fontsize=20)
        ax.set_title('Sample Size', fontsize=25)
        for i, s in enumerate(self.col_list[1:-1]):
            ax.text(i-.2,500, f'{len(s):2.0f}', fontsize=25)
        ax.axhline(y=len(self.col_list[1]), color='m', linestyle=':', linewidth=3, label='covid sample size')
        ax.legend(loc='upper center', facecolor='lightgrey')
        
    def plot_hists(self, ax):
        binA = int(np.sqrt(len(self.A)))+25
        binB = int(np.sqrt(len(self.B)))+25
        ax.hist(self.A, color = self.colA, alpha=.75,bins=binA,density=True, label=self.name_list[0])
        ax.hist(self.B, color = self.colB, alpha=.75, bins=binB, density=True, label=self.name_list[1])
        ax.set_xlabel('days to purchase contract')
        ax.set_ylabel('frequency')
        ax.set_title('Distribution by Pre-Covid-19 & During Covid-19')
        ax.grid(color='white')
        ax.legend()

    def plot_sorted(self, ax, lw=1):
        l = min(len(self.A), len(self.B))
        x = np.arange(l)
        yA = sorted(random.sample(list(self.A), k=l))
        yB = sorted(random.sample(list(self.B), k=l))
        ax.plot(x, yA, color=self.colA, label = self.name_list[0], linewidth=lw)
        ax.plot(x, yB, color=self.colB, label = self.name_list[1], linewidth=lw)
        ax.set_title('Sorted Values by Pre-Covid & During Covid')
        ax.set_xlabel('index')
        ax.set_ylabel('sorted random sample')
        ax.legend()

    def plot_sampling_distributions(self, ax):
        start = min(self.xbarA-self.seA*3, self.xbarB-self.seB*3)
        stop = max(self.xbarA+self.seA*3, self.xbarB+self.seB*3)
        x = np.linspace(start, stop, 1000)
        yA = stats.norm(self.xbarA, self.seA).pdf(x)
        yB = stats.norm(self.xbarB, self.seB).pdf(x)
        ax.plot(x, yB, color = self.colB, label = self.name_list[1], lw=3)
        ax.axvline(x=self.xbarB, color=self.colB, linestyle='--', lw=3, 
                   label=f'{self.name_list[1]} mean = {self.xbarB:2.2f}, std error = {self.seB:2.2f}, n = {self.nB}')
        ax.plot(x, yA, color = self.colA, label = self.name_list[0], lw=3)
        ax.axvline(x=self.xbarA, color=self.colA, linestyle='--', lw=3, 
                   label=f'{self.name_list[0]} mean = {self.xbarA:2.2f}, std error = {self.seA:2.2f}, n = {self.nA}')
        ax.set_xlabel('average number of days to purchase contract')
        ax.set_ylabel('pdf')
        ax.set_title('Means Sampling Distributions for Pre-Covid & During Covid')
        ax.legend()

    def plot_diff_of_means(self, ax):
        p = self.p_value_by_hand()
        if self.alternative == 'two-sided':
            start = min(0-self.pooled_sd*3, self.diff-self.pooled_sd*3)
            stop = max(self.pooled_sd*3, self.diff+self.pooled_sd*3)
            x = np.linspace(start, stop, 1000)
            ynull = stats.norm(0, self.pooled_sd).pdf(x)
            yalt = stats.norm(self.diff, self.pooled_sd).pdf(x)
            ax.plot(x, ynull, color = self.col_null, label = 'null',  lw=3)
            ax.axvline(x=0, color=self.col_null, linestyle='--', lw=3, label=f'null mean: 0')
            ax.plot(x, yalt, color = self.col_alt, label = 'alternative', lw=3)
            ax.axvline(x=self.diff, color=self.col_alt, linestyle='--', 
                       lw=3, label=f'alt mean: {self.diff:2.2f}, pval: {p:2.6f}')
            ax.axvline(x = self.rr_l, color=self.col_rr, 
                    linestyle =':', lw=3 , label=f'critical value: +/-{abs(self.rr_l):2.2f}')
            ax.axvline(x = self.rr_h, color=self.col_rr, linestyle =':', lw=3)
            ax.fill_between(x, ynull, 0, where=(x<=self.rr_l), 
                            color=self.col_rr, alpha=.5, label='rejection region')
            ax.fill_between(x, yalt, 0, where=(x<=self.rr_l), color=self.col_power, alpha=.5)
            ax.fill_between(x, ynull, 0, where=(x>=self.rr_h), 
                            color=self.col_rr, alpha=.5)
            ax.fill_between(x, yalt, 0, where=(x>=self.rr_h), color=self.col_power, alpha=.5, 
                            label=f'power: {self.power:2.5f}')
            
            
            
        elif self.alternative == 'less':
            start = min(self.pooled_sd*3, -self.diff-self.pooled_sd*3)
            stop = max(self.pooled_sd*3, -self.diff+self.pooled_sd*3)
            x = np.linspace(start, stop, 1000)
            yalt = stats.norm(-self.diff, self.pooled_sd).pdf(x)
            ynull = stats.norm(0, self.pooled_sd).pdf(x)
            ax.plot(x, yalt, color = self.col_alt, label = 'alternative', lw=3)
            ax.axvline(x=-self.diff, color=self.col_alt, linestyle='--', lw=3, label=f'alt mean: {-self.diff:2.2f}, pval: {p:2.6f}')
            ax.plot(x, ynull, color = self.col_null, label = 'null',  lw=3)
            ax.axvline(x=0, color=self.col_null, linestyle='--', lw=3, label=f'null mean: 0')
            ax.axvline(x = self.rr_lt, color=self.col_rr, 
                       linestyle =':', lw=3 , label=f'critical value: {self.rr_lt:2.2f}')
            ax.fill_between(x, ynull, 0, where=(x<=self.rr_lt), 
                            color=self.col_rr, alpha=.5, label='rejection region')
            ax.fill_between(x, yalt, 0, where=(x<=self.rr_lt), color=self.col_power, 
                            alpha=.5, label = f'power: {self.power:2.5f}')
            
            
            
        else:
            start = min(0-self.pooled_sd*3, self.diff-self.pooled_sd*3)
            stop = max(self.pooled_sd*3, self.diff+self.pooled_sd*3)
            x = np.linspace(start, stop, 1000)
            ynull = stats.norm(0, self.pooled_sd).pdf(x)
            yalt = stats.norm(self.diff, self.pooled_sd).pdf(x)
            ax.plot(x, ynull, color = self.col_null, label = 'null',  lw=3)
            ax.axvline(x=0, color=self.col_null, linestyle='--', lw=3, label=f'null mean: 0')
            ax.plot(x, yalt, color = self.col_alt, label = 'alternative', lw=3)
            ax.axvline(x=self.diff, color=self.col_alt, linestyle='--', lw=3, label=f'alt mean: {self.diff:2.2f}, pval: {p:2.6f}')
            ax.axvline(x = self.rr_gt, color=self.col_rr, 
                       linestyle =':', lw=3 , label=f'critical value: {self.rr_gt:2.2f}')
            ax.fill_between(x, ynull, 0, where=(x>=self.rr_gt), 
                            color=self.col_rr, alpha=.5, label='rejection region')
            ax.fill_between(x, yalt, 0, where=(x>=self.rr_gt), color=self.col_power, alpha=.5, 
                            label=f'power: {self.power:2.5f}')
            
            a
        ax.set_xlabel('days')
        ax.set_ylabel('pdf')
        ax.set_title('Difference of Means Null and Alternative Distributions')
        ax.legend(facecolor='lightgrey')

if __name__=="__main__":

    train = pd.read_csv('../data/train.csv')
    s1 = train[['dtp','pcd_mo_year']][train.pcd_mo_year>='2020-03'].set_index('pcd_mo_year')
    s1 = [val for val in s1.dtp]
    s2 = train[['dtp','pcd_mo_year']][train.pcd_mo_year<'2020-03'].set_index('pcd_mo_year')
    s2 = [val for val in s2.dtp]
    
    ht = HypTest([s1, s2], ['post', 'pre'], alternative='two-sided')

    print(ht.descriptive_stats_df())
    print(ht.inferential_stats_df())

    print(ht.pooled_sd)
    print(f'cohens-d for similar variance: {ht.effect_cohens_d}')
    print(f'glass-delta for different sizes: {ht.effect_glass_delta}') 
    print(f'hedges-g for different sizes and variances: {ht.effect_hedges_g}')

    print(f'p-value-scipy = {ht.p_value()}')
    print(f'pval-byhand = {ht.p_value_by_hand()}')
    print(f'pval-1sided = {ht.pval_1sided}')
    print(f'pval-2sided = {ht.pval_2sided}')
    print(f'pval-z-2sided = {ht.pval_z_1sided}')
    
    print(f'cv_l = {ht.cv_l}')
    print(f'cv_r = {ht.cv_r}')


    fig, ax = plt.subplots(figsize=(20,5))
    ht.plot_mean_sd(ax)
#     fig, ax = plt.subplots(figsize=(20,5))
#     ht.box_plot(ax)
    fig, ax = plt.subplots(figsize=(20,5))
    ht.plot_hists(ax)
    fig, ax = plt.subplots(figsize=(20,5))
    ht.plot_sorted(ax)
    fig, ax = plt.subplots(figsize=(20,5))
    ht.plot_sampling_distributions(ax)
    fig, ax = plt.subplots(figsize=(20,5))
    ht.plot_diff_of_means(ax)
    plt.show()

    