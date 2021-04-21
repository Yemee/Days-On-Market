import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

# Helper functions
def br_stick(x, br,a=1, b=1, c=1, d=1, e=.15):
    return a*(br-x) * b*(x<br) + c*(br-x)**2 * d*(x<br) + x + e*x**2+6

def sine(x):
    return 3.12*np.sin((np.pi/60)*(x+1.5))+33.2

def combine_dict(d1,d2):
    d = d1.copy()
    for k, v in d.items():
        d[k] += d2[k]
    return d

def data_loss_dict(d, n=34519):
    df = pd.DataFrame.from_dict(d, orient='index', columns=['preserved data'])
    x1 = list(d.values())
    x2 = x1[1:]+[x1[-1]]
    diff = [0]+list(np.array(x1)-np.array(x2))[:-1]
    df['loss'] = np.array(diff)
    df['% loss'] = ( np.array(diff)/n)*100
    new_df = df.T
    new_df['totals'] = [0, df[['loss', '% loss']].sum(axis=0)[0], df[['loss', '% loss']].sum(axis=0)[1]]
    return new_df.T

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

# Graphing functions
def data_loss_index(ax, df, col='dtp', thresh=180, color = 'yellowgreen', title = 'days to purchase contract'):
    n = len(df[col])
    ax.scatter(np.arange(n), df[col], alpha=.5, color=color)
    ax.axhline(y=thresh, color = 'k', label = f'{thresh} days', ls='--')
    ax.set_title(title, fontsize=25)
    ax.set_ylabel('days', fontsize=20)
    ax.set_xlabel('index', fontsize=20)
    ax.text(5000, 400, f'data loss: {(df[col]>thresh).sum()} datapoints = {int(round((df[col]>thresh).sum()/df.shape[0],2)*100)}%', fontsize=20)
    ax.fill_between(x = np.arange(df.shape[0]), y1=thresh, y2=df[col].max(), facecolor='slategrey', alpha=.2)
    ax.legend(loc='upper center', facecolor='white', fontsize=20)
    ax.set_ylim(0,600)
    ax.set_xlim(0,df.shape[0]);
    
def data_loss_target(ax, df, thresh_dtp=180, thresh_dtc=77):
    df = df[['dom', 'dtp', 'dtc']] 
    df['idx'] = np.arange(df.shape[0])
    df_keep = df[(df.dtp<=thresh_dtp) & (df.dtc<=thresh_dtc)]
    df_drop_dtp = df[~(df.dtp<=thresh_dtp)]
    df_drop_dtc = df[~(df.dtc<=thresh_dtc)]
    ax.scatter(df_keep.idx, df_keep.dom, color = 'tab:orange', alpha=.05, 
               label = f'total loss: {df.shape[0]-df_keep.shape[0]}')
    ax.scatter(df_drop_dtp.idx, df_drop_dtp.dom, color='yellowgreen', alpha = 1, label = f'dtp loss: {df_drop_dtp.shape[0]}')
    ax.scatter(df_drop_dtc.idx, df_drop_dtc.dom, color='plum', alpha = 1, label = f'dtc loss: {df_drop_dtc.shape[0]}')
    ax.scatter(0,0,color='tab:orange',label = f'data kept: {int(round(df_keep.shape[0]/df.shape[0],2)*100)}%')
    ax.set_ylim(0,600)
    ax.set_title('days on market',fontsize=25)
    ax.set_ylabel('days', fontsize=20)
    ax.set_xlabel('index', fontsize=20)
    ax.set_xlim(0,df.shape[0])
    ax.legend(loc = 'upper right', fontsize=20, facecolor='white');

# Winsor outlier class for high side outliers > Q3 + 1.5*IQR
class Winsorize():
    def __init__(self, df, col_list):
        self.df = df
        self.col_list = col_list
    
    def find_outlier_cut_off(self, arr):
        q1 = np.percentile(arr,25)
        q3 = np.percentile(arr,75)
        iqr = q3 - q1
        cut_off = q3 + 1.5*iqr
        return cut_off

    def find_count(self, arr, cut_off):
        return  len(arr[arr>=cut_off])
    
    def find_perc(self, arr, cut_off):
        return len(arr[arr>cut_off])/self.df.shape[0]

    def out_df(self):
        d = {}
        for col in self.col_list:
            cut_off = self.find_outlier_cut_off(self.df[col])
            cnt = self.find_count(self.df[col], cut_off)
            pct = (self.find_perc(self.df[col], cut_off))*100
            pct = round(pct,2)
            d[col] = (cut_off, cnt, pct)
        return pd.DataFrame.from_dict(d, orient='index', columns=['cut_off', 'count', 'percent'])#.set_index(['cut_off', 'count', 'percent'])
    

# Split class for pre-cleaning split into traning and holdout
class SplitData():
    '''
    input: full export dataframe from REColorado
    reduce to sufficiently populated features
    convert dates to datetime and create targets
    create training, quick, slow, and holdout sets
    '''

    identity = ['Listing Id', 'Latitude', 'Longitude', 
                'Street Name', 'Street Number', 'Street Suffix']
    # used to engineer targets
    dates = ['Listing Contract Date', 'Purchase Contract Date', 'Close Date']
    # used to engineer 'ratio', 'fin_sqft' (numeric)
    utility = ['Original List Price', 'Close Price', 
               'Below Grade Finished Area', 'Above Grade Finished Area' ]
    nums = ['Buyer Agency Compensation', 'Bathrooms Total Integer', 
            'Garage Spaces', 'Bedrooms Total', 'Contract Min Earnest', 
            'Lot Size Square Feet', 'Walk Score', 'PSF Finished', 
            'Fireplaces Total', 'Year Built', 'Photos Count']
    levs = ['Buyer Financing', 'Levels', 'Ownership', 'List Agent AOR',  
            'Virtual Tour URL Unbranded', 'Basement YN', 
            'Dual Variable Compensation YN']
    # used for NLP
    text = ['Private Remarks', 'Public Remarks', 'Showing Instructions']
    # potential for slicing or engineering
    many_levs = ['Elementary School', 'High School', 
                 'Middle Or Junior School', 'Postal Code', 'Subdivision Name',
                 'Zoning','Elementary School District']

    def __init__(self, df):
        self.df = df

    def feature_df(self):
        return self.df[self.identity+self.dates+self.utility+self.nums+self.levs+self.text+self.many_levs]

    def change_to_datetime(self):
        df = self.feature_df().copy()
        for col in self.dates:
            df[col] = pd.to_datetime(df[col])
        return df

    def create_targets(self):
        df = self.change_to_datetime()
        df['dtp'] = (df[self.dates[1]] - df[self.dates[0]])/np.timedelta64(1, 'D') 
        df['dtc'] = (df[self.dates[2]] - df[self.dates[1]])//np.timedelta64(1, 'D') 
        df['dom'] = df['dtp'] + df['dtc']
        return df

    def train_quick_slow_holdout(self):
        train_df, holdout_df = train_test_split(self.create_targets(), 
                                                test_size=.2, 
                                                shuffle=True,
                                                random_state=42)
        md = train_df['dtp'].mode()[0]
        quick_df = train_df[train_df['dtp']<=2*md]
        slow_df = train_df[train_df['dtp']>2*md]
        for df in [train_df, quick_df, slow_df, holdout_df]:
            df.set_index('Listing Id',inplace=True)
        return train_df, quick_df, slow_df, holdout_df
    
    
# cleaning class and data loss dict creator
class DataFeatures():
    '''

    '''
    list_of_date_names = ['lcd', 'pcd', 'cd']
    dates = ['Listing Contract Date', 'Purchase Contract Date', 'Close Date']
    text = ['Private Remarks', 'Public Remarks', 'Showing Instructions']
    attributes = ['quick', 'dtp', 'dtc', 'dom', 
                  'ratio', 'fin_sqft', 'inventory', 'good_agent', 
                  'n_baths', 'n_beds', 'n_fireplaces', 'n_photos',
                  'orig_price', 'close_price', 'lot_sqft', 'walkscore', 'psf', 
                  'em', 'ba_comp',  
                  'back_on_market', 'is_vacant',  
                  'has_fireplace', 'has_bsmt', 'one_story',
                  'is_realtor', 'offer_var_comp', 'has_virtual_tour', 'has_showing_instr',  
                  'with_cash', 'is_corp_owner', 'is_warm',
                  'financing', 'ownership', 'levels', 
                  'garage_spaces', 'year_built',
                  'garage_size', 'garage_zero', 'garage_one', 'garage_two', 
                  'period', 'period_turn', 'period_midmod', 'period_modern', 
                  'photo', 'photo_small', 'photo_medium', 'photo_large',
                  'private','public', 'showing_instructions', 'all', 'low_all',
                  'lat', 'long', 'zip',
                  'Street Name', 'Street Number', 'str_suffix', 
                  'Elementary School', 'Middle Or Junior School', 'High School',
                  #'Elementary School District', 'Subdivision Name', 'Zoning',
                  'lcd_day_of_month','lcd_day_of_week', 'lcd_month', 'lcd_year', 
                  'lcd_day_mo_year', 'lcd_mo_year', 
                  'pcd_day_of_month', 'pcd_day_of_week', 'pcd_month', 'pcd_year', 
                  'pcd_day_mo_year', 'pcd_mo_year', 
                  'cd_day_of_month', 'cd_day_of_week', 'cd_month', 'cd_year', 
                  'cd_day_mo_year', 'cd_mo_year']
    attributes_subset = []
    
    def __init__(self, df):
        self.df = df

    def extract_date_info(self):
        '''get more specific date information'''
        df = self.df.copy()
        for idx, col in enumerate(self.dates):
            df[self.list_of_date_names[idx]+'_day_of_month'] = pd.DatetimeIndex(df[col]).day
            df[self.list_of_date_names[idx]+'_day_of_week'] = pd.DatetimeIndex(df[col]).dayofweek
            df[self.list_of_date_names[idx]+'_month'] = pd.DatetimeIndex(df[col]).month
            df[self.list_of_date_names[idx]+'_year'] = pd.DatetimeIndex(df[col]).year
            df[self.list_of_date_names[idx]+'_day_mo_year'] = pd.to_datetime(df[col]).dt.to_period('D')
            df[self.list_of_date_names[idx]+'_mo_year'] = pd.to_datetime(df[col]).dt.to_period('M')
        return df

    def inventory_feature(self): 
        '''this is a count of contracts by month, not market inventory'''
        df = self.extract_date_info()
        temp = df.groupby(['pcd_mo_year']).count()
        temp = temp[['dom']]
        df = df.join(temp, on='pcd_mo_year', how='left', rsuffix='_inventory')
        return df

    def ratio_feature(self):
        df = self.inventory_feature()
        df['ratio'] = df['Original List Price']/df['Close Price']
        return df
    
    def fin_sqft_feature(self):
        df = self.ratio_feature()
        df['Below Grade Finished Area'].fillna(0, inplace=True)
        df['fin_sqft'] = df['Above Grade Finished Area'] + df['Below Grade Finished Area']
        return df
    
    def impute_em_feature(self):
        df = self.fin_sqft_feature()
        temp = df[df['Contract Min Earnest']>0][['Original List Price', 'Contract Min Earnest']]
        temp['y'] = np.log(temp['Contract Min Earnest'])
        temp['x'] = np.log(temp['Original List Price'])
        lmod = smf.ols('y~x', data=temp).fit()
        betas = lmod.params
        df['em'] = [val if val>0 else(np.e**betas[0])*(new_x**betas[1]) 
                    for val, new_x in zip(df['Contract Min Earnest'], 
                    df['Original List Price'])]
        return df

    def impute_psf_feature(self):
        df = self.impute_em_feature()
        df['PSF Finished'].fillna(df['PSF Finished'].mean(), inplace=True)
        df['psf'] = df['Close Price']/df['fin_sqft']
        return df

    def buyer_agency_feature(self):
        df = self.impute_psf_feature()
        temp_list = []
        for string in df['Buyer Agency Compensation'].values:
            temp_string = ''
            for char in str(string):
                if char in '.0123456789':
                    temp_string+=char
            temp_list.append(temp_string)
        df['Buyer Agency Compensation'] = temp_list
        df['Buyer Agency Compensation'] = pd.to_numeric(df['Buyer Agency Compensation'], 
                                                        errors='coerce')
        df['Buyer Agency Compensation'] = [val if val<5 else (val/price)*100 for val, 
                                           price in zip(df['Buyer Agency Compensation'], 
                                           df['Close Price'])]
        df['Buyer Agency Compensation'].fillna(2.8, inplace=True)                                         
        df['ba_comp'] = df['Buyer Agency Compensation']
        return df

    def has_showing_instr_feature(self):
        df = self.buyer_agency_feature()
        df['has_showing_instr'] = ~df['Showing Instructions'].isna()
        return df

    def nlp_feature(self):
        df = self.has_showing_instr_feature()
        renames = ['private', 'public', 'showing_instructions']
        for name, col in zip(renames, self.text):
            df[name] = df[col].fillna('none')
        df['all'] = df['private']+' '+['public']+' '+['showing_instructions']
        df['low_all'] = [string.lower() for string in df['all']]
        return df
    
    def back_on_market_feature(self):
        df = self.nlp_feature()
        terms = ['back on market', 'cold feet', 'fell through']
        df['back_on_market'] = df['low_all'].str.contains('|'.join(terms))==True
        return df
        
    def is_vacant_feature(self):
        df = self.back_on_market_feature()
        terms = ['vacant', 'go & show', 'go and show', 'go show']
        df['is_vacant'] = df['low_all'].str.contains('|'.join(terms))==True
        return df
    
    def has_fireplace_feature(self):
        df = self.is_vacant_feature()
        df['Fireplaces Total'].fillna(0, inplace=True)
        df['has_fireplace'] = df['Fireplaces Total']>0
        return df
    
    def is_realtor_feature(self):
        df = self.has_fireplace_feature()
        df['List Agent AOR'] = df['List Agent AOR'].fillna('Non Board Members')
        df['is_realtor'] = df['List Agent AOR']!='Non Board Members'
        return df

    def offer_variable_comp_feature(self):
        df = self.is_realtor_feature()
        df['Dual Variable Compensation YN'].fillna(False, inplace=True)
        df['offer_var_comp'] = df['Dual Variable Compensation YN']
        return df
    
    def has_virtual_tour_feature(self):
        df = self.offer_variable_comp_feature()
        df['has_virtual_tour'] = ~df['Virtual Tour URL Unbranded'].isna()
        return df
    
    def is_warm_feature(self):
        df = self.has_virtual_tour_feature()
        df['is_warm'] = (df.pcd_month>=3) & (df.pcd_month<=8)
        return df

    def quick_feature(self):
        df = self.is_warm_feature()
        lt = 2*df['dtp'].mode()[0]
        df['quick'] = (df['dtp']<=(lt+1))
        return df

    def with_cash_feature(self):
        df = self.quick_feature()
        df['with_cash'] = df['Buyer Financing']=='Cash'
        return df
    
    def one_story_feature(self):
        df = self.with_cash_feature()
        df['one_story'] = df['Levels']=='One'
        return df

    def is_corp_owner_feature(self):
        df = self.one_story_feature()
        df['Ownership'].fillna('other', inplace=True)
        df['is_corp_owner'] = df['Ownership']=='Corporation/Trust'
        return df
    
    def has_bsmt_feature(self):
        df = self.is_corp_owner_feature()
        df['Basement YN'].fillna(False, inplace=True)
        df['has_bsmt'] = df['Basement YN']==True
        return df

    def garage_size_feature(self):
        df = self.has_bsmt_feature()
        df['garage_size'] = ['zero' if val==0 else 'one' if val==1 else 'two' for val in df['Garage Spaces']]
        return df
    
    def num_photos_feature(self):
        df = self.garage_size_feature()
        temp_list = []
        for val in df['Photos Count']:
            if val<2:
                temp_list.append('small')
            elif val<36:
                temp_list.append('medium')
            else: 
                temp_list.append('large')
        df['photo'] = temp_list
        return df

    def period_built_feature(self):
        df = self.num_photos_feature()
        temp_list = []
        for val in df['Year Built']:
            if val<1930:
                temp_list.append('turn')
            elif val<1976:
                temp_list.append('midmod')
            else:
                temp_list.append('modern')
        df['period'] = temp_list
        return df

    def financing_cleanup(self):
        df = self.period_built_feature()
        df['Buyer Financing'].fillna('unk', inplace=True)
        temp_list = []
        for val in df['Buyer Financing']:
            if val=='Conventional':
                temp_list.append('conv')
            elif val=='Cash':
                temp_list.append('cash')
            elif val=='FHA':
                temp_list.append('fha')
            elif val=='VA':
                temp_list.append('va')
            else:
                temp_list.append('other')
        df['financing'] = temp_list
        return df

    def zip_code_cleanup(self):
        df = self.financing_cleanup()
        df['zip'] = df['Postal Code'].astype('str').str[:5]
        return df

    def identity_cleanup(self):
        df = self.zip_code_cleanup()
        df['str_suffix'] = df['Street Suffix']
        df['str_suffix'].fillna(' ', inplace=True)
        return df

    def rename_features(self):
        df = self.identity_cleanup()
        df['lat'], df['long'] = df['Latitude'], df['Longitude']
        new_names = ['n_baths', 'n_beds', 'orig_price', 
                     'close_price', 'inventory', 'lot_sqft', 'walkscore',
                     'n_fireplaces', 'year_built', 'n_photos', 'levels',
                     'ownership', 'Latitude', 'Longitude',
                     'garage_spaces']
        old_names = ['Bathrooms Total Integer', 'Bedrooms Total', 'Original List Price', 
                     'Close Price', 'dom_inventory', 'Lot Size Square Feet', 
                     'Walk Score', 'Fireplaces Total', 'Year Built', 
                     'Photos Count', 'Levels', 'Ownership', 'lat', 'long',
                     'Garage Spaces']
        for new, old in zip(new_names, old_names):
            df[new] = df[old]

        # make good agent feature
        df['good_agent'] = df[['is_realtor',
                               'offer_var_comp', 
                               'has_virtual_tour', 
                               'has_showing_instr']].sum(axis=1)/4
        return df

    def dummy_variables(self):
        df = self.rename_features()
        col_list = ['garage_size', 'period', 'photo']
        prefix_list =  ['garage', 'period', 'photo']
        for col, pre in zip(col_list, prefix_list):
            pre = pd.get_dummies(df[col], drop_first=False, prefix=pre)
            df = df.join(pre)
        return df
    
    def clean_data(self):
        df = self.dummy_variables()

        df['Street Suffix'].fillna('', inplace=True)
        for col in ['n_baths', 'n_beds', 'lot_sqft', 'walkscore']:
            df[col].fillna(df[col].mean(), inplace=True)
        df['n_fireplaces'] = [7 if val>7 else val for val in df['n_fireplaces']]

        data_loss_dict = {}
        data_loss_dict['orig'] = df.shape[0]
        df = df[(df['lat']>39.5) & (df['lat']<40)]
        data_loss_dict['lat'] = df.shape[0]
        df = df[(df['long']>-105.5) & (df['long']<-104.5)]
        data_loss_dict['long'] = df.shape[0]
        df = df[df['n_baths']>0]
        data_loss_dict['n_baths'] = df.shape[0]
        df = df[df['dtc']>0]
        data_loss_dict['dtc > 0'] = df.shape[0]
        return df, data_loss_dict

    def ready_data(self):
        df, data_loss_dict = self.clean_data()
        df = df[(df.lcd_year - df.year_built)>12]
        data_loss_dict['new_build'] = df.shape[0]
        df = df[(df['ratio']>=.85) & (df['ratio']<=1.36)]
        data_loss_dict['ratio 99'] = df.shape[0]
        df = df[df['dtp']<=180]
        data_loss_dict['dtp 180'] = df.shape[0]
        df = df[df['dtc']<=77]
        data_loss_dict['dtc 77'] = df.shape[0]
        idx = list(df[(df.with_cash==False) & (df.dtc<7)].index)
        df.drop(idx, inplace=True)
        data_loss_dict['TRID'] = df.shape[0]
        return df[self.attributes], data_loss_dict


if __name__=="__main__":

    JanJun2015 = pd.read_csv('../data/JanJun2015.csv', low_memory=False)
    JulDec2015 = pd.read_csv('../data/JulDec2015.csv', low_memory=False)
    JanJun2016 = pd.read_csv('../data/JanJun2016.csv', low_memory=False)
    JulDec2016 = pd.read_csv('../data/JulDec2016.csv', low_memory=False) 
    JanJun2017 = pd.read_csv('../data/JanJun2017.csv', low_memory=False)
    JulDec2017 = pd.read_csv('../data/JulDec2017.csv', low_memory=False)
    JanJun2018 = pd.read_csv('../data/JanJun2018.csv', low_memory=False)
    JulDec2018 = pd.read_csv('../data/JulDec2018.csv', low_memory=False)
    JanJun2019 = pd.read_csv('../data/JanJun2019.csv', low_memory=False)
    JulDec2019 = pd.read_csv('../data/JulDec2019.csv', low_memory=False)
    JanJun2020 = pd.read_csv('../data/JanJun2020.csv', low_memory=False)
    JulDec2020 = pd.read_csv('../data/JulDec2020.csv', low_memory=False)

    # combine datasets
    file_list = [JanJun2015, JulDec2015, JanJun2016, JulDec2016, JanJun2017, 
                 JulDec2017, JanJun2018, JulDec2018, JanJun2019, JulDec2019]
    orig_df = pd.concat(file_list)
    
    covid_list = [JanJun2020, JulDec2020]
    covid_df = pd.concat(covid_list)
    
    all_the_data_list = [JanJun2015, JulDec2015, JanJun2016, JulDec2016, JanJun2017, 
                 JulDec2017, JanJun2018, JulDec2018, JanJun2019, JulDec2019, JanJun2020, JulDec2020]
    all_the_data_df = pd.concat(all_the_data_list)
    
    # split data to protect against leakage
    train_df, quick_df, slow_df, holdout_df = SplitData(orig_df).train_quick_slow_holdout()
    train_df.to_csv('../data/train_df.csv')
    quick_df.to_csv('../data/quick_df.csv')
    slow_df.to_csv('../data/slow_df.csv')
    holdout_df.to_csv('../data/holdout_df.csv')
    
    covid_train_df, covid_quick_df, covid_slow_df, covid_holdout_df = SplitData(covid_df).train_quick_slow_holdout()
    covid_train_df.to_csv('../data/covid_train_df.csv')
    covid_quick_df.to_csv('../data/covid_quick_df.csv')
    covid_slow_df.to_csv('../data/covid_slow_df.csv')
    covid_holdout_df.to_csv('../data/covid_holdout_df.csv')
    
    all_the_data_train_df, all_the_data_quick_df, all_the_data_slow_df, all_the_data_holdout_df = SplitData(all_the_data_df).train_quick_slow_holdout()
    all_the_data_train_df.to_csv('../data/all_the_data_train_df.csv')
    all_the_data_quick_df.to_csv('../data/all_the_data_quick_df.csv')
    all_the_data_slow_df.to_csv('../data/all_the_data_slow_df.csv')
    all_the_data_holdout_df.to_csv('../data/all_the_data_holdout_df.csv')

    # clean data sets separately
    train, train_data_loss_dict = DataFeatures(train_df).ready_data()
    quick, quick_data_loss_dict = DataFeatures(quick_df).ready_data()
    slow, slow_data_loss_dict = DataFeatures(slow_df).ready_data()
    holdout, holdout_data_loss_dict = DataFeatures(holdout_df).ready_data()
    
    covid_train, covid_train_data_loss_dict = DataFeatures(covid_train_df).ready_data()
    covid_quick, covid_quick_data_loss_dict = DataFeatures(covid_quick_df).ready_data()
    covid_slow, covid_slow_data_loss_dict = DataFeatures(covid_slow_df).ready_data()
    covid_holdout, covid_holdout_data_loss_dict = DataFeatures(covid_holdout_df).ready_data()
    
    all_the_data_train, all_the_data_train_data_loss_dict = DataFeatures(all_the_data_train_df).ready_data()
    all_the_data_quick, all_the_data_quick_data_loss_dict = DataFeatures(all_the_data_quick_df).ready_data()
    all_the_data_slow, all_the_data_slow_data_loss_dict = DataFeatures(all_the_data_slow_df).ready_data()
    all_the_data_holdout, all_the_data_holdout_data_loss_dict = DataFeatures(all_the_data_holdout_df).ready_data()
    
    # save dataframes for use in EDA and modeling
    train.to_csv('../data/train.csv')
    quick.to_csv('../data/quick.csv')
    slow.to_csv('../data/slow.csv')
    holdout.to_csv('../data/holdout.csv')
    
    covid_train.to_csv('../data/covid_train.csv')
    covid_quick.to_csv('../data/covid_quick.csv')
    covid_slow.to_csv('../data/covid_slow.csv')
    covid_holdout.to_csv('../data/covid_holdout.csv')
    
    all_the_data_train.to_csv('../data/all_the_data_train.csv')
    all_the_data_quick.to_csv('../data/all_the_data_quick.csv')
    all_the_data_slow.to_csv('../data/all_the_data_slow.csv')
    all_the_data_holdout.to_csv('../data/all_the_data_holdout.csv')

    print('train')
    print(train.shape, train_data_loss_dict)
    print('quick')
    print(quick.shape, quick_data_loss_dict)
    print('slow')
    print(slow.shape, slow_data_loss_dict)
    print('holdout')
    print(holdout.shape, holdout_data_loss_dict)
    print('covid')
    print(covid_train.shape, covid_train_data_loss_dict)
    print('all_train')
    print(all_the_data_train.shape, all_the_data_train_data_loss_dict)
    print('all_holdout')
    print(all_the_data_holdout.shape, all_the_data_holdout_data_loss_dict)
    
    
    # auxilary data
    active = pd.read_csv('../data/active.csv')
    new = pd.read_csv('../data/new.csv')
    pending = pd.read_csv('../data/pending.csv')
    closed = pd.read_csv('../data/closed.csv')
    showings = pd.read_csv('../data/showings.csv')

    active = convert_to_date(active, 'date', 'active')
    new = convert_to_date(new, 'date', 'new')
    pending = convert_to_date(pending, 'date', 'pending')
    closed = convert_to_date(closed, 'date', 'closed')
    showings = convert_to_date(showings, 'date', 'showings')
    activity = active.join(new).join(pending).join(closed).join(showings)

    all_the_data_train = pd.read_csv('../data/all_the_data_train.csv')
    all_the_data_holdout = pd.read_csv('../data/all_the_data_holdout.csv')
    all_df = pd.concat([all_the_data_train, all_the_data_holdout])

    df = all_df[['dtp', 'dtc', 'pcd_mo_year']]
    df = df.groupby('pcd_mo_year').mean()
    activity = df.join(activity)
    activity.to_csv('../data/activity.csv')
    
    
    


   
    
    
    