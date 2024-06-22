#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# In[2]:


import pandas as pd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
# from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


import random
import tqdm


# In[3]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[4]:


# # Load the data
# # ads_usecols=['log_id', 'label',  'user_id', 'adv_id', 'task_id',
# #              'age', 'gender', 'residence', 'pt_d','series_dev',
# #              'device_name', 'device_size', 'net_type',
# #               'creat_type_cd', 'ad_click_list_v001', 'ad_close_list_v001']
# ads_csv_raw = pd.read_csv("../data/train_data_ads.csv") # usecols =  ads_usecols )

# # feeds_usecols = [ 'u_userId', 'u_phonePrice', 'u_feedLifeCycle',  'u_browserLifeCycle', 'i_docId', 'i_cat', 'u_click_ca2_news' , 'i_dislikeTimes']
# feeds_csv_raw = pd.read_csv("../data/train_data_feeds.csv") # usecols =  feeds_usecols,   )

 
ads_csv_raw = pd.read_csv("./train_data_ads.csv") 


# In[5]:


ads_csv_raw.head()


# In[6]:


ads_csv_raw = reduce_mem_usage(ads_csv_raw)


# In[7]:


import gc
gc.collect()


# In[8]:


feeds_csv_raw = pd.read_csv("./train_data_feeds.csv"    )


# In[9]:


feeds_csv_raw.head()


# In[10]:


feeds_csv_raw = reduce_mem_usage(feeds_csv_raw)


# In[11]:


# ads_csv_raw = ads_csv.copy()
# feeds_csv_raw = feeds_csv.copy()


# In[12]:


common_user_id =    list( set(ads_csv_raw['user_id']) & set(feeds_csv_raw['u_userId']) ) 


# In[13]:


pd.Series(data = common_user_id).to_csv("./common_user_id.csv", index = False)


# In[14]:


selected_user_id_sample =  random.sample( common_user_id, round(len(common_user_id) * 0.02) ) 


# In[15]:


len(selected_user_id_sample) 


# In[16]:


ads_csv = ads_csv_raw.loc[ads_csv_raw['user_id'].isin(selected_user_id_sample)]
feeds_csv = feeds_csv_raw.loc[feeds_csv_raw['u_userId'].isin(selected_user_id_sample)]


# In[17]:


del ads_csv_raw, feeds_csv_raw

##  FEATURE ENGINEER 

ads_csv['pt_d'] = pd.to_datetime(ads_csv['pt_d'], format='%Y%m%d%H%M')
# Temporal Features
ads_csv['hour'] = ads_csv['pt_d'].dt.hour
ads_csv['day_of_week'] = ads_csv['pt_d'].dt.dayofweek
# Calculate session length (assuming the provided data is ordered by time)
ads_csv['session_length'] = ads_csv.groupby('user_id')['pt_d'].diff().dt.total_seconds().fillna(0)
# Interaction Ratios
ads_csv['click_close_ratio'] = ads_csv['ad_click_list_v001'].str.count('\^') / (ads_csv['ad_close_list_v001'].str.count('\^') + 1)
# ad interaction rate
ads_csv['ad_click_count'] = ads_csv['ad_click_list_v001'].apply(lambda x: len(str(x).split('^')))
ads_csv['ad_close_count'] = ads_csv['ad_close_list_v001'].apply(lambda x: len(str(x).split('^')))
# # Device Usage Patterns
# device_encoder = OneHotEncoder()
# encoded_device_csv = device_encoder.fit_transform(ads_csv[['series_dev']]).toarray()
# Combine encoded features
# encoded_device_df_csv = pd.DataFrame(encoded_device_csv, columns=device_encoder.get_feature_names_out(['series_dev']))
# ads_csv = pd.concat([ads_csv.reset_index(drop=True), encoded_device_df_csv.reset_index(drop=True)], axis=1)
#  Engagement Scores
feeds_csv['engagement_score'] = feeds_csv['u_feedLifeCycle'] + feeds_csv['u_browserLifeCycle']
# interest app
feeds_csv['app_interest_feeds'] = feeds_csv['u_newsCatInterests'] .apply(lambda x: len(str(x).split('^')))


# In[19]:


gap_max =  3
gap_list = list(range(1, gap_max+1))  
print(f'gap list：{gap_list}')
cols = ['user_id',   'ad_click_list_v001',  'ad_close_list_v001',
        'u_newsCatInterestsST', 'u_refreshTimes']

for col in cols:
    for gap in gap_list:
        tmp = ads_csv.groupby([col])['pt_d'].shift(-gap)  #shift to left,future data point 
        ads_csv['ts_{}_{}_diff_last'.format(col, gap)] = tmp - ads_csv['pt_d']   

    for gap in gap_list:
        tmp = ads_csv.groupby([col])['pt_d'].shift(+gap)  #shift to right, past data point 
        ads_csv['ts_{}_{}_diff_next'.format(col, gap)] = tmp - ads_csv['pt_d']    


# In[20]:


for col in ads_csv.filter( regex =  '_diff') .columns : 
    ads_csv[col] = ads_csv[col].dt.total_seconds()


# In[21]:


# ads_csv.head()


# In[22]:


ads_csv['app_interest'] = ads_csv['u_newsCatInterestsST'].apply(lambda x: len(str(x).split('^')))


# In[23]:


ads_csv = ads_csv.select_dtypes(exclude=['object'])


# In[24]:


feeds_csv = feeds_csv.select_dtypes(exclude=['object'])



#  aggregate statistics
# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(ads_csv['age'], bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Gender Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='gender', data=ads_csv)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Distribution of Ad Click Counts
plt.figure(figsize=(10, 6))
sns.histplot(ads_csv['ad_click_count'], bins=20)
plt.title('Ad Click Count Distribution')
plt.xlabel('Ad Click Count')
plt.ylabel('Frequency')
plt.show()

# Distribution of Ad Close Counts
plt.figure(figsize=(10, 6))
sns.histplot(ads_csv['ad_close_count'], bins=20)
plt.title('Ad Close Count Distribution')
plt.xlabel('Ad Close Count')
plt.ylabel('Frequency')
plt.show()

# Distribution of Device Series Usage
plt.figure(figsize=(10, 6))
sns.countplot(x='series_dev', data=ads_csv)
plt.title('Device Series Distribution')
plt.xlabel('Device Series')
plt.ylabel('Count')
plt.show()

# Distribution of Engagement Scores
plt.figure(figsize=(10, 6))
sns.histplot(merged_csv['feeds_csv'], bins=20)
plt.title('Engagement Score Distribution')
plt.xlabel('Engagement Score')
plt.ylabel('Frequency')
plt.show()

# In[25]:


merged_csv = pd.merge(ads_csv, feeds_csv, left_on='user_id', right_on='u_userId', suffixes=('_ads', '_feeds'))


# In[26]:


merged_csv.shape


# In[27]:


merged_csv = reduce_mem_usage(merged_csv)
`

# merged_csv.to_csv("./merged_csv.csv", index = False)


# In[45]:


y = merged_csv['label_ads']
X = merged_csv[[
    # 'log_id',
 #'label_ads', 
 #'user_id',
 'age',
 'gender',
 'residence',
 'city',
 'city_rank',
 'series_dev',
 'series_group',
 'emui_dev',
 'device_name',
 'device_size',
 'net_type',
# 'task_id',
# 'adv_id',
 'creat_type_cd',
# 'adv_prim_id',
 'inter_type_cd',
 #'slot_id',
# 'site_id',
 # 'spread_app_id',
 'hispace_app_tags',
 'app_second_class',
 'app_score',
# 'pt_d',
 'u_refreshTimes_ads',
 'u_feedLifeCycle_ads',
 'hour',
 'day_of_week',
 'session_length',
 'click_close_ratio',
 'ad_click_count',
 'ad_close_count',
 'ts_user_id_1_diff_last',
 'ts_user_id_2_diff_last',
 'ts_user_id_3_diff_last',
 'ts_user_id_1_diff_next',
 'ts_user_id_2_diff_next',
 'ts_user_id_3_diff_next',
 'ts_ad_click_list_v001_1_diff_last',
 'ts_ad_click_list_v001_2_diff_last',
 'ts_ad_click_list_v001_3_diff_last',
 'ts_ad_click_list_v001_1_diff_next',
 'ts_ad_click_list_v001_2_diff_next',
 'ts_ad_click_list_v001_3_diff_next',
 'ts_ad_close_list_v001_1_diff_last',
 'ts_ad_close_list_v001_2_diff_last',
 'ts_ad_close_list_v001_3_diff_last',
 'ts_ad_close_list_v001_1_diff_next',
 'ts_ad_close_list_v001_2_diff_next',
 'ts_ad_close_list_v001_3_diff_next',
 'ts_u_newsCatInterestsST_1_diff_last',
 'ts_u_newsCatInterestsST_2_diff_last',
 'ts_u_newsCatInterestsST_3_diff_last',
 'ts_u_newsCatInterestsST_1_diff_next',
 'ts_u_newsCatInterestsST_2_diff_next',
 'ts_u_newsCatInterestsST_3_diff_next',
 'ts_u_refreshTimes_1_diff_last',
 'ts_u_refreshTimes_2_diff_last',
 'ts_u_refreshTimes_3_diff_last',
 'ts_u_refreshTimes_1_diff_next',
 'ts_u_refreshTimes_2_diff_next',
 'ts_u_refreshTimes_3_diff_next',
# 'u_userId',
 'u_phonePrice',
 # 'u_browserLifeCycle',
 # 'u_browserMode',
 # 'u_feedLifeCycle_feeds',
 # 'u_refreshTimes_feeds',
 # 'i_regionEntity',
 # 'i_cat',
 'i_dislikeTimes',
 'i_upTimes',
 # 'i_dtype',
 # 'e_ch',
 # 'e_m',
 # 'e_po',
 # 'e_pl',
 # 'e_rn',
 # 'e_section',
 # 'e_et',
 # 'label_feeds',
 # 'cillabel',
 # 'pro',
 'engagement_score',
 'app_interest']]
 


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # encoding
# def label_encode(series, series2):
#     unique = list(series.unique())
#     return series2.map(dict(zip(
#         unique, range(series.nunique())
#     )))

# for col in ['ad_click_list_v001','ad_click_list_v002','ad_click_list_v003',
#             'ad_close_list_v001','ad_close_list_v002','ad_close_list_v003',
#             'u_newsCatInterestsST']:
#     data_adds[col] = label_encode(data_adds[col], data_adds[col])

# cols = [f for f in data_feeds.columns if f not in ['label','u_userId']]
# for col in tqdm.tqdm(cols):
#     tmp = data_feeds.groupby(['u_userId'])[col].nunique().reset_index()
#     tmp.columns = ['user_id', col+'_feeds_nuni']
#     data_ads = data_adds.merge(tmp, on='user_id', how='left')

# cols = [f for f in data_feeds.columns if f not in ['u_userId','u_newsCatInterests','u_newsCatDislike','u_newsCatInterestsST','u_click_ca2_news','i_docId','i_s_sourceId','i_entities']]
# for col in tqdm.tqdm(cols):
#     tmp = data_feeds.groupby(['u_userId'])[col].mean().reset_index()
#     tmp.columns = ['user_id', col+'_feeds_mean']
#     data_ads = data_adds.merge(tmp, on='user_id', how='left')



# In[36]:

# # # use  random forest to calculate feature importance 
# # clf = RandomForestClassifier(n_estimators=100, random_state=42)
# # clf.fit(X, y)
# # importances = clf.feature_importances_
# # feature_importance_df = pd.DataFrame({
# #     'feature': X.columns,
# #     'importance': importances
# # }).sort_values(by='importance', ascending=False)

# # param_grid = {
# #     'n_estimators': [100, 200],
# #     'max_depth': [3, 5, 7],
# #     'learning_rate': [0.01, 0.1, 0.2]
# # }

# xgb_model = XGBClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X, y)
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print("Best parameters found: ", best_params)
# print("Best accuracy score: ", best_score)

# best_model = grid_search.best_estimator_
# cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
# print("Cross-validation accuracy scores: ", cv_scores)
# print("Mean cross-validation accuracy: ", cv_scores.mean())


# # AdaBoost 
# ada_model = AdaBoostClassifier(base_estimator=best_model, n_estimators=100, random_state=42)
# ada_model.fit(X, y)

# # Cross validation 
# ada_cv_scores = cross_val_score(ada_model, X, y, cv=10, scoring='accuracy')
# print("AdaBoost cross-validation accuracy scores: ", ada_cv_scores)
# print("Mean AdaBoost cross-validation accuracy: ", ada_cv_scores.mean())


#  VotingClassifier 
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model3 = AdaBoostClassifier(n_estimators=100, random_state=42)
model4 = LogisticRegression(random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('rf', model1),
    ('xgb', model2),
    ('ada', model3),
    ('lr', model4)
], voting='soft')

# train and evaluate 
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print(classification_report(y_test, y_pred))
