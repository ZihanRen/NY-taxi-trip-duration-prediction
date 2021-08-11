# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:49:14 2019
@author: zur74
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
from sklearn.model_selection import KFold
import datetime as dt
import datetime


# start time for feature extraction
t0 = dt.datetime.now()

def date_info(train):
    holiday = ['2016-01-01', '2016-01-21', '2016-02-14', '2016-02-18', '2016-05-27', '2016-07-04', '2016-09-02',
               '2016-10-14', '2016-11-05', '2016-11-11', '2016-11-28', '2016-12-25']
    pickup_time = train['pickup_datetime'].values
    pickup_date = np.empty_like(pickup_time)
    is_holiday = np.zeros_like(pickup_time)
    weekday = np.empty_like(pickup_time)
    hour = np.empty_like(pickup_time)
    for i in range(pickup_time.size):
        dt1=pickup_time[i].split(' ')[0]
        pickup_date[i] = dt1
        year, month, day = (int(x) for x in dt1.split('-'))
        weekday[i] = datetime.date(year, month, day).weekday()

        dt2 = pickup_time[i].split(' ')[1]
        hour[i] = int(dt2.split(':')[0])

        if pickup_date[i] in holiday:
            is_holiday[i] = 1
    train['is_holiday'] = np.array(list(is_holiday), dtype=np.float)

    return train


def preprocess(train):

    '''
    Return train with columns of
    The week ordinal of the year
    The hour of pick up time
    The minute of pick up time
    The weekhour of pickup time. Representative which week in which hour
    '''

    # convert object time data to datetime
    train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
    train['pickup_date'] = train['pickup_datetime'].dt.date
    # get the date value without time zone
    train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday
    train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear
    train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour
    train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute
    train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
    train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']

    train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
    return train


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def jfk_distance_h(lat1, lng1):

    lat2 = 40.6413
    lng2 = -73.7781
    h = haversine_array(lat1,lng1,lat2,lng2)
    return h


def LAG_distance_h(lat1, lng1):

    lat2 = 40.7769
    lng2 = -73.8740

    return haversine_array(lat1,lng1,lat2,lng2)


def distance_haversine_man_bearing(train):

    '''
    INPUT:
        train or test in the condition that

    '''

    # tradational distance
    train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values,
                 train['pickup_longitude'].values,
                 train['dropoff_latitude'].values,
                 train['dropoff_longitude'].values)

    train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values,
                 train['pickup_longitude'].values,
                 train['dropoff_latitude'].values,
                 train['dropoff_longitude'].values)

    train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values,
                 train['pickup_longitude'].values,
                 train['dropoff_latitude'].values,
                 train['dropoff_longitude'].values)


    train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
    train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2


    # distance to airports
    train.loc[:,'distpickjfk_h'] = jfk_distance_h(train['pickup_latitude'].values, train['pickup_longitude'].values)
    train.loc[:,'distdropjfk_h'] = jfk_distance_h(train['dropoff_latitude'].values, train['dropoff_longitude'].values)

    train.loc[:,'distpickLAG_h'] = LAG_distance_h(train['pickup_latitude'].values, train['pickup_longitude'].values)
    train.loc[:,'distdropLAG_h'] = LAG_distance_h(train['pickup_latitude'].values, train['pickup_longitude'].values)

    return train


def location_bin(train,test):


    train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)
    train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 2)
    train.loc[:, 'center_lat_bin'] = np.round(train['center_latitude'], 2)
    train.loc[:, 'center_long_bin'] = np.round(train['center_longitude'], 2)
    train.loc[:, 'pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))
    test.loc[:, 'pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)
    test.loc[:, 'pickup_long_bin'] = np.round(test['pickup_longitude'], 2)
    test.loc[:, 'center_lat_bin'] = np.round(test['center_latitude'], 2)
    test.loc[:, 'center_long_bin'] = np.round(test['center_longitude'], 2)
    test.loc[:, 'pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))

    return train,test


def speed(train):

    '''
    Only valid to train; A features used to extract more labels location cluster
    '''

    train.loc[:,'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
    train.loc[:,'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

    return train


def clean(train):

    # clean the trip duration according to above
    train = train[ (train['trip_duration']<22*3600) &
                            (train['trip_duration']>10) &
                            (train['distance_haversine']>0.01) &
                            (train['distpickjfk_h']<1e2) &
                            (train['distdropjfk_h']<1e2) &
                            (train['avg_speed_h']<100/3.6) ]


    train.loc[:,'log_trip_duration'] = np.log( train['trip_duration'].values + 1 )
    return train


def pca_transform(train,test):

    '''
    transform location data using PCA transformation - no dimension reduction
    pca0: latitude
    pca1: longitude
    '''

    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

    pca = PCA().fit(coords)
    train.loc[:,'pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
    train.loc[:,'pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
    train.loc[:,'dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    train.loc[:,'dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    test.loc[:,'pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
    test.loc[:,'pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
    test.loc[:,'dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    test.loc[:,'dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

    return train,test


def location_cluster(train,test):
    # the input data need to be cleaned
    '''
    INPUT:
        cleaned data frame and PCA transformed location data
    OUTPUT:
        extracted features of locations using K-means clustering
    Explaination:
        The features of locations are dummy variables using OneHotEncoder
    '''

    # merge pick up longitude and drop off longitude
    # merge pick up latitude and drop off latitude

    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))



    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])



    train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
    train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
    test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
    test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])


    return train,test


def geospatial(train,test):

    for gby_col in ['pickup_hour', 'pickup_date', 'pickup_dt_bin',
                   'pickup_week_hour', 'pickup_cluster', 'dropoff_cluster']:
        gby = train.groupby(gby_col).mean()[['avg_speed_h', 'avg_speed_m', 'log_trip_duration']]
        gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
        train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)
        test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)

    for gby_cols in [['center_lat_bin', 'center_long_bin'],
                     ['pickup_hour', 'center_lat_bin', 'center_long_bin'],
                     ['pickup_hour', 'pickup_cluster'],  ['pickup_hour', 'dropoff_cluster'],
                     ['pickup_cluster', 'dropoff_cluster']]:
        coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
        coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
        coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
        coord_stats = coord_stats[coord_stats['id'] > 100]
        coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' %  '_'.join(gby_cols)]
        train = pd.merge(train, coord_stats, how='left', on=gby_cols)
        test = pd.merge(test, coord_stats, how='left', on=gby_cols)

    group_freq = '60min'
    df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
    train.loc[:, 'pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
    test.loc[:, 'pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

    # Count trips over 60min
    df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
    df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']
    train = train.merge(df_counts, on='id', how='left')
    test = test.merge(df_counts, on='id', how='left')

    # Count how many trips are going to each cluster over time
    dropoff_counts = df_all \
        .set_index('pickup_datetime') \
        .groupby([pd.Grouper(freq=group_freq), 'dropoff_cluster']) \
        .agg({'id': 'count'}) \
        .reset_index().set_index('pickup_datetime') \
        .groupby('dropoff_cluster').rolling('240min').mean() \
        .drop('dropoff_cluster', axis=1) \
        .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
        .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})

    train['dropoff_cluster_count'] = train[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)
    test['dropoff_cluster_count'] = test[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)

    return train,test


def osmr_feature(train,file):
    osrm0 = file[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
    train = train.merge(osrm0, left_on='id', right_on='id', how='inner')
    return train

def weather(train,file):
    weather = file[['date', 'precipitation', 'snow fall']].values
    weather_feature = np.empty([366, 3], dtype='object')
    for i in range(366):
        date = weather[i, 0].split('-')
        dt = datetime.date(int(date[2]), int(date[1]), int(date[0]))
        weather_feature[i, 0] = dt.strftime('%Y-%m-%d')
        # replace 'T' with 0.1
        if weather[i, 1] == 'T':
            weather[i, 1] = 0.1
        if weather[i, 2] == 'T':
            weather[i, 2] = 0.1
        # if has rain or snow
        if float(weather[i, 1]) == 0:
            weather_feature[i, 1] = 0
        else:
            weather_feature[i, 1] = 1
        if float(weather[i, 2]) == 0:
            weather_feature[i, 2] = 0
        else:
            weather_feature[i, 2] = 1

    weather_pd = pd.DataFrame(weather_feature, columns=['pickup_date', 'if_rain', 'if_snow'])
    train['pickup_date']=train['pickup_date'].astype(str)
    weather_pd['pickup_date']=weather_pd['pickup_date'].astype(str)
    weather_pd['average temperature']=file['average temperature']
    train = train.merge(weather_pd, left_on='pickup_date', right_on='pickup_date', how='left')
    train[['if_rain','if_snow']]=train[['if_rain','if_snow']].astype(int)
    return train


def noisy_data(train,file):

    train = train.merge(file, left_on='id', right_on='id', how='inner')

    return train


traindata = pd.read_csv("C:\\Users\\zur74\\OneDrive\\project-557\\train.csv")
testdata = pd.read_csv("C:\\Users\\zur74\\OneDrive\\project-557\\test.csv")
weather_file = pd.read_csv('C:\\Users\\zur74\\OneDrive\\project-557\\whether\\weather_data_nyc_centralpark_2016.csv')
osmr1 = pd.read_csv('C:\\Users\\zur74\\OneDrive\\project-557\\OSRM\\fastest_routes_train_part_1.csv')
osmr2 = pd.read_csv('C:\\Users\\zur74\\OneDrive\\project-557\\OSRM\\fastest_routes_train_part_2.csv')
osmr_train = pd.concat((osmr1,osmr2))
osmr_test = pd.read_csv(
        'C:\\Users\\zur74\\OneDrive\\project-557\\OSRM\\fastest_routes_test.csv')

noisy_train=pd.read_csv('C:\\Users\\zur74\\OneDrive\\project-557\\train_augmented.csv')
noisy_test = pd.read_csv('C:\\Users\\zur74\\OneDrive\\project-557\\test_augmented.csv')

traindata = pd.read_csv("C:\\Users\\zur74\\OneDrive\\project-557\\train.csv")
testdata = pd.read_csv("C:\\Users\\zur74\\OneDrive\\project-557\\test.csv")



# feature extraction
train = date_info(traindata)
test = date_info(testdata)


train = preprocess(traindata)
test = preprocess(testdata)

train = distance_haversine_man_bearing(train)
test = distance_haversine_man_bearing(test)

train,test = location_bin(train,test)

train = speed(train)
train = clean(train)

train,test = pca_transform(train,test)
train,test = location_cluster(train,test)
train,test = geospatial(train,test)

train = osmr_feature(train,osmr_train)
test = osmr_feature(test,osmr_test)


train = weather(train,weather_file)
test = weather(test,weather_file)

train = noisy_data(train,noisy_train)
test = noisy_data(test,noisy_test)


t1 = dt.datetime.now()
print('Feature extraction time: %i seconds' % (t1 - t0).seconds)

feature_names_train = list(train.columns)
feature_names_test = list(test.columns)
invalid = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime',
                           'trip_duration', 'check_trip_duration',
                           'pickup_date', 'avg_speed_h', 'avg_speed_m',
                           'pickup_lat_bin', 'pickup_long_bin',
                           'center_lat_bin', 'center_long_bin',
                           'pickup_dt_bin', 'pickup_datetime_group','motorway']

feature_names = [f for f in train.columns if f not in invalid]
print(train[feature_names])
x = train[feature_names].values
y = np.log(train['trip_duration'].values + 1)
predict = test[feature_names].values



########################## parameter tuning ##################################
t2 = dt.datetime.now()
xgb_pars = []
for MCW in [10, 20, 50, 75, 100]:
    for ETA in [0.05, 0.1, 0.15]:
        for CS in [0.3, 0.4, 0.5]:
            for MD in [6, 8, 10, 12, 15]:
                for SS in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    for LAMBDA in [0.5, 1., 1.5,  2., 3.]:
                        xgb_pars.append({'min_child_weight': MCW, 'eta': ETA, 
                                         'colsample_bytree': CS, 'max_depth': MD,
                                         'subsample': SS, 'lambda': LAMBDA, 
                                         'nthread': -1, 'booster' : 'gbtree', 'eval_metric': 'rmse',
                                         'silent': 1, 'objective': 'reg:linear'})

    
def train_xgboost(xgb_pars,x_train,x_valid):
    
    parameter_index = []   
    score = [] 
    
    dtrain = xgb.DMatrix(x_train,label=y_train)
    dvalid = xgb.DMatrix(x_test,label=y_test)
    
    for i in range(100):
        current_index = int(np.random.choice(len(xgb_pars), 1))
        xgb_par = xgb_pars[current_index]
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        model = xgb.train(xgb_par, dtrain, 60, watchlist, early_stopping_rounds=50,
                          maximize=False, verbose_eval=10)
        
        
        parameter_index.append(current_index)
        score.append(model.best_score)
    
    return parameter_index,score

def max_index(parameter_sum,score_sum):
    score_list1 = score_sum[0]
    score_list2 = score_sum[1]
    
    m1 = min(score_list1)
    best_index1 = [i for i, j in enumerate(score_list1) if j == m1]
    
    m2 = min(score_list2)
    best_index2 = [i for i, j in enumerate(score_list2) if j == m2]
    
    best_par1_index = parameter_sum[0][best_index1[0]]
    best_par2_index = parameter_sum[1][best_index2[0]]
    
    best_par1 = xgb_pars[best_par1_index]
    best_par2 = xgb_pars[best_par2_index]
    
    return best_par1,best_par2


# 3 folds cross validation    
k_fold = KFold(n_splits = 3)
parameter_sum = []
score_sum = []
for train_index, test_index in k_fold.split(x,y):
    x_train,y_train = x[train_index],y[train_index]
    x_test,y_test = x[test_index],y[test_index]

    parameter_one,score_one = train_xgboost(xgb_pars,x_train,x_test)

    parameter_sum.append(parameter_one)
    score_sum.append(score_one)


par1,par2 = max_index(parameter_sum,score_sum)

# here is a copy of par1 and par2 in case of losing in re-run
good_1 = {'min_child_weight': 75, 'eta': 0.15, 'colsample_bytree': 0.5, 'max_depth': 15,
            'subsample': 0.9, 'lambda': 1.5, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

good_2 = {'min_child_weight': 20, 'eta': 0.15, 'colsample_bytree': 0.3, 'max_depth': 15,
            'subsample': 0.9, 'lambda': 2, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}


t3 = dt.datetime.now()
print('Parameter tuning time: %i seconds' % (t3 - t2).seconds)



from sklearn.model_selection import train_test_split
Xtr, Xv, ytr, yv = train_test_split(x, y, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(predict)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


xgb_pars1 = par1
xgb_pars2 = par2

model = xgb.train(xgb_pars1, dtrain, 100, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=10)

print(model.best_score)


model = xgb.train(good_2, dtrain, 60, watchlist, early_stopping_rounds=100,
                  maximize=False, verbose_eval=10)
print(model.best_score)

# feature importance analysis
feature_importance_dict = model.get_fscore()
fs = ['f%i' % i for i in range(len(feature_names))]
f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
                   'importance': list(feature_importance_dict.values())})

f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})

feature_importance = pd.merge(f1, f2, how='right', on='f')
feature_importance = feature_importance.fillna(0)

list_importance = feature_importance[['feature_name', 'importance']].sort_values(by='importance', ascending=False)

# model prediction     par2 is better
ytest = model.predict(dtest)
test1 = test.copy()
test1['trip_duration'] = np.exp(ytest) - 1
test1[['id', 'trip_duration']].to_csv('submission2.csv', index=False)






# this part is just for some visulizations and have no relationship with training
'''
Data Fields
id - a unique identifier for each trip
vendor_id - a code indicating the provider associated with the trip record
pickup_datetime - date and time when the meter was engaged
dropoff_datetime - date and time when the meter was disengaged
passenger_count - the number of passengers in the vehicle (driver entered value)
pickup_longitude - the longitude where the meter was engaged
pickup_latitude - the latitude where the meter was engaged
dropoff_longitude - the longitude where the meter was disengaged
dropoff_latitude - the latitude where the meter was disengaged
store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
trip_duration - duration of the trip in seconds





train = clean(train)

# data visulization - for more understanding on different variables distributions
print(train.dtypes)
print(train.head())
conclusion = train.describe()

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])

# try to visulize the result
plt.scatter(train['pickup_pca0'], train['dropoff_pca0'])
plt.xlabel('pick up longitude')
plt.ylabel('drop off longitude')
plt.title('train data longitude')
plt.show()

plt.scatter(train['pickup_latitude'], train['dropoff_latitude'])
plt.xlabel('pick up latitude')
plt.ylabel('drop off latitude')
plt.title('train data latitude')
plt.show()


plt.scatter(test['pickup_longitude'], test['dropoff_longitude'])
plt.xlabel('pick up longitude')
plt.ylabel('drop off longitude')
plt.title('test data longitude')
plt.show()

plt.scatter(test['pickup_latitude'], test['dropoff_latitude'])
plt.xlabel('pick up latitude')
plt.ylabel('drop off latitude')
plt.title('test data latitude')
plt.show()

plt.scatter(train['trip_duration'],train['pickup_latitude'])
plt.xlabel('Trip Duration')
plt.ylabel('pick up latitude')
plt.title('trip duration versus pick up latitude')
plt.show()

# check the duration time and geographic location
piece = train[['pickup_longitude','dropoff_longitude','pickup_latitude','dropoff_latitude','trip_duration']].values
high_longitude_duration = piece[piece[:,0]<-100,:]
high_latitude_duration = piece[piece[:,2]>50,:]
high_duration = piece[piece[:,4]>500000,:]
'''
