import sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
from memory_profiler import profile

def y_RSSIs(file):
    y = pd.read_csv(file)
    y_rssi_1_1 = y.drop(["rssi_1_2", "rssi_1_3", "rssi_2_1", "rssi_2_2", "rssi_2_3", "rssi_3_1", "rssi_3_2", "rssi_3_3"], axis = 1)
    y_rssi_1_2 = y.drop(["rssi_1_1","rssi_1_3", "rssi_2_1", "rssi_2_2", "rssi_2_3", "rssi_3_1", "rssi_3_2", "rssi_3_3"], axis = 1)
    y_rssi_1_3 = y.drop(["rssi_1_1","rssi_1_2", "rssi_2_1", "rssi_2_2", "rssi_2_3", "rssi_3_1", "rssi_3_2", "rssi_3_3"], axis = 1)
    y_rssi_2_1 = y.drop(["rssi_1_1","rssi_1_2", "rssi_1_3", "rssi_2_2", "rssi_2_3", "rssi_3_1", "rssi_3_2", "rssi_3_3"], axis = 1)
    y_rssi_2_2 = y.drop(["rssi_1_1","rssi_1_2", "rssi_1_3", "rssi_2_1", "rssi_2_3", "rssi_3_1", "rssi_3_2", "rssi_3_3"], axis = 1)
    y_rssi_2_3 = y.drop(["rssi_1_1","rssi_1_2", "rssi_1_3", "rssi_2_1", "rssi_2_2", "rssi_3_1", "rssi_3_2", "rssi_3_3"], axis = 1)
    y_rssi_3_1 = y.drop(["rssi_1_1","rssi_1_2", "rssi_1_3", "rssi_2_1", "rssi_2_2", "rssi_2_3", "rssi_3_2", "rssi_3_3"], axis = 1)
    y_rssi_3_2 = y.drop(["rssi_1_1","rssi_1_2", "rssi_1_3", "rssi_2_1", "rssi_2_2", "rssi_2_3", "rssi_3_1", "rssi_3_3"], axis = 1)
    y_rssi_3_3 = y.drop(["rssi_1_1","rssi_1_2", "rssi_1_3", "rssi_2_1", "rssi_2_2", "rssi_2_3", "rssi_3_1", "rssi_3_2"], axis = 1)
    
    return y_rssi_1_1, y_rssi_1_2, y_rssi_1_3, y_rssi_2_1, y_rssi_2_2, y_rssi_2_3, y_rssi_3_1, y_rssi_3_2, y_rssi_3_3

@profile
def LightGBM_TCC(X_train, y_train, X_test):

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    d_train = lgb.Dataset(X_train, label=y_train)
    #d_train
    params = {}
    params['learning_rate'] = 0.005 #0.003
    params['boosting_type'] = 'goss'
    params['metric'] = 'rmse' 
    params['max_depth'] = 7 #ok
    params['num_leaves'] = 32 #ok
    params['min_data_in_leaf'] = 101 #ok
    params['feature_fraction'] = 1.0 # ok
    params['lambda_l1'] = 0.001 #ok
    params['lambda_l2'] = 8 #ok
    params['min_split_gain'] = 0.4 #
    params['min_child_weight'] = 80 #52
    params['nthreads'] = 2 #ok
    params['top_rate'] = 0.90 #ok
    params['other_rate'] = 0.07 # ok
    
    clf = lgb.train(
        params,
        d_train,
        num_boost_round=1000,
        verbose_eval=100)
    
    #Prediction
    y_pred=clf.predict(X_test)

    return y_pred

#@profile
def y_pred_write_File(X_train, y_train_rssi_1_1, y_train_rssi_1_2, y_train_rssi_1_3, y_train_rssi_2_1, y_train_rssi_2_2, y_train_rssi_2_3, y_train_rssi_3_1, y_train_rssi_3_2, y_train_rssi_3_3, X_test, Metodo_Num):
    #y_train_lat, y_train_lon#
     
    y_pred_rssi_1_1 = LightGBM_TCC(X_train, y_train_rssi_1_1, X_test)
    y_pred_rssi_1_2 = LightGBM_TCC(X_train, y_train_rssi_1_2, X_test)
    y_pred_rssi_1_3 = LightGBM_TCC(X_train, y_train_rssi_1_3, X_test)
    
    y_pred_rssi_2_1 = LightGBM_TCC(X_train, y_train_rssi_2_1, X_test)
    y_pred_rssi_2_2 = LightGBM_TCC(X_train, y_train_rssi_2_2, X_test)
    y_pred_rssi_2_3 = LightGBM_TCC(X_train, y_train_rssi_2_3, X_test)
    
    y_pred_rssi_3_1 = LightGBM_TCC(X_train, y_train_rssi_3_1, X_test)
    y_pred_rssi_3_2 = LightGBM_TCC(X_train, y_train_rssi_3_2, X_test)
    y_pred_rssi_3_3 = LightGBM_TCC(X_train, y_train_rssi_3_3, X_test)
    
    
    y_pred_rssi_1_1 = pd.DataFrame({'rssi_1_1':  y_pred_rssi_1_1})
    y_pred_rssi_1_2 = pd.DataFrame({'rssi_1_2':  y_pred_rssi_1_2})
    y_pred_rssi_1_3 = pd.DataFrame({'rssi_1_3':  y_pred_rssi_1_3})
    
    y_pred_rssi_2_1 = pd.DataFrame({'rssi_2_1':  y_pred_rssi_2_1})
    y_pred_rssi_2_2 = pd.DataFrame({'rssi_2_2':  y_pred_rssi_2_2})
    y_pred_rssi_2_3 = pd.DataFrame({'rssi_2_3':  y_pred_rssi_2_3})
    
    y_pred_rssi_3_1 = pd.DataFrame({'rssi_3_1':  y_pred_rssi_3_1})
    y_pred_rssi_3_2 = pd.DataFrame({'rssi_3_2':  y_pred_rssi_3_2})
    y_pred_rssi_3_3 = pd.DataFrame({'rssi_3_3':  y_pred_rssi_3_3})
    
    y_pred = pd.concat([y_pred_rssi_1_1, y_pred_rssi_1_2, y_pred_rssi_1_3, y_pred_rssi_2_1, y_pred_rssi_2_2, y_pred_rssi_2_3, y_pred_rssi_3_1, y_pred_rssi_3_2, y_pred_rssi_3_3], axis=1)
  
    return y_pred

#y_train lat and Lon Outdoor 90%
#y_train_lat, y_train_lon 
y_train_rssi_1_1, y_train_rssi_1_2, y_train_rssi_1_3, y_train_rssi_2_1, y_train_rssi_2_2, y_train_rssi_2_3, y_train_rssi_3_1, y_train_rssi_3_2, y_train_rssi_3_3 = y_RSSIs("DatabasesTCC/Outdoor_y_train_90%.csv")
y_train_rssi_1_1.head()


#y_train lat and Lon Outdoor 90%
#y_train_lat, y_train_lon 
y_test_rssi_1_1, y_test_rssi_1_2, y_test_rssi_1_3, y_test_rssi_2_1, y_test_rssi_2_2, y_test_rssi_2_3, y_test_rssi_3_1, y_test_rssi_3_2, y_test_rssi_3_3 = y_RSSIs("DatabasesTCC/Outdoor_y_test_10%.csv")
y_test_rssi_1_1.head()


X_train = pd.read_csv("databasesTCC/Outdoor_X_train_90%.csv")
X_train = X_train.drop("lat", axis = 1)
X_train = X_train.drop("lon", axis = 1)
X_train = X_train.drop("idx", axis = 1)
X_train.head()

y_train = pd.read_csv("databasesTCC/Outdoor_y_train_90%.csv")
y_train.head()

X_test = pd.read_csv("databasesTCC/Outdoor_X_test_10%.csv")
X_test = X_test.drop("lat", axis = 1)
X_test = X_test.drop("lon", axis = 1)
X_test = X_test.drop("idx", axis = 1)
X_test.head()


y_test = pd.read_csv("databasesTCC/Outdoor_y_test_10%.csv")
y_test.head()

X_test = pd.read_csv("databasesTCC/Outdoor_X_test_10%.csv")
X_test = X_test.drop("lat", axis = 1)
X_test = X_test.drop("lon", axis = 1)
X_test = X_test.drop("idx", axis = 1)
X_test.head()


# X_test Outdoor X_test 10%
X_test = pd.read_csv("databasesTCC/Outdoor_X_test_10%.csv")
X_test = X_test.drop("lat", axis = 1)
X_test = X_test.drop("lon", axis = 1)
idx_test = X_test["idx"]
X_test = X_test.drop("idx", axis = 1)
X_test.head()

print('init')
df_meds_CDB = pd.read_csv("./CDB/CDB_20.csv")#20
df_meds_CDB = df_meds_CDB.drop("lat", axis = 1)
df_meds_CDB = df_meds_CDB.drop("lon", axis = 1)
#tracemalloc.start()
#T_Init = time.process_time() 
df_meds_CDB = y_pred_write_File(X_train, y_train_rssi_1_1, y_train_rssi_1_2, y_train_rssi_1_3, y_train_rssi_2_1, y_train_rssi_2_2, y_train_rssi_2_3, y_train_rssi_3_1, y_train_rssi_3_2, y_train_rssi_3_3, df_meds_CDB, 2)
#T_Finish = time.process_time()
#T_Train = T_Finish - T_Init
df_meds_CDB.to_csv("./CDB/CDB_20_LightGBM.csv", index=False)
#df_meds_CDB.head()


df_meds_CDB = pd.read_csv("./CDB/CDB_20.csv")
df_meds_CDB_LightGBM = pd.read_csv("./CDB/CDB_20_LightGBM.csv")
df_meds_CDB_LightGBM['delay_1'] = df_meds_CDB['delay_1']
df_meds_CDB_LightGBM['delay_2'] = df_meds_CDB['delay_2']
df_meds_CDB_LightGBM['delay_3'] = df_meds_CDB['delay_3']
df_meds_CDB_LightGBM['delay_12'] = df_meds_CDB['delay_12']
df_meds_CDB_LightGBM['delay_13'] = df_meds_CDB['delay_13']
df_meds_CDB_LightGBM['delay_23'] = df_meds_CDB['delay_23']
df_meds_CDB_LightGBM['lat'] = df_meds_CDB['lat']
df_meds_CDB_LightGBM['lon'] = df_meds_CDB['lon']
df_meds_CDB_LightGBM.to_csv("./CDB/CDB_20_LightGBM_Complete.csv", index=False)
#df_meds_CDB_LightGBM.head()
#current, peak = tracemalloc.get_traced_memory()
#print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
#tracemalloc.stop()
print('fin')