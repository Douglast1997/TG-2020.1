import sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import svm
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
def SVR_TCC(X_train, y_train, X_test):

    X = X_train
    y = y_train
    regr = svm.SVR(C=64, cache_size=200, coef0=0.1, degree=3, epsilon=0.1, gamma=0.6, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    regr.fit(X, y.values.ravel())
    y_pred = regr.predict(X_test)
    
    return y_pred

#@profile
def y_pred_SVR(X_train, y_train_rssi_1_1, y_train_rssi_1_2, y_train_rssi_1_3, y_train_rssi_2_1, y_train_rssi_2_2, y_train_rssi_2_3, y_train_rssi_3_1, y_train_rssi_3_2, y_train_rssi_3_3, X_test):
     
    y_pred_rssi_1_1 = SVR_TCC(X_train, y_train_rssi_1_1, X_test)
    y_pred_rssi_1_2 = SVR_TCC(X_train, y_train_rssi_1_2, X_test)
    y_pred_rssi_1_3 = SVR_TCC(X_train, y_train_rssi_1_3, X_test)
    
    y_pred_rssi_2_1 = SVR_TCC(X_train, y_train_rssi_2_1, X_test)
    y_pred_rssi_2_2 = SVR_TCC(X_train, y_train_rssi_2_2, X_test)
    y_pred_rssi_2_3 = SVR_TCC(X_train, y_train_rssi_2_3, X_test)
    
    y_pred_rssi_3_1 = SVR_TCC(X_train, y_train_rssi_3_1, X_test)
    y_pred_rssi_3_2 = SVR_TCC(X_train, y_train_rssi_3_2, X_test)
    y_pred_rssi_3_3 = SVR_TCC(X_train, y_train_rssi_3_3, X_test)
    
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

#import time
df_meds_CDB = pd.read_csv("./CDB/CDB_20.csv")
df_meds_CDB = df_meds_CDB.drop("lat", axis = 1)
df_meds_CDB = df_meds_CDB.drop("lon", axis = 1)
#tracemalloc.start()
#T_Init = time.process_time() 
df_meds_CDB = y_pred_SVR(X_train, y_train_rssi_1_1, y_train_rssi_1_2, y_train_rssi_1_3, y_train_rssi_2_1, y_train_rssi_2_2, y_train_rssi_2_3, y_train_rssi_3_1, y_train_rssi_3_2, y_train_rssi_3_3, df_meds_CDB)
#T_Finish = time.process_time()
#T_Train = T_Finish - T_Init
df_meds_CDB.to_csv("./CDB/CDB_20_SVR.csv", index=False)
#df_meds_CDB.head()

df_meds_CDB = pd.read_csv("./CDB/CDB_20.csv")
df_meds_CDB_SVR = pd.read_csv("./CDB/CDB_20_SVR.csv")
df_meds_CDB_SVR['delay_1'] = df_meds_CDB['delay_1']
df_meds_CDB_SVR['delay_2'] = df_meds_CDB['delay_2']
df_meds_CDB_SVR['delay_3'] = df_meds_CDB['delay_3']
df_meds_CDB_SVR['delay_12'] = df_meds_CDB['delay_12']
df_meds_CDB_SVR['delay_13'] = df_meds_CDB['delay_13']
df_meds_CDB_SVR['delay_23'] = df_meds_CDB['delay_23']
df_meds_CDB_SVR['lat'] = df_meds_CDB['lat']
df_meds_CDB_SVR['lon'] = df_meds_CDB['lon']
df_meds_CDB_SVR.to_csv("./CDB/CDB_20_SVR_Complete.csv", index=False)
#df_meds_CDB_SVR.head()
#current, peak = tracemalloc.get_traced_memory()
#print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
#tracemalloc.stop()

print('fin')