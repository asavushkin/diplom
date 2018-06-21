
# coding: utf-8

# In[1117]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import genfromtxt
my_data = genfromtxt('/Users/admin/Desktop/apple.csv', delimiter=',')
#делаем первый chain data_1
minim_1 = []
maxim_1 =[]
for row in my_data:
    minim_1.append(row[1])
    maxim_1.append(row[0])
del(minim_1[0])
del(maxim_1[0])
month_minim_1_1 = []
month_maxim_1_1 = []

month_minim_1_2 = []
month_maxim_1_2 = []
for min_price_1 in range(0,5): 
    month_minim_1_1.append(minim_1[min_price_1] +5+ np.random.normal(0, 0.1))
for max_price_1 in range(0,5): 
    month_maxim_1_1.append(maxim_1[max_price_1] +5+ np.random.normal(0, 0.1))
for min_price_1 in range(0,5): 
    month_minim_1_2.append(minim_1[min_price_1] +5+ np.random.normal(0, 0.1))
for max_price_1 in range(0,5): 
    month_maxim_1_2.append(maxim_1[max_price_1] +5+ np.random.normal(0, 0.1))
timestamps_1  = []
for timestamp in range (0,5):
    timestamps_1.append(timestamp)
timestamps_1 = np.array(timestamps_1)

timestamps_2  = []
for timestamp in range (5,10):
    timestamps_2.append(timestamp)
timestamps_2  = np.array(timestamps_2)
chain1_1 = np.column_stack((month_minim_1_1,month_maxim_1_1))
chain2_2 = np.column_stack((month_minim_1_2,month_maxim_1_2))
def sim_chains(chain1, chain2):
    zero_one_chain1 = []
    zero_one_chain2 = []
    minim_chain1 = chain1[0]
    maxim_chain1 = chain1[1]
    minim_chain2 = chain2[0]
    maxim_chain2 = chain2[1]  
    for i in range(len(minim_chain1)-1):
        if minim_chain1[i+1] > minim_chain1[i]:
            zero_one_chain1.append(1)
        else:
            zero_one_chain1.append(0)
        if maxim_chain1[i+1] > maxim_chain1[i]:
            zero_one_chain1.append(1)
        else:
            zero_one_chain1.append(0)
    print zero_one_chain1   
    for i in range(len(minim_chain2)-1):
        if minim_chain2[i+1] > minim_chain2[i]:
            zero_one_chain2.append(1)
        else:
            zero_one_chain2.append(0)
        if maxim_chain2[i+1] > maxim_chain2[i]:
            zero_one_chain2.append(1)
        else:
            zero_one_chain2.append(0)
    print zero_one_chain2
    
    simm = 0
    for i in range(2*len(minim_chain2)-3): 
        if zero_one_chain1[i] == zero_one_chain2[i] and zero_one_chain1[i+1] == zero_one_chain2[i+1]:
            simm = simm + 1
    simmilarity = simm/1.0/(len(minim_chain2)-1)
min_angles = []
max_angles = []
def angle(chain1):
    angles = []
    for i in range(len(chain1[0])-1):
        min_v_2 = chain1[0][i+1]-chain1[0][i]        
        angles.append(np.arctan2(min_v_2,0.5)*180/np.pi)
        max_v_2 = chain1[1][i+1]-chain1[1][i]
        angles.append(np.arctan2(max_v_2,0.5)*180/np.pi)
    return angles
def angle_simm(chain1, chain2):
    simm = 0
    angles1 = angle(chain1)
    angles2 = angle(chain2)
    for i in range(len(angles1)):
        if np.sign(angles1[i])==np.sign(angles2[i]):
            simm = simm + abs(abs(angles1[i])-abs(angles2[i]))
        if np.sign(angles1[i])!=np.sign(angles2[i]):
            simm = simm + abs(angles1[i])+abs(angles2[i])
    return simm/1.0/len(angles1)


# In[1118]:


def sim_chains(chain1, chain2):
    zero_one_chain1 = []
    zero_one_chain2 = []
    minim_chain1 = chain1[0]
    maxim_chain1 = chain1[1]
    minim_chain2 = chain2[0]
    maxim_chain2 = chain2[1]
    for i in range(len(minim_chain1)-1):
        if minim_chain1[i+1] > minim_chain1[i]:
            zero_one_chain1.append(1)
        else:
            zero_one_chain1.append(0)
        if maxim_chain1[i+1] > maxim_chain1[i]:
            zero_one_chain1.append(1)
        else:
            zero_one_chain1.append(0)
    print zero_one_chain1   
    for i in range(len(minim_chain2)-1):
        if minim_chain2[i+1] > minim_chain2[i]:
            zero_one_chain2.append(1)
        else:
            zero_one_chain2.append(0)
        if maxim_chain2[i+1] > maxim_chain2[i]:
            zero_one_chain2.append(1)
        else:
            zero_one_chain2.append(0)
    print zero_one_chain2
    
    simm = 0
    for i in range(2*len(minim_chain2)-3): 
        if zero_one_chain1[i] == zero_one_chain2[i] and zero_one_chain1[i+1] == zero_one_chain2[i+1]:
            simm = simm + 1
    simmilarity = simm/1.0/(len(minim_chain2)-1)
min_angles = []
max_angles = []

def angle_simm(chain1, chain2):
    simm = 0
    angles1 = angle(chain1)
    angles2 = angle(chain2)
    for i in range(len(angles1)):
        if np.sign(angles1[i])==np.sign(angles2[i]):
            simm = simm + abs(abs(angles1[i])-abs(angles2[i]))
        if np.sign(angles1[i])!=np.sign(angles2[i]):
            simm = simm + abs(angles1[i])+abs(angles2[i])
    return simm/1.0/len(angles1)


# In[461]:


#/Users/admin/Desktop/bits.csv 
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd


def p_matrix(path,num_obs):
    my_data = genfromtxt(path, delimiter=',')
    #print my_data
#делаем первый chain data_1
    minim_1 = []
    maxim_1 =[]
    len_data=0
    for row in my_data:
        len_data=len_data+1
        minim_1.append(row[1])
        maxim_1.append(row[0])
    del(minim_1[0])
    del(maxim_1[0])
    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    stack_month=[]
    size = 0
    for j in range(len_data-num_obs):
        if (j%num_obs) == 0:
            size = size + 1
            month_minim_1 = []
            month_maxim_1 = []
            data_1 = []
            timestamps_1  = []
            for min_price_1 in range(j,num_obs+j):    
                month_minim_1.append(minim_1[min_price_1])
            
            for max_price_1 in range(j,num_obs+j):
                month_maxim_1.append(maxim_1[max_price_1])
            data_1 = np.vstack((month_minim_1,month_maxim_1))
            stack_month.append(data_1)
    stack_month = np.array(stack_month)
    P = np.zeros((size, size))
    for i in range(size):
        for j in range(i,size):
            P[i,j] = angle_simm(stack_month[i] ,stack_month[j])
            stack_month[i]
    return P
def stack(path,num_obs):
    my_data = genfromtxt(path, delimiter=',')
    minim_1 = []
    maxim_1 =[]
    len_data=0
    for row in my_data:
        len_data=len_data+1
        minim_1.append(row[1])
        maxim_1.append(row[0])
    del(minim_1[0])
    del(maxim_1[0])

    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    stack_month=[]
    size = 0
    for j in range(len_data-num_obs):
        if (j%num_obs) == 0:
            size = size + 1
            month_minim_1 = []
            month_maxim_1 = []
            data_1 = []
            timestamps_1  = []
            for min_price_1 in range(j,num_obs+j):    
                month_minim_1.append(minim_1[min_price_1])

            for max_price_1 in range(j,num_obs+j):
                month_maxim_1.append(maxim_1[max_price_1])
            data_1 = np.vstack((month_minim_1,month_maxim_1))
            stack_month.append(data_1)
    stack_month = np.array(stack_month)  
    return stack_month


# In[442]:


import pandas as pd
import numpy as np
def p_transfrom():
    p = p_matrix("/Users/admin/Desktop/apple.csv",7)[0]
    W = np.maximum( p, p.transpose())
    return  W/np.max(W) 


# In[ ]:



import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
p=p_transfrom()
db = DBSCAN(eps=0.18, min_samples=5,metric="precomputed")
db.fit(p)
labels = db.labels_
no_clusters = len(set(labels)) - (1 if -1 in labels else 0)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
my_data = genfromtxt('/Users/admin/Desktop/apple.csv', delimiter=',')
#делаем первый chain data_1
minim_1 = []
maxim_1 =[]
for row in my_data:
    minim_1.append(row[1])
    maxim_1.append(row[0])
del(minim_1[0])
del(maxim_1[0])
month_minim_1 = []
month_maxim_1 = []
timestamps_1  = []
for min_price_1 in range(60*7,60*7+7): 
    month_minim_1.append(minim_1[min_price_1])
for max_price_1 in range(60*7,60*7+7): 
    month_maxim_1.append(maxim_1[max_price_1])
month_maxim_1_back =month_maxim_1[::-1]
for timestamp in range (60*7,60*7+7):
    timestamps_1.append(timestamp)
timestamps_back_1 = timestamps_1[::-1]

min_time_1 = np.column_stack([month_minim_1,timestamps_1])
max_time_1 = np.column_stack([month_maxim_1_back,timestamps_back_1]) 
min_time_1 = list(min_time_1)
max_time_1 = list(max_time_1)
data_1 = np.vstack((min_time_1,max_time_1))
plt.plot(timestamps_1, month_minim_1)
plt.plot(timestamps_back_1, month_maxim_1_back)
plt.show()


# In[ ]:


import os  
patterns = []
root = "/Users/admin/Desktop/Новая папка" 

def patern_finder():
    for k, dirs, filenames in os.walk(root):
        for f in filenames:
            if not f.startswith('.') and os.path.isfile(os.path.join(root, f)):
                print f 
                p = p_matrix(os.path.join(root, f),7)
                W = np.maximum(p, p.transpose())
                p = W/np.max(W)
                #print p, " это пэ"
                db = DBSCAN(eps=0.18, min_samples=5,metric="precomputed")
                db.fit(p)
                labels = db.labels_
                no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                for i in range(no_clusters):
                    print 'Cluster : ',i , np.nonzero(labels == i)[0]
                    for j in range(len(np.nonzero(labels == i)[0])):
                        ris(os.path.join(root, f),np.nonzero(labels == i)[0][j])
                        chainn = stack(os.path.join(root, f),7)
                        patterns.append(chainn[j])
    return patterns
result = patern_finder()
            


# In[467]:


np.array(result).shape


# In[468]:


def stack(path,num_obs):
    my_data = genfromtxt(path, delimiter=',')
    #print my_data
#делаем первый chain data_1
    minim_1 = []
    maxim_1 =[]
    len_data=0
    for row in my_data:
        len_data=len_data+1
        minim_1.append(row[1])
        maxim_1.append(row[0])
    del(minim_1[0])
    del(maxim_1[0])

    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    stack_month=[]
    size = 0
    for j in range(len_data-num_obs):
        if (j%num_obs) == 0:
            size = size + 1
            month_minim_1 = []
            month_maxim_1 = []
            data_1 = []
            timestamps_1  = []
            for min_price_1 in range(j,num_obs+j):    
                month_minim_1.append(minim_1[min_price_1])

            for max_price_1 in range(j,num_obs+j):
                month_maxim_1.append(maxim_1[max_price_1])
            data_1 = np.vstack((month_minim_1,month_maxim_1))
            stack_month.append(data_1)
    stack_month = np.array(stack_month)  
    return stack_month


# In[1112]:


def ris(path, place):
    plt.figure(figsize=(3,3))
    my_data = genfromtxt(path, delimiter=',')
    #делаем первый chain data_1
    minim_1 = []
    maxim_1 =[]
    for row in my_data:
        minim_1.append(row[1])
        maxim_1.append(row[0])
    del(minim_1[0])
    del(maxim_1[0])
    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    for min_price_1 in range(place*14,place*14+14): 
        month_minim_1.append(minim_1[min_price_1])
    for max_price_1 in range(place*14,place*14+14): 
        month_maxim_1.append(maxim_1[max_price_1])
    month_maxim_1_back =month_maxim_1[::-1]
    for timestamp in range (place*14,place*14+14):
        timestamps_1.append(timestamp)
    timestamps_back_1 = timestamps_1[::-1]

    min_time_1 = np.column_stack([month_minim_1,timestamps_1])
    max_time_1 = np.column_stack([month_maxim_1_back,timestamps_back_1]) 
    min_time_1 = list(min_time_1)
    max_time_1 = list(max_time_1)
    data_1 = np.vstack((min_time_1,max_time_1))
    plt.plot(timestamps_1, month_minim_1)
    plt.plot(timestamps_back_1, month_maxim_1_back)
    plt.show()
    timestamps_11 = []
    minimm = testt_result1[index][0]
    maximm = testt_result1[index][1]
    maximm_back = maximm[::-1]
    plt.show()


# In[ ]:


import os  
patterns = []
root = "/Users/admin/Desktop/sandp500 3/individual_stocks_5yr" 
num_obs = 6
testt_result1 = result
stack_testt_result1 = []
def divide(window, path,num_obs):
    my_data = genfromtxt(path, delimiter=',')
    minim_1 = []
    maxim_1 =[]
    len_data=0
    for row in my_data:
        len_data=len_data+1
        minim_1.append(row[1])
        maxim_1.append(row[0])
    z = window - num_obs - 1
    del minim_1[:z]
    del maxim_1[:z]
    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    stack_month=[]
    size = 0
    for j in range(len_data - num_obs - window ):
        if (j%num_obs) == 0:
            size = size + 1
            month_minim_1 = []
            month_maxim_1 = []
            data_1 = []
            timestamps_1  = []
            for min_price_1 in range(j,num_obs+j):    
                month_minim_1.append(minim_1[min_price_1])

            for max_price_1 in range(j,num_obs+j):
                month_maxim_1.append(maxim_1[max_price_1])
            data_1 = np.vstack((month_minim_1,month_maxim_1))
            stack_month.append(data_1)
    stack_month = np.array(stack_month)
    return stack_month
for i in range(len(testt_result1)):
    data_1 = np.vstack(( testt_result1[i][0][:-1],testt_result1[i][1][:-1]))
    stack_testt_result1.append(data_1)
all_pattern_mse_results = []
def mse(window):
    for k, dirs, filenames in os.walk(root):
        for f in filenames:
            if not f.startswith('.') and os.path.isfile(os.path.join(root, f)):
                print f
                stack_month = divide(window, os.path.join(root, f),6)
                print np.array(stack_month).shape
                testt_stack_month = divide(window, os.path.join(root, f),7)
                print np.array(testt_stack_month).shape, "testt_stack_month"
                summa_mse = 0
                stock_mse = []
                for i in range(len(stack_month)-2):
                    anglesim_stack = []
                    for j in range(len(testt_result1)-1):
                        anglesim_stack.append(angle_simm(stack_testt_result1[j], stack_month[i]))
                    index = anglesim_stack.index(min(anglesim_stack))
                    msee = (stack_month[i+1][1][0] - (stack_month[i][1][5] + (testt_result1[index][1][6]-testt_result1[index][1][5])))**2 
                    stock_mse.append(msee)
                all_pattern_mse_results.append(stock_mse)
    return all_pattern_mse_results
patern_result_mse63 = mse(63)


# In[ ]:


import pandas as pd
import numpy as np
from pandas import datetime

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as plt
import matplotlib.dates as dates
from matplotlib.pylab import rcParams
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

maxx_ar = 5
maxx_ma = 5

p_values = range(5, 5)
d_values = range(1, 1)
q_values = range(0, 0)
window1 = 63
window2 = 91
window3 = 119
def splitt(dataset, window, num_obs):
    arima_stock_db = []
    for i in range(int((len(dataset)-window)/num_obs)):
        arima_stock_db.append(dataset[num_obs*i:window+num_obs*i])
    return arima_stock_db

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score_AIC, best_cfg_aic = float("inf"), None  
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                model = ARIMA(series, order=(p,d,q)).fit()
                AIC = model.aic
                print AIC
                if AIC < best_score_AIC:
                    best_score_AIC, best_cfg_aic = AIC, order
    model = ARIMA(dataset, (0,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()[0]
    last_element = len(dataset)
    obs = dataset[last_element-1]
    return (output[0]-obs)**2


def parser(x):
    splited = x.split('-')
    return datetime.strptime( splited[0] + "-" + splited[1]+"-" + splited[2], "%Y-%m-%d")

root_1 = "/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr"  
all_mse_results = []
for k, dirs, filenames in os.walk(root):
    for f in filenames: 
            if not f.startswith('.') and os.path.isfile(os.path.join(root_1, f)):
                print f
                dataMaster = pd.read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f)
                sp_500 = dataMaster['high']
                ts = pd.Series(dataMaster['high'].values)
                sp500_diff = ts - ts.shift()
                diff = sp500_diff.dropna()
                series = read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
                splitted_series = splitt(series, 63, num_obs)
                print np.array(splitted_series).shape, "splitted_seriesSHAPE"
                stock_mse = []
                for i in range(len(splitted_series)-1):
                    mse_result1 = evaluate_models(splitted_series[i], p_values, d_values, q_values)
                    stock_mse.append(mse_result1)
                all_mse_results.append(stock_mse)
                
r_1_63  =  pd.DataFrame(all_mse_results)


# In[ ]:


r_1_63  =  pd.DataFrame(all_mse_results)
r_1_63 = r_1_63.transpose()
pat_data_mse_6363 = pd.DataFrame(patern_result_mse63)
pat_data_mse_6363 = pat_data_mse_6363.transpose()
columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
        , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
       'CL_data.csv','CLX_data.csv']
pat_data_mse_6363.columns = columns
r_1_63.columns = columns
mse_compare_1_63 = pat_data_mse_6363 - r_1_63
print mse_compare_1_63[mse_compare_1_63  !=900].count()
print mse_compare_1_63[mse_compare_1_63  < 0].count() 


# In[ ]:


import pandas as pd
import numpy as np
from pandas import datetime

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as plt
import matplotlib.dates as dates
from matplotlib.pylab import rcParams
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

maxx_ar = 5
maxx_ma = 5

p_values = range(5, 5)
d_values = range(1, 1)
q_values = range(0, 0)

window1 = 63
window2 = 91
window3 = 119

def splitt(dataset, window, num_obs):
    arima_stock_db = []
    for i in range(int((len(dataset)-window)/num_obs)):
        #print int((len(dataset)-window)/num_obs), "int((len(dataset)-window)/num_obs)"
        arima_stock_db.append(dataset[num_obs*i:window+num_obs*i])
    return arima_stock_db

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score_AIC, best_cfg_aic = float("inf"), None  
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                model = ARIMA(series, order=(p,d,q)).fit()
                AIC = model.aic
                print AIC
                if AIC < best_score_AIC:
                    best_score_AIC, best_cfg_aic = AIC, order
    model = ARIMA(dataset, (0,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()[0]
    last_element = len(dataset)
    obs = dataset[last_element-1]
    return (output[0]-obs)**2


def parser(x):
    splited = x.split('-')
    return datetime.strptime( splited[0] + "-" + splited[1]+"-" + splited[2], "%Y-%m-%d")

root_1 = "/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr"  
#WINDOW = 63
all_mse_results = []
for k, dirs, filenames in os.walk(root):
    for f in filenames: 
            if not f.startswith('.') and os.path.isfile(os.path.join(root_1, f)):
                print f
                dataMaster = pd.read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f)
                sp_500 = dataMaster['high']
                ts = pd.Series(dataMaster['high'].values)
                sp500_diff = ts - ts.shift()
                diff = sp500_diff.dropna()
                series = read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
                splitted_series = splitt(series, 63, num_obs)
                print np.array(splitted_series).shape, "splitted_seriesSHAPE"
                stock_mse = []
                for i in range(len(splitted_series)-1):
                    mse_result1 = evaluate_models(splitted_series[i], p_values, d_values, q_values)
                    stock_mse.append(mse_result1)
                all_mse_results.append(stock_mse)
                
r =  pd.DataFrame(all_mse_results)


# In[ ]:


#WINDOW = 91


import os  
patterns = []
root = "/Users/admin/Desktop/sandp500 3/individual_stocks_5yr" 

num_obs = 6

testt_result1 = result
stack_testt_result1 = []


def divide(window, path,num_obs):
    my_data = genfromtxt(path, delimiter=',')
    minim_1 = []
    maxim_1 =[]
    len_data=0
    for row in my_data:
        len_data=len_data+1
        minim_1.append(row[1])
        maxim_1.append(row[0])
    z = window - num_obs - 1
    del minim_1[:z]
    del maxim_1[:z]
    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    stack_month=[]
    size = 0
    for j in range(len_data - num_obs - window ):
        if (j%num_obs) == 0:
            size = size + 1
            month_minim_1 = []
            month_maxim_1 = []
            data_1 = []
            timestamps_1  = []
            for min_price_1 in range(j,num_obs+j):    
                month_minim_1.append(minim_1[min_price_1])

            for max_price_1 in range(j,num_obs+j):
                month_maxim_1.append(maxim_1[max_price_1])
            data_1 = np.vstack((month_minim_1,month_maxim_1))
            stack_month.append(data_1)
    stack_month = np.array(stack_month)
    
    return stack_month

for i in range(len(testt_result1)):
    data_1 = np.vstack(( testt_result1[i][0][:-1],testt_result1[i][1][:-1]))
    stack_testt_result1.append(data_1)
all_pattern_mse_results_91 = []
def mse(window):
    for k, dirs, filenames in os.walk(root):
        for f in filenames:
            if not f.startswith('.') and os.path.isfile(os.path.join(root, f)):
                print f
                stack_month = divide(window, os.path.join(root, f),6)
                print np.array(stack_month).shape
                testt_stack_month = divide(window, os.path.join(root, f),7)
                print np.array(testt_stack_month).shape, "testt_stack_month"
                summa_mse = 0
                stock_mse = []
                for i in range(len(stack_month)-2):
                    anglesim_stack = []
                    for j in range(len(testt_result1)-1):
                        anglesim_stack.append(angle_simm(stack_testt_result1[j], stack_month[i]))
                    index = anglesim_stack.index(min(anglesim_stack))
                    msee = (stack_month[i+1][1][0] - (stack_month[i][1][5] + (testt_result1[index][1][6]-testt_result1[index][1][5])))**2 
                    stock_mse.append(msee)
                all_pattern_mse_results_91.append(stock_mse)
    return all_pattern_mse_results_91
                    
              
patern_result_mse_91 = mse(91)


# In[ ]:


#WINDOW 91
import pandas as pd
import numpy as np
from pandas import datetime

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as plt
import matplotlib.dates as dates
from matplotlib.pylab import rcParams
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

maxx_ar = 5
maxx_ma = 5

p_values = range(5, 5)
d_values = range(1, 1)
q_values = range(0, 0)

window1 = 63
window2 = 91
window3 = 119

def splitt(dataset, window, num_obs):
    arima_stock_db = []
    for i in range(int((len(dataset)-window)/num_obs)):
        #print int((len(dataset)-window)/num_obs), "int((len(dataset)-window)/num_obs)"
        arima_stock_db.append(dataset[num_obs*i:window+num_obs*i])
    return arima_stock_db

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score_AIC, best_cfg_aic = float("inf"), None  
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                model = ARIMA(series, order=(p,d,q)).fit()
                AIC = model.aic
                print AIC
                if AIC < best_score_AIC:
                    best_score_AIC, best_cfg_aic = AIC, order
    model = ARIMA(dataset, (0,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()[0]
    last_element = len(dataset)
    obs = dataset[last_element-1]
    return (output[0]-obs)**2


def parser(x):
    splited = x.split('-')
    return datetime.strptime( splited[0] + "-" + splited[1]+"-" + splited[2], "%Y-%m-%d")

root_1 = "/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr"  
#WINDOW = 63
all_mse_results_91 = []
for k, dirs, filenames in os.walk(root):
    for f in filenames: 
            if not f.startswith('.') and os.path.isfile(os.path.join(root_1, f)):
                print f
                dataMaster = pd.read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f)
                sp_500 = dataMaster['high']
                ts = pd.Series(dataMaster['high'].values)
                sp500_diff = ts - ts.shift()
                diff = sp500_diff.dropna()
                series = read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
                splitted_series = splitt(series, 91, num_obs)
                print np.array(splitted_series).shape, "splitted_seriesSHAPE"
                stock_mse = []
                for i in range(len(splitted_series)-1):
                    mse_result1 = evaluate_models(splitted_series[i], p_values, d_values, q_values)
                    stock_mse.append(mse_result1)
                all_mse_results_91.append(stock_mse)
                
r_91 =  pd.DataFrame(all_mse_results_91)


# In[928]:


res_91 = r_91.transpose()
columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
         , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
        'CL_data.csv','CLX_data.csv']
res_91.columns = columns


# In[ ]:


pat_data_mse_91 = pd.DataFrame(patern_result_mse_91)
pat_data_mse_91 = pat_data_mse_91.transpose()
columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
        , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
       'CL_data.csv','CLX_data.csv']
pat_data_mse_91.columns = columns
mse_compare_91 = pat_data_mse_91 - res_91
print "ARIMA window = 63"
print mse_compare_91[mse_compare_91 !=900].count()
print mse_compare_91[mse_compare_91 < 0].count() 


# In[ ]:


#WINDOW = 118
import os  
patterns = []
root = "/Users/admin/Desktop/sandp500 3/individual_stocks_5yr" 

num_obs = 6

testt_result1 = result
stack_testt_result1 = []


def divide(window, path,num_obs):
    my_data = genfromtxt(path, delimiter=',')
    minim_1 = []
    maxim_1 =[]
    len_data=0
    for row in my_data:
        len_data=len_data+1
        minim_1.append(row[1])
        maxim_1.append(row[0])
    z = window - num_obs - 1
    del minim_1[:z]
    del maxim_1[:z]
    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    stack_month=[]
    size = 0
    for j in range(len_data - num_obs - window ):
        if (j%num_obs) == 0:
            size = size + 1
            month_minim_1 = []
            month_maxim_1 = []
            data_1 = []
            timestamps_1  = []
            for min_price_1 in range(j,num_obs+j):    
                month_minim_1.append(minim_1[min_price_1])

            for max_price_1 in range(j,num_obs+j):
                month_maxim_1.append(maxim_1[max_price_1])
            data_1 = np.vstack((month_minim_1,month_maxim_1))
            stack_month.append(data_1)
    stack_month = np.array(stack_month)
    
    return stack_month

for i in range(len(testt_result1)):
    data_1 = np.vstack(( testt_result1[i][0][:-1],testt_result1[i][1][:-1]))
    stack_testt_result1.append(data_1)
all_pattern_mse_results_91 = []
def mse(window):
    for k, dirs, filenames in os.walk(root):
        for f in filenames:
            if not f.startswith('.') and os.path.isfile(os.path.join(root, f)):
                print f
                stack_month = divide(window, os.path.join(root, f),6)
                print np.array(stack_month).shape
                testt_stack_month = divide(window, os.path.join(root, f),7)
                print np.array(testt_stack_month).shape, "testt_stack_month"
                summa_mse = 0
                stock_mse = []
                for i in range(len(stack_month)-2):
                    anglesim_stack = []
                    for j in range(len(testt_result1)-1):
                        anglesim_stack.append(angle_simm(stack_testt_result1[j], stack_month[i]))
                    index = anglesim_stack.index(min(anglesim_stack))
                    msee = (stack_month[i+1][1][0] - (stack_month[i][1][5] + (testt_result1[index][1][6]-testt_result1[index][1][5])))**2 
                    stock_mse.append(msee)
                all_pattern_mse_results_91.append(stock_mse)
    return all_pattern_mse_results_91
                    
              
patern_result_mse_118 = mse(118)


# In[ ]:


#WINDOW 118
import pandas as pd
import numpy as np
from pandas import datetime

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as plt
import matplotlib.dates as dates
from matplotlib.pylab import rcParams
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

maxx_ar = 5
maxx_ma = 5

p_values = range(5, 5)
d_values = range(1, 1)
q_values = range(0, 0)

window1 = 63
window2 = 91
window3 = 119

def splitt(dataset, window, num_obs):
    arima_stock_db = []
    for i in range(int((len(dataset)-window)/num_obs)):
        #print int((len(dataset)-window)/num_obs), "int((len(dataset)-window)/num_obs)"
        arima_stock_db.append(dataset[num_obs*i:window+num_obs*i])
    return arima_stock_db

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score_AIC, best_cfg_aic = float("inf"), None  
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                model = ARIMA(series, order=(p,d,q)).fit()
                AIC = model.aic
                print AIC
                if AIC < best_score_AIC:
                    best_score_AIC, best_cfg_aic = AIC, order
    model = ARIMA(dataset, (0,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()[0]
    last_element = len(dataset)
    obs = dataset[last_element-1]
    return (output[0]-obs)**2


def parser(x):
    splited = x.split('-')
    return datetime.strptime( splited[0] + "-" + splited[1]+"-" + splited[2], "%Y-%m-%d")

root_1 = "/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr"  
all_mse_results_118 = []
for k, dirs, filenames in os.walk(root):
    for f in filenames: 
            if not f.startswith('.') and os.path.isfile(os.path.join(root_1, f)):
                print f
                dataMaster = pd.read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f)
                sp_500 = dataMaster['high']
                ts = pd.Series(dataMaster['high'].values)
                sp500_diff = ts - ts.shift()
                diff = sp500_diff.dropna()
                series = read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
                splitted_series = splitt(series, 118, num_obs)
                print np.array(splitted_series).shape, "splitted_seriesSHAPE"
                stock_mse = []
                for i in range(len(splitted_series)-1):
                    mse_result1 = evaluate_models(splitted_series[i], p_values, d_values, q_values)
                    stock_mse.append(mse_result1)
                all_mse_results_118.append(stock_mse)
                
r_118 =  pd.DataFrame(all_mse_results_118)


# In[933]:


res_118 = r_118.transpose()
columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
         , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
        'CL_data.csv','CLX_data.csv']
res_118.columns = columns


# In[ ]:


pat_data_mse_118 = pd.DataFrame(patern_result_mse_118)
pat_data_mse_118 = pat_data_mse_118.transpose()
columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
        , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
       'CL_data.csv','CLX_data.csv']
pat_data_mse_118.columns = columns
mse_compare_118 = pat_data_mse_118 - res_118
print "ARIMA window = 118"
print mse_compare_118[mse_compare_118 !=900].count()
print mse_compare_118[mse_compare_118 < 0].count() 


# In[ ]:


import os  
patterns = []
root = "/Users/admin/Desktop/Новая папка" 

def patern_finder():
    for k, dirs, filenames in os.walk(root):
        for f in filenames:
            if not f.startswith('.') and os.path.isfile(os.path.join(root, f)):
                print f 
                p = p_matrix(os.path.join(root, f),14)
                W = np.maximum(p, p.transpose())
                p = W/np.max(W)
                #print p, " это пэ"
                db = DBSCAN(eps=0.3, min_samples=3,metric="precomputed")
                db.fit(p)
                labels = db.labels_
                no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                for i in range(no_clusters):
                    print 'Cluster : ',i , np.nonzero(labels == i)[0]
                    for j in range(len(np.nonzero(labels == i)[0])):
                        ris(os.path.join(root, f),np.nonzero(labels == i)[0][j])
                        chainn = stack(os.path.join(root, f),14)
                        patterns.append(chainn[j])
    return patterns
result_14 = patern_finder()


# In[ ]:


import os  
patterns = []
root = "/Users/admin/Desktop/sandp500 3/individual_stocks_5yr" 

num_obs = 7

testt_result_14 = result_14
stack_testt_result_14 = []

def divide(window, path,num_obs):
    my_data = genfromtxt(path, delimiter=',')
    minim_1 = []
    maxim_1 =[]
    len_data=0
    for row in my_data:
        len_data=len_data+1
        minim_1.append(row[1])
        maxim_1.append(row[0])
    z = window - num_obs - 1
    del minim_1[:z]
    del maxim_1[:z]
    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    stack_month=[]
    size = 0
    for j in range(len_data - num_obs - window ):
        if (j%num_obs) == 0:
            size = size + 1
            month_minim_1 = []
            month_maxim_1 = []
            data_1 = []
            timestamps_1  = []
            for min_price_1 in range(j,num_obs+j):    
                month_minim_1.append(minim_1[min_price_1])

            for max_price_1 in range(j,num_obs+j):
                month_maxim_1.append(maxim_1[max_price_1])
            data_1 = np.vstack((month_minim_1,month_maxim_1))
            stack_month.append(data_1)
    stack_month = np.array(stack_month)
    return stack_month

for i in range(len(testt_result_14)):
    data_1 = np.vstack(( testt_result_14[i][0][:-7],testt_result_14[i][1][:-7]))
    stack_testt_result_14.append(data_1)
all_pattern_mse_results = []

def mse(window):
    for k, dirs, filenames in os.walk(root):
        for f in filenames:
            if not f.startswith('.') and os.path.isfile(os.path.join(root, f)):
                print f
                stack_month = divide(window, os.path.join(root, f),7)
                print np.array(stack_month).shape
                testt_stack_month = divide(window, os.path.join(root, f),7)
                print np.array(testt_stack_month).shape, "testt_stack_month"
                summa_mse = 0
                stock_mse = []
                for i in range(len(stack_month)-1):
                    anglesim_stack = []
                    for j in range(len(testt_result1)):
                        anglesim_stack.append(angle_simm(stack_testt_result1[j], stack_month[i]))
                    index = anglesim_stack.index(min(anglesim_stack))
                    testtttt_result_14 = []
                    for k in range(len(testt_result_14[index][1][6:13])):
                        testtttt_result_14.append(stack_month[i][1][6] + (testt_result_14[index][1][6:13][k] - testt_result_14[index][1][5]))                    
                    msee =  mean_squared_error(stack_month[i+1][1], testtttt_result_14)
                    stock_mse.append(msee)
                    print msee 
                all_pattern_mse_results.append(stock_mse)
    return all_pattern_mse_results
                    
#WINDOW = 63               

patern_result_mse = mse(63)


# In[ ]:



import pandas as pd
import numpy as np
from pandas import datetime

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as plt
import matplotlib.dates as dates
from matplotlib.pylab import rcParams
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

maxx_ar = 5
maxx_ma = 5

p_values = range(5, 5)
d_values = range(1, 1)
q_values = range(0, 0)

window1 = 63
window2 = 91
window3 = 119

num_obs = 14

def splitt(dataset, window, num_obs):
    arima_stock_db = []
    for i in range(int((len(dataset)-window)/num_obs)):
        #print int((len(dataset)-window)/num_obs), "int((len(dataset)-window)/num_obs)"
        arima_stock_db.append(dataset[num_obs*i:window+num_obs*i])
    return arima_stock_db

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score_AIC, best_cfg_aic = float("inf"), None  
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                model = ARIMA(series, order=(p,d,q)).fit()
                AIC = model.aic
                print AIC
                if AIC < best_score_AIC:
                    best_score_AIC, best_cfg_aic = AIC, order
    start = len(dataset)-13
    end = len(dataset)
    model = ARIMA(dataset, (0,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=7)[0]
    obs = dataset[start+6:]
    #print len(output[7:]), len(obs)
    return mean_squared_error(output,obs)


def parser(x):
    splited = x.split('-')
    return datetime.strptime( splited[0] + "-" + splited[1]+"-" + splited[2], "%Y-%m-%d")

root_1 = "/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr"  
#WINDOW = 63
all_mse_results_14 = []
for k, dirs, filenames in os.walk(root):
    for f in filenames: 
            if not f.startswith('.') and os.path.isfile(os.path.join(root_1, f)):
                print f
                dataMaster = pd.read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f)
                sp_500 = dataMaster['high']
                ts = pd.Series(dataMaster['high'].values)
                sp500_diff = ts - ts.shift()
                diff = sp500_diff.dropna()
                series = read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
                splitted_series = splitt(series, 63+14, num_obs)
                print np.array(splitted_series).shape, "splitted_seriesSHAPE"
                stock_mse = []
                for i in range(len(splitted_series)-1):
                    mse_result1 = evaluate_models(splitted_series[i], p_values, d_values, q_values)
                    stock_mse.append(mse_result1)
                all_mse_results_14.append(stock_mse)
                
r_14 =  pd.DataFrame(all_mse_results_14)


# In[ ]:


r_14 = r_14.transpose()

columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
         , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
        'CL_data.csv','CLX_data.csv']
r_14.columns = columns
#print r_14
pat_data_mse = pd.DataFrame(patern_result_mse)
pat_data_mse = pat_data_mse.transpose()
columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
        , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
       'CL_data.csv','CLX_data.csv']
pat_data_mse.columns = columns
#если разница положительная тогда mse patternov больше а значит ARIMA предсказала лучше

mse_compare = pat_data_mse - r_14

#print "ARIMA window = 118"
print mse_compare[mse_compare !=900].count()
print mse_compare[mse_compare < 0].count() 


# In[ ]:


import os  
patterns = []
root = "/Users/admin/Desktop/sandp500 3/individual_stocks_5yr" 

num_obs = 7

testt_result_14 = result_14
stack_testt_result_14 = []

def divide(window, path,num_obs):
    my_data = genfromtxt(path, delimiter=',')
    minim_1 = []
    maxim_1 =[]
    len_data=0
    for row in my_data:
        len_data=len_data+1
        minim_1.append(row[1])
        maxim_1.append(row[0])
    z = window - num_obs - 1
    del minim_1[:z]
    del maxim_1[:z]
    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    stack_month=[]
    size = 0
    for j in range(len_data - num_obs - window ):
        if (j%num_obs) == 0:
            size = size + 1
            month_minim_1 = []
            month_maxim_1 = []
            data_1 = []
            timestamps_1  = []
            for min_price_1 in range(j,num_obs+j):    
                month_minim_1.append(minim_1[min_price_1])

            for max_price_1 in range(j,num_obs+j):
                month_maxim_1.append(maxim_1[max_price_1])
            data_1 = np.vstack((month_minim_1,month_maxim_1))
            stack_month.append(data_1)
    stack_month = np.array(stack_month)
    return stack_month

for i in range(len(testt_result_14)):
    data_1 = np.vstack(( testt_result_14[i][0][:-7],testt_result_14[i][1][:-7]))
    stack_testt_result_14.append(data_1)
all_pattern_mse_results = []

def mseee(window):
    for k, dirs, filenames in os.walk(root):
        for f in filenames:
            if not f.startswith('.') and os.path.isfile(os.path.join(root, f)):
                print f
                stack_month = divide(window, os.path.join(root, f),7)
                print np.array(stack_month).shape
                testt_stack_month = divide(window, os.path.join(root, f),7)
                print np.array(testt_stack_month).shape, "testt_stack_month"
                summa_mse = 0
                stock_mse = []
                for i in range(len(stack_month)-1):
                    anglesim_stack = []
                    for j in range(len(testt_result1)):
                        anglesim_stack.append(angle_simm(stack_testt_result1[j], stack_month[i]))
                    index = anglesim_stack.index(min(anglesim_stack))
                    testtttt_result_14 = []
                    for k in range(len(testt_result_14[index][1][6:13])):
                        testtttt_result_14.append(stack_month[i][1][6] + (testt_result_14[index][1][6:13][k] - testt_result_14[index][1][5]))                    
                    msee =  mean_squared_error(stack_month[i+1][1], testtttt_result_14)
                    stock_mse.append(msee)
                    print msee 
                all_pattern_mse_results.append(stock_mse)
    return all_pattern_mse_results
                    
#WINDOW = 63               
patern_result_mse_14_91 = mseee(91)


# In[ ]:


import pandas as pd
import numpy as np
from pandas import datetime

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as plt
import matplotlib.dates as dates
from matplotlib.pylab import rcParams
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

maxx_ar = 5
maxx_ma = 5

p_values = range(5, 5)
d_values = range(1, 1)
q_values = range(0, 0)

window1 = 63
window2 = 91
window3 = 119

num_obs = 14

def splitt(dataset, window, num_obs):
    arima_stock_db = []
    for i in range(int((len(dataset)-window)/num_obs)):
        #print int((len(dataset)-window)/num_obs), "int((len(dataset)-window)/num_obs)"
        arima_stock_db.append(dataset[num_obs*i:window+num_obs*i])
    return arima_stock_db

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score_AIC, best_cfg_aic = float("inf"), None  
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                model = ARIMA(series, order=(p,d,q)).fit()
                AIC = model.aic
                print AIC
                if AIC < best_score_AIC:
                    best_score_AIC, best_cfg_aic = AIC, order
    start = len(dataset)-13
    end = len(dataset)
    model = ARIMA(dataset, (0,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=7)[0]
    obs = dataset[start+6:]
    #print len(output[7:]), len(obs)
    return mean_squared_error(output,obs)


def parser(x):
    splited = x.split('-')
    return datetime.strptime( splited[0] + "-" + splited[1]+"-" + splited[2], "%Y-%m-%d")

root_1 = "/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr"  
#WINDOW = 63
all_mse_results_14 = []
for k, dirs, filenames in os.walk(root):
    for f in filenames: 
            if not f.startswith('.') and os.path.isfile(os.path.join(root_1, f)):
                print f
                dataMaster = pd.read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f)
                sp_500 = dataMaster['high']
                ts = pd.Series(dataMaster['high'].values)
                sp500_diff = ts - ts.shift()
                diff = sp500_diff.dropna()
                series = read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
                splitted_series = splitt(series, 91+14, num_obs)
                print np.array(splitted_series).shape, "splitted_seriesSHAPE"
                stock_mse = []
                for i in range(len(splitted_series)-1):
                    mse_result1 = evaluate_models(splitted_series[i], p_values, d_values, q_values)
                    stock_mse.append(mse_result1)
                all_mse_results_14.append(stock_mse)
                
r_14_91 =  pd.DataFrame(all_mse_results_14)


# In[ ]:


r_14_91 = r_14_91.transpose()

columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
         , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
        'CL_data.csv','CLX_data.csv']
r_14_91.columns = columns
#print r_14
pat_data_mse = pd.DataFrame(patern_result_mse_14_91)
pat_data_mse = pat_data_mse.transpose()
columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
        , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
       'CL_data.csv','CLX_data.csv']
pat_data_mse.columns = columns
#если разница положительная тогда mse patternov больше а значит ARIMA предсказала лучше
mse_compare = pat_data_mse - r_14_91
#print "ARIMA window = 118"
print mse_compare[mse_compare !=900].count()
print mse_compare[mse_compare < 0].count() 


# In[ ]:


import os  
patterns = []
root = "/Users/admin/Desktop/sandp500 3/individual_stocks_5yr" 

num_obs = 7

testt_result_14 = result_14
stack_testt_result_14 = []

def divide(window, path,num_obs):
    my_data = genfromtxt(path, delimiter=',')
    minim_1 = []
    maxim_1 =[]
    len_data=0
    for row in my_data:
        len_data=len_data+1
        minim_1.append(row[1])
        maxim_1.append(row[0])
    z = window - num_obs - 1
    del minim_1[:z]
    del maxim_1[:z]
    month_minim_1 = []
    month_maxim_1 = []
    timestamps_1  = []
    stack_month=[]
    size = 0
    for j in range(len_data - num_obs - window ):
        if (j%num_obs) == 0:
            size = size + 1
            month_minim_1 = []
            month_maxim_1 = []
            data_1 = []
            timestamps_1  = []
            for min_price_1 in range(j,num_obs+j):    
                month_minim_1.append(minim_1[min_price_1])

            for max_price_1 in range(j,num_obs+j):
                month_maxim_1.append(maxim_1[max_price_1])
            data_1 = np.vstack((month_minim_1,month_maxim_1))
            stack_month.append(data_1)
    stack_month = np.array(stack_month)
    return stack_month

for i in range(len(testt_result_14)):
    data_1 = np.vstack(( testt_result_14[i][0][:-7],testt_result_14[i][1][:-7]))
    stack_testt_result_14.append(data_1)
all_pattern_mse_results = []

def mseee(window):
    for k, dirs, filenames in os.walk(root):
        for f in filenames:
            if not f.startswith('.') and os.path.isfile(os.path.join(root, f)):
                print f
                stack_month = divide(window, os.path.join(root, f),7)
                print np.array(stack_month).shape
                testt_stack_month = divide(window, os.path.join(root, f),7)
                print np.array(testt_stack_month).shape, "testt_stack_month"
                summa_mse = 0
                stock_mse = []
                for i in range(len(stack_month)-1):
                    anglesim_stack = []
                    for j in range(len(testt_result1)):
                        anglesim_stack.append(angle_simm(stack_testt_result1[j], stack_month[i]))
                    index = anglesim_stack.index(min(anglesim_stack))
                    testtttt_result_14 = []
                    for k in range(len(testt_result_14[index][1][6:13])):
                        testtttt_result_14.append(stack_month[i][1][6] + (testt_result_14[index][1][6:13][k] - testt_result_14[index][1][5]))                    
                    msee =  mean_squared_error(stack_month[i+1][1], testtttt_result_14)
                    stock_mse.append(msee)
                    print msee 
                all_pattern_mse_results.append(stock_mse)
    return all_pattern_mse_results
                    
#WINDOW = 63               
patern_result_mse_14_118 = mseee(118)


# In[ ]:


import pandas as pd
import numpy as np
from pandas import datetime

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as plt
import matplotlib.dates as dates
from matplotlib.pylab import rcParams
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

maxx_ar = 5
maxx_ma = 5

p_values = range(5, 5)
d_values = range(1, 1)
q_values = range(0, 0)

window1 = 63
window2 = 91
window3 = 119

num_obs = 14

def splitt(dataset, window, num_obs):
    arima_stock_db = []
    for i in range(int((len(dataset)-window)/num_obs)):
        #print int((len(dataset)-window)/num_obs), "int((len(dataset)-window)/num_obs)"
        arima_stock_db.append(dataset[num_obs*i:window+num_obs*i])
    return arima_stock_db

def evaluate_models(dataset, p_values, d_values, q_values):
    best_score_AIC, best_cfg_aic = float("inf"), None  
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                model = ARIMA(series, order=(p,d,q)).fit()
                AIC = model.aic
                print AIC
                if AIC < best_score_AIC:
                    best_score_AIC, best_cfg_aic = AIC, order
    start = len(dataset)-13
    end = len(dataset)
    model = ARIMA(dataset, (0,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=7)[0]
    obs = dataset[start+6:]
    #print len(output[7:]), len(obs)
    return mean_squared_error(output,obs)


def parser(x):
    splited = x.split('-')
    return datetime.strptime( splited[0] + "-" + splited[1]+"-" + splited[2], "%Y-%m-%d")

root_1 = "/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr"  
#WINDOW = 63
all_mse_results_14 = []
for k, dirs, filenames in os.walk(root):
    for f in filenames: 
            if not f.startswith('.') and os.path.isfile(os.path.join(root_1, f)):
                print f
                dataMaster = pd.read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f)
                sp_500 = dataMaster['high']
                ts = pd.Series(dataMaster['high'].values)
                sp500_diff = ts - ts.shift()
                diff = sp500_diff.dropna()
                series = read_csv('/Users/admin/Desktop/sandp500 3 — копия/individual_stocks_5yr/'+f, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
                splitted_series = splitt(series, 118+14, num_obs)
                print np.array(splitted_series).shape, "splitted_seriesSHAPE"
                stock_mse = []
                for i in range(len(splitted_series)-1):
                    mse_result1 = evaluate_models(splitted_series[i], p_values, d_values, q_values)
                    stock_mse.append(mse_result1)
                all_mse_results_14.append(stock_mse)
                
r_14_118 =  pd.DataFrame(all_mse_results_14)


# In[ ]:


r_14_118 = r_14_118.transpose()

columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
         , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
        'CL_data.csv','CLX_data.csv']
r_14_118.columns = columns
#print r_14
pat_data_mse = pd.DataFrame(patern_result_mse_14_118)
pat_data_mse = pat_data_mse.transpose()
columns=['CBOE_data.csv', 'CBS_data.csv', 'CCI_data.csv','CCL_data.csv', 'CDNS_data.csv', 'CELG_data.csv', 'CERN_data.csv', 'CF_data.csv'
        , 'CFG_data.csv', 'CHD_data.csv', 'CHK_data.csv', 'CHRW_data.csv','CHTR_data.csv', 'CI_data.csv','CINF_data.csv',
       'CL_data.csv','CLX_data.csv']
pat_data_mse.columns = columns
#если разница положительная тогда mse patternov больше а значит ARIMA предсказала лучше

mse_compare = pat_data_mse - r_14_118

#print "ARIMA window = 118"
print mse_compare[mse_compare !=900].count()
print mse_compare[mse_compare < 0].count() 


# In[ ]:


IPython.Cell.options_default.cm_config.lineNumbers = true;

