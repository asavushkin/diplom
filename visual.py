
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
my_data = genfromtxt('/Users/admin/Desktop/generator.csv', delimiter=',')
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
for min_price_1 in range(0,4): 
    month_minim_1.append(minim_1[min_price_1])
#пошумим
#month_minim_1 = numpy.random.normal(month_minim_1)

for max_price_1 in range(0,4): 
    month_maxim_1.append(maxim_1[max_price_1])
month_maxim_1_back =month_maxim_1[::-1]

#пошумим
#month_maxim_1_back = numpy.random.normal(month_maxim_1_back)
                    #np.random.normal(x,3.,(10,))
for timestamp in range (0,4):
    timestamps_1.append(timestamp)
timestamps_back_1 = timestamps_1[::-1]

min_time_1 = np.column_stack([month_minim_1,timestamps_1])
max_time_1 = np.column_stack([month_maxim_1_back,timestamps_back_1]) 
min_time_1 = list(min_time_1)
max_time_1 = list(max_time_1)
data_1 = np.vstack((min_time_1,max_time_1))

#print data_1
plt.plot(timestamps_1, month_minim_1)
plt.plot(timestamps_back_1, month_maxim_1_back)

plt.show()
#print dist(data_1,data_2)

