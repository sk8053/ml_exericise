# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:03:15 2022

@author: seongjoon kang
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import scipy.constants as cs

os.getcwd()
sample_size = 40000

#df1 = pd.read_csv('ray_tracing_data/boston_ground_user_1.csv', delimiter=',', index_col = False)
#df2 = pd.read_csv('ray_tracing_data/boston_ground_user_2.csv', delimiter=',', index_col = False)
#df3 = pd.read_csv('ray_tracing_data/boston_ground_user_3.csv', delimiter=',', index_col = False)

#df_ground = pd.concat([df1, df2, df3], axis = 0, ignore_index = True)
#df_ground = pd.read_csv('ray_tracing_data/beijing_1_6m.csv', delimiter=',', index_col = False)
df_30 = pd.read_csv('ray_tracing_data/beijing_30m.csv', delimiter=',', index_col = False)
df_60 = pd.read_csv('ray_tracing_data/beijing_60m.csv', delimiter=',', index_col = False)
df_90 = pd.read_csv('ray_tracing_data/beijing_90m.csv', delimiter=',', index_col = False)
df_120 = pd.read_csv('ray_tracing_data/beijing_120m.csv', delimiter=',', index_col = False)
#df_150 = pd.read_csv('ray_tracing_data/boston_150m_1.csv', delimiter=',', index_col = False)


def data_processing(data, h = 30, ground = False):
    
  I1 = np.where(data['tx_z']!=10)[0]
  if ground is True:
      I2 = np.where((data['rx_z']>2.6) & (data['rx_z']<1.6))[0]
  else:
      I2 = np.where (data['rx_z'] != h)[0]
  I = np.union1d(I1, I2)
  data = data.drop(labels = I, axis = 0)
  
  I = np.random.permutation(np.arange(len(data)))
  data = data.iloc[I]
  data = data.iloc[:sample_size]
 
  tx, ty, tz = data['tx_x'], data['tx_y'], data['tx_z']
  rx, ry, rz = data['rx_x'], data['rx_y'], data['rx_z']
  dist_vect_2d = np.column_stack((tx-rx, ty-ry))
  dist_vect_3d = np.column_stack((tx-rx, ty-ry, tz-rz))
  
  distance_2D = np.linalg.norm(dist_vect_2d, axis = 1)
  distance_3D = np.linalg.norm(dist_vect_3d, axis = 1)
  min_delay = distance_3D/cs.speed_of_light
  fspl = 20*np.log10(distance_3D) + 20*np.log10(28e9) -147.55
  
  if ground is True:
      dist_height = np.ones_like(rz)
  else:
      dist_height = rz - tz
  #print(np.max(rz))
  #elv_ang = np.rad2deg(np.arctan2(distance_2D, dist_height))
  #dist_height = np.array(dist_height/10, dtype = int)*10
  def get_feature_value(key):
      feature_pd = pd.DataFrame()
      for i in range(25):
          feature_pd = pd.concat([feature_pd, data[key+'_%d'%(i+1)]], axis =1)
      return feature_pd
  
  data_all = np.zeros([len(data), 7, 25])
  data_all[:,0,:] = get_feature_value('path_loss') 
  data_all[:,1,:] = get_feature_value('delay') 
  data_all[:,2,:] = get_feature_value('zoa')
  data_all[:,3,:] = get_feature_value('aoa')
  data_all[:,4,:] = get_feature_value('zod')
  data_all[:,5,:] = get_feature_value('aod')
  link_state = data['link state']
  link_state[link_state==2] = np.random.uniform(0,0.1,len(link_state[link_state==2]))
  link_state[link_state==1] = np.random.uniform(1.9,2,len(link_state[link_state==1]))
  data_all[:,6,:] = np.repeat(link_state[:,None], 25, axis = -1)
  
  #data_all[:,7,:] = np.repeat(distance_2D[:,None], 25, axis = -1) # add 2-d distance
  #data_all[:,8,:] = np.repeat(dist_height[:,None], 25, axis = -1) # add height
  
  b_index = np.isnan(data_all[:,0,:])
  # 250
  
  data_all[:,0,:][np.isnan(data_all[:,0,:])] = np.random.uniform(low= 220, high=250, size = (np.sum(b_index),))
  data_all[:,0,:] -=fspl[:, None]
  
  data_all[:,1,:] = (data_all[:,1,:] -  min_delay[:,None])*1e7
  data_all[:,1,:][np.isnan(data_all[:,1,:])] = 75#np.random.uniform(low = 0.6, high = 155, size = (np.sum(b_index),))
  
  data_all[:,2,:][np.isnan(data_all[:,2,:])] =90# np.random.uniform(low = 0, high= 180, size = (np.sum(b_index),))
  data_all[:,3,:][np.isnan(data_all[:,3,:])] = 180 #np.random.uniform(low = -180, high =180, size = (np.sum(b_index),))
  data_all[:,3,:][data_all[:,3,:]<0] += 360
  data_all[:,4,:][np.isnan(data_all[:,4,:])] =90# np.random.uniform(low = 0, high = 180, size = (np.sum(b_index),))
  data_all[:,5,:][np.isnan(data_all[:,5,:])] = 180#np.random.uniform(low = -180, high = 180, size = (np.sum(b_index),))
  data_all[:,5,:][data_all[:,5,:]<0] += 360
  
  
  data_all[:,0,:]/=78
  data_all[:,1,:]/=75
  data_all[:,2,:]/=90 
  data_all[:,3,:]/=180  
  data_all[:,4,:]/=90 
  data_all[:,5,:]/=180 
  
  #data_all[:,7,:]/=1050 # 2d distance
  #data_all[:,8,:]/=56
  
  # make the range of data value [-1, 1] approximately
  #data_all = data_all
  
  return data_all, np.column_stack((distance_2D, dist_height))

#data_g, loc_g =  data_processing(df_ground, ground = True)
data_30, loc_30 =  data_processing(df_30, h = 30)
data_60, loc_60 =  data_processing(df_60, h = 60)
data_90, loc_90 =  data_processing(df_90, h = 90)
data_120, loc_120 =  data_processing(df_120, h = 120)
#data_150, loc_150 =  data_processing(df_150, h = 150)

#if len(np.where(loc_g[:,1] !=1.0)[0]) !=0:
#    raise RuntimeError('ground loc error')
if len(np.where(loc_30[:,1] !=20.0)[0]) !=0:
    raise RuntimeError('loc error at 30m height')
if len(np.where(loc_60[:,1] !=50.0)[0]) !=0:
    raise RuntimeError(' loc error at 60m height')
if len(np.where(loc_90[:,1] !=80.0)[0]) !=0:
    raise RuntimeError('loc error at 90m height')
if len(np.where(loc_120[:,1] !=110.0)[0]) !=0:
    raise RuntimeError('loc error at 120m height')
#if len(np.where(loc_150[:,1] !=140.0)[0]) !=0:
#    raise RuntimeError('loc error at 150m height')
    
# we can check the following as well
#np.where((height!=1.0) & (height!=20.0) & (height!=50.0)&(height!=80.0)&(height!=110.0)&(height!=140.0))[0]

data_all_1 = np.vstack((data_30, data_60, data_90, data_120))
loc_all_1 = np.vstack(( loc_30,  loc_60, loc_90, loc_120))

for i in range(7):
  print(min(data_all_1[:,i,:].reshape(-1)), max(data_all_1[:,i,:].reshape(-1)))

data_all_new_1 = data_all_1[:,None,:,:]
print(data_all_new_1.shape)

dir_ = 'C:\study of deep learning\Diffusion Model EX'
with open ('%s/data_all_1.pickle'%dir_, 'wb') as handle:
    pickle.dump(data_all_new_1-1, handle)
with open ('%s/loc_all_1.pickle'%dir_, 'wb') as handle:
    pickle.dump(loc_all_1, handle)

print(loc_all_1.shape)#, loc_all_2.shape)
#dist_2D = location_all[:,0]
#for i in np.arange(2000, step = 100):
#    I = np.where((dist_2D>i) & (dist_2D<i+100))[0]
#    print(i,i+100, len(I))    