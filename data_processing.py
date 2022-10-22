# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:03:15 2022

@author: seongjoon kang
"""
import numpy as np
import pandas as pd
import pickle

df120 = pd.read_csv('ray_tracing_data/beijing_paths_120.csv', delimiter=',', index_col = False)
df30 = pd.read_csv('ray_tracing_data/beijing_paths_30.csv', delimiter=',', index_col = False)
df60 = pd.read_csv('ray_tracing_data/beijing_paths_60.csv', delimiter=',', index_col = False)
df90 = pd.read_csv('ray_tracing_data/beijing_paths_90.csv', delimiter=',', index_col = False)

def data_processing(data):
  t = np.array([],dtype = int)
  for i in range (25):
    t = np.append(t, np.arange(12*i, 12*i+6))
  tx, ty, tz = np.array(data)[:,1], np.array(data)[:,2], np.array(data)[:,3]
  rx, ry, rz = np.array(data)[:,4], np.array(data)[:,5], np.array(data)[:,6]
  dist_vect_2d = np.column_stack((tx-rx, ty-ry))
  
  distance_2D = np.linalg.norm(dist_vect_2d, axis = 1)
  distance_2D = np.sort(distance_2D)
  
  sort_idx = np.argsort(distance_2D)
  
  distance_2D_adj = distance_2D[1:]
  distance_2D_adj = np.append(distance_2D_adj, distance_2D[-1]+5)
  
  dist_height = tz 
  
  data= np.array(data.T[10:].T)
  data = data[:,t]
  data = data.reshape(-1,25,6)
  data = np.transpose(data, axes = (0,2,1))
  return data[sort_idx], np.column_stack((distance_2D, distance_2D_adj, dist_height[sort_idx]))

data_30, loc_30 =  data_processing(df30)
data_60, loc_60 = data_processing(df60)
data_90, loc_90 = data_processing(df90)
data_120, loc_120 = data_processing(df120)

data_all = np.vstack(((data_30,data_60, data_90, data_120)))
location_all = np.vstack(((loc_30,loc_60, loc_90, loc_120)))
data_all[:,0,:] = -data_all[:,0,:]
#print(np.max(data_all[:,0,:][np.isnan(data_all[:,0,:])==False]))
data_all[:,1,:] = data_all[:,1,:]*1e7
b_index = np.isnan(data_all[:,0,:])
data_all[:,0,:][np.isnan(data_all[:,0,:])] =  np.random.uniform(low= 220, high=250, size = (np.sum(b_index),))
data_all[:,0,:] -=87 
data_all[:,1,:][np.isnan(data_all[:,1,:])] =  np.random.uniform(low = 0.6, high = 155, size = (np.sum(b_index),) )
data_all[:,2,:][np.isnan(data_all[:,2,:])] = np.random.uniform(low = 0, high= 180, size = (np.sum(b_index),))
data_all[:,3,:][np.isnan(data_all[:,3,:])] = np.random.uniform(low = -180, high =180, size = (np.sum(b_index),))
data_all[:,3,:][data_all[:,3,:]<0] += 360
data_all[:,4,:][np.isnan(data_all[:,4,:])] =  np.random.uniform(low = 0, high = 180, size = (np.sum(b_index),))
data_all[:,5,:][np.isnan(data_all[:,5,:])] = np.random.uniform(low = -180, high = 180, size = (np.sum(b_index),))
data_all[:,5,:][data_all[:,5,:]<0] += 360

for i in range(6):
  print(min(data_all[:,i,:].reshape(-1)), max(data_all[:,i,:].reshape(-1)))
data_all = data_all[:,None,:,:]

data_all_new = np.repeat(data_all, 8, axis = -2)
data_all_new = np.repeat(data_all_new, 2, axis = -1)

print(data_all_new.shape)

with open ('data_all.pickle', 'wb') as handle:
    pickle.dump(data_all_new/16, handle)

with open ('loc_all.pickle', 'wb') as handle:
    pickle.dump(location_all, handle)
    