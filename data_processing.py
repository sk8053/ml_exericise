# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:03:15 2022

@author: seongjoon kang
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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
  dist_vect_3d = np.column_stack((tx-rx, ty-ry, tz-rz))
  
  link_state = np.array(data)[:,9]*160
  link_state = np.repeat(link_state[:,None][:,None], 25, axis = -1)
  
  distance_2D = np.linalg.norm(dist_vect_2d, axis = 1)
  distance_3D = np.linalg.norm(dist_vect_3d, axis = 1)
  fspl = 20*np.log10(distance_3D) + 20*np.log10(28e9) -147.55
  
  dist_height = tz 
  
  data= np.array(data.T[10:].T)
  data = data[:,t]
  data = data.reshape(-1,25,6)
  data_all = np.transpose(data, axes = (0,2,1))
  
  data_all[:,0,:] = -data_all[:,0,:]
  #print(np.max(data_all[:,0,:][np.isnan(data_all[:,0,:])==False]))
  data_all[:,1,:] = data_all[:,1,:]*1e7
  b_index = np.isnan(data_all[:,0,:])
  data_all[:,0,:][np.isnan(data_all[:,0,:])] =  np.random.uniform(low= 220, high=250, size = (np.sum(b_index),))
  data_all[:,0,:] -=fspl[:, None] 
  data_all[:,1,:][np.isnan(data_all[:,1,:])] =  np.random.uniform(low = 0.6, high = 155, size = (np.sum(b_index),) )
  data_all[:,2,:][np.isnan(data_all[:,2,:])] = np.random.uniform(low = 0, high= 180, size = (np.sum(b_index),))
  data_all[:,3,:][np.isnan(data_all[:,3,:])] = np.random.uniform(low = -180, high =180, size = (np.sum(b_index),))
  data_all[:,3,:][data_all[:,3,:]<0] += 360
  data_all[:,4,:][np.isnan(data_all[:,4,:])] =  np.random.uniform(low = 0, high = 180, size = (np.sum(b_index),))
  data_all[:,5,:][np.isnan(data_all[:,5,:])] = np.random.uniform(low = -180, high = 180, size = (np.sum(b_index),))
  data_all[:,5,:][data_all[:,5,:]<0] += 360
  
  #data_all[:,[0,1,2,4],:] = 20*(data_all[:,[0,1,2,4],:] - 90)/180
  #data_all[:,[3,5],:] = 20*(data_all[:,[3,5],:] - 180)/360
  
  L = data_all.shape[0]
  #data = np.append(data, np.ones([L, 1, 25]), axis = 1)
  
  distance_2D_new = np.repeat(distance_2D[:,None]/10,25, axis = 1)
  distance_2D_new = distance_2D_new[:,None,:]
  dist_height_new = np.repeat(dist_height[:,None],25, axis =1)
  dist_height_new = dist_height_new[:,None,:]
  data_all = np.append(data_all, link_state, axis = 1)
  data_all = np.append(data_all, distance_2D_new, axis = 1)
  #data_all = np.append(data_all, dist_height_new, axis = 1)
  '''
  if tz[0] == 30:
      data_all = np.append(data_all, np.ones([L, 1, 25])*100, axis = 1)
      data_all = np.append(data_all, np.ones([L, 8, 7])*100, axis = -1)
  elif tz[0] == 60:
      data_all = np.append(data_all, np.ones([L, 1, 25])*200, axis = 1)
      data_all = np.append(data_all, np.ones([L, 8, 7])*200, axis = -1)
  elif tz[0] ==90:
      data_all = np.append(data_all, np.ones([L, 1, 25])*300, axis = 1)
      data_all = np.append(data_all, np.ones([L, 8, 7])*300, axis = -1)
  else:
      data_all = np.append(data_all, np.ones([L, 1, 25])*400, axis = 1)
      data_all = np.append(data_all, np.ones([L, 8, 7])*400, axis = -1)
 '''
  print(data_all.shape)
  return data_all, np.column_stack((distance_3D,distance_2D, dist_height))

data_30, loc_30 =  data_processing(df30)
data_60, loc_60 = data_processing(df60)
data_90, loc_90 = data_processing(df90)
data_120, loc_120 = data_processing(df120)

data_all = np.vstack(((data_30,data_60, data_90, data_120)))
location_all = np.vstack(((loc_30,loc_60, loc_90, loc_120)))

for i in range(6):
  print(min(data_all[:,i,:].reshape(-1)), max(data_all[:,i,:].reshape(-1)))
data_all = data_all[:,None,:,:]

data_all_new = np.repeat(data_all, 8, axis = -2)
data_all_new = np.repeat(data_all_new, 2, axis = -1)

print(data_all_new.shape)
half = int(data_all.shape[0]/2)

#dir_ = 'C:\study of deep learning\Diffusion Model EX'
dir_ = 'C:\study of deep learning\GAN_EX'
with open ('%s/data_all_1.pickle'%dir_, 'wb') as handle:
    pickle.dump(data_all_new[:half,:,:]/160, handle)
with open ('%s/data_all_2.pickle'%dir_, 'wb') as handle:
    pickle.dump(data_all_new[half:,:,:]/160, handle)

with open ('loc_all.pickle', 'wb') as handle:
    pickle.dump(location_all, handle)
print(location_all.shape)
dist_2D = location_all[:,0]
for i in np.arange(2000, step = 100):
    I = np.where((dist_2D>i) & (dist_2D<i+100))[0]
    print(i,i+100, len(I))    