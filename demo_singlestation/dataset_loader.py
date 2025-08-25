
""" Dataloader for all datasets. """
import os.path as osp
import os
import h5py
import torch
import random
from torch.utils.data import Dataset
import numpy as np

# class TrainSetLoader(Dataset):
#     def __init__(self, args):
#         self.args=args       

#         if args.ref_type == 'Constant':
#             self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Constant')
#             TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
#         elif args.ref_type == 'Reference':
#             self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
#             TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
#         with h5py.File(TrainSetSamplePath, 'r') as f:
#             self.pwr = f['Feature'][:]
#             self.label = f['Label'][:]
#             self.velcurve = f['VelCurve'][:]
#             self.Vrange = f['Vrange'][:]
#             self.LSG = f['LSG'][:]
#             self.RefVel = f['RefVel'][:]
       
#         self.samplenum = self.pwr.shape[0]

#     def __len__(self):
#         return self.samplenum

#     def __getitem__(self, i):

#         pwr = self.pwr[i]
#         label = self.label[i]
#         velcurve = self.velcurve[i]
#         Vrange = self.Vrange[i]
#         LSG = self.LSG[i]
#         RefVel = self.RefVel[i]

#         if self.args.ref_type == 'Reference':
#             pwr = np.repeat(pwr.reshape(1,pwr.shape[0],pwr.shape[1],pwr.shape[2]), self.args.ref_num, axis=0)
#             label = np.repeat(label.reshape(1,label.shape[0],label.shape[1]), self.args.ref_num, axis=0)
#             velcurve = np.repeat(velcurve.reshape(1,velcurve.shape[0],velcurve.shape[1]), self.args.ref_num, axis=0)
#             Vrange = np.repeat(Vrange.reshape(1,Vrange.shape[0]), self.args.ref_num, axis=0)

#         return pwr, label, velcurve, Vrange, LSG, RefVel

class LocTrainSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['Input_data'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['label_data'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]
        


#### Repeat the dataset 这还没写完
        if self.args.ref_type == 'Reference':
            Inputdata = np.repeat(Inputdata.reshape(1,Inputdata.shape[0],Inputdata.shape[1],Inputdata.shape[2]), self.args.ref_num, axis=0)



        return Inputdata, Labeldata

class LocXYTrainSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['Input_data'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['label_data'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]
        


#### Repeat the dataset 这还没写完
        if self.args.ref_type == 'Reference':
            Inputdata = np.repeat(Inputdata.reshape(1,Inputdata.shape[0],Inputdata.shape[1],Inputdata.shape[2]), self.args.ref_num, axis=0)



        return Inputdata, Labeldata


class LocDepthTrainSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['Input_data'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['label_data'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]
        


#### Repeat the dataset 这还没写完
        if self.args.ref_type == 'Reference':
            Inputdata = np.repeat(Inputdata.reshape(1,Inputdata.shape[0],Inputdata.shape[1],Inputdata.shape[2]), self.args.ref_num, axis=0)



        return Inputdata, Labeldata


class LocValidSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_Valid_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_Valid_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['val_Input'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['val_label'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]


        return Inputdata, Labeldata 



class LocDepthTrainSetLoader_epi_xy(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['Input_data'][:]
            self.Station_X = f['Station_X'][:]
            self.Station_Y = f['Station_Y'][:]
            self.Epicentral_distance = f['Epicentral_distance'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['label_data'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]
        Epicentral_distance = self.Epicentral_distance[i]
        Station_X = self.Station_X[i]
        Station_Y = self.Station_Y[i]
        


#### Repeat the dataset 这还没写完
        if self.args.ref_type == 'Reference':
            Inputdata = np.repeat(Inputdata.reshape(1,Inputdata.shape[0],Inputdata.shape[1],Inputdata.shape[2]), self.args.ref_num, axis=0)



        return Inputdata, Labeldata, Epicentral_distance, Station_X, Station_Y

class LocDepthValidSetLoader_epi_xy(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_Valid_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_Valid_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['val_Input'][:]
            self.Station_X = f['Station_X'][:]
            self.Station_Y = f['Station_Y'][:]
            self.Epicentral_distance = f['Epicentral_distance'][:]

        with h5py.File(TrainSetLabelPath, 'r') as f:
            
            self.Labeldata = f['val_label'][:]


       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]
        Epicentral_distance = self.Epicentral_distance[i]
        Station_X = self.Station_X[i]
        Station_Y= self.Station_Y[i]


        return Inputdata, Labeldata, Epicentral_distance, Station_X, Station_Y



class LocXYValidSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_Valid_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_Valid_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['val_Input'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['val_label'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]


        return Inputdata, Labeldata 

class LocDepthValidSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_Valid_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_Valid_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['val_Input'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['val_label'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]


        return Inputdata, Labeldata 


        
class LocTestSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

     
        self.filepath = os.path.join(args.test_dataset_dir)
        TestSetInputPath = os.path.join(self.filepath, 'Input_testdata.hdf5')
        # TestSetLabelPath = os.path.join(self.filepath, 'label_testdata.hdf5')

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TestSetInputPath, 'r') as f:
            self.Inputdata = f['Input_testdata'][:]
        
        # with h5py.File(TrainSetLabelPath, 'r') as f:
        #     self.Labeldata = f['label_testdata'][:]

       
        self.samplenum = self.Inputdata.shape[0]
 

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]

        return Inputdata

    
class LocDepthTestSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

     
        self.filepath = os.path.join(args.test_dataset_dir)
        TestSetInputPath = os.path.join(self.filepath, args.pre_input_data)
        # TestSetLabelPath = os.path.join(self.filepath, 'label_testdata.hdf5')

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TestSetInputPath, 'r') as f:
            self.Inputdata = f['Input_testdata'][:]
        
        # with h5py.File(TrainSetLabelPath, 'r') as f:
        #     self.Labeldata = f['label_testdata'][:]

       
        self.samplenum = self.Inputdata.shape[0]
 

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]

        return Inputdata

class LocDepthTestSetLoader_epi_xy(Dataset):
    def __init__(self, args):
        self.args=args       

     
        self.filepath = os.path.join(args.test_dataset_dir)
        TestSetInputPath = os.path.join(self.filepath, args.pre_input_data)
        # TestSetLabelPath = os.path.join(self.filepath, 'label_testdata.hdf5')

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TestSetInputPath, 'r') as f:
            self.Inputdata = f['Input_testdata'][:]
            self.Station_X = f['Station_X'][:]
            self.Station_Y = f['Station_Y'][:]
            self.Epicentral_distance = f['Epicentral_distance'][:]
        # with h5py.File(TrainSetLabelPath, 'r') as f:
        #     self.Labeldata = f['label_testdata'][:]

       
        self.samplenum = self.Inputdata.shape[0]
 

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Epicentral_distance = self.Epicentral_distance[i]
        Station_X = self.Station_X[i]
        Station_Y = self.Station_Y[i]

        return Inputdata, Epicentral_distance, Station_X, Station_Y

# class TestSetLoader(Dataset):
#     """The class to load the dataset"""
#     def __init__(self, args, start, samplenum):
#         self.args=args
#         self.samplenum = samplenum
#         self.start = start

#         self.TestSetNumber = np.arange(0, args.test_dataset_totalnum, 1)
#         self.TestSetPath = os.path.join(args.test_dataset_dir, args.test_dataset_name, args.test_dataset, 'total', 'Constant')
        
#         self.TrainSetPath = os.path.join(args.test_dataset_dir, args.test_dataset_name, args.test_dataset, args.test_trainset_ver, 'Constant')
#         self.TrainSetNumber = np.load(os.path.join(self.TrainSetPath, 'TrainSetNumber.npy'))

#         self.ValidSetPath = os.path.join(args.test_dataset_dir, args.test_dataset_name, args.test_dataset, args.test_trainset_ver, 'Constant')
#         self.ValidSetNumber = np.load(os.path.join(self.ValidSetPath, 'ValidSetNumber.npy'))

#         self.TestSetNumber = list(set(self.TestSetNumber) - set(self.TrainSetNumber) - set(self.ValidSetNumber))

#         self.samplenum_total = min(len(self.TestSetNumber), self.samplenum)

#     def __len__(self):
#         return self.samplenum_total

#     def __getitem__(self, i):

#         fileindex = self.TestSetNumber[self.start+i]

#         TestSetSamplePath = os.path.join(self.TestSetPath, 'DataSetSample'+str(fileindex)+'.h5')

#         with h5py.File(TestSetSamplePath, 'r') as f:
#             pwr = f['Feature'][:]
#             label = f['Label'][:]
#             velcurve = f['VelCurve'][:]
#             Vrange = f['Vrange'][:]
#             LSG = f['LSG'][:]
#             RefVel = f['RefVel'][:]
#             cdp = f['cdp'][()]
#             inline = f['inline'][()]
#             crossline = f['crossline'][()]

#         return pwr, label, velcurve, Vrange, LSG, RefVel, cdp, inline, crossline, fileindex
class LocSingleStationDepthTrainSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['Input_data'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['label_data'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]
        ### 训练改这里
        #Inputdata = Inputdata[:][0:1][:][:]
        Inputdata = np.expand_dims(Inputdata, axis=0)
        
        #Inputdata=Inputdata[1:2,:,:]
        #print(Inputdata.shape)
#### Repeat the dataset 这还没写完
        # if self.args.ref_type == 'Reference':
        #     Inputdata = np.repeat(Inputdata.reshape(1,Inputdata.shape[0],Inputdata.shape[1],Inputdata.shape[2]), self.args.ref_num, axis=0)



        return Inputdata, Labeldata


class LocSingleStation12DepthTrainSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['Input_data'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['label_data'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]

#### Repeat the dataset 这还没写完
        if self.args.ref_type == 'Reference':
            Inputdata = np.repeat(Inputdata.reshape(1,Inputdata.shape[0],Inputdata.shape[1],Inputdata.shape[2]), self.args.ref_num, axis=0)



        return Inputdata, Labeldata

class LocSingleStationDepthValidSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_Valid_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_Valid_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['val_Input'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['val_label'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]
      ##### 训练改这里  
        #Inputdata = Inputdata[:][0:1][:][:] 
       
        Inputdata = np.expand_dims(Inputdata, axis=0)
        
        #Inputdata=Inputdata[1:2,:,:]  
        #print(Inputdata.shape)

        return Inputdata, Labeldata
        



class LocSingleStation12DepthValidSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

        if args.ref_type == 'Constant':  # Constant means directly use the dataset
            self.filepath = os.path.join(args.pre_dataset_dir)
            TrainSetInputPath = os.path.join(self.filepath, args.pre_Valid_input_data)
            TrainSetLabelPath = os.path.join(self.filepath, args.pre_Valid_label_data)

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TrainSetInputPath, 'r') as f:
            self.Inputdata = f['val_Input'][:]
        
        with h5py.File(TrainSetLabelPath, 'r') as f:
            self.Labeldata = f['val_label'][:]

       
        self.samplenum = self.Inputdata.shape[0]

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    
        Inputdata = self.Inputdata[i]
        Labeldata = self.Labeldata[i]

#### Repeat the dataset 这还没写完
        if self.args.ref_type == 'Reference':
            Inputdata = np.repeat(Inputdata.reshape(1,Inputdata.shape[0],Inputdata.shape[1],Inputdata.shape[2]), self.args.ref_num, axis=0)



        return Inputdata, Labeldata

class LocSingleStationDepthTestSetLoader(Dataset):
    def __init__(self, args):
        self.args=args       

     
        self.filepath = os.path.join(args.test_dataset_dir)
        TestSetInputPath = os.path.join(self.filepath, args.pre_input_data)
        # TestSetLabelPath = os.path.join(self.filepath, 'label_testdata.hdf5')

        # elif args.ref_type == 'Reference': # Reference means use the dataset and repeat the dataset
        #     self.filepath = os.path.join(args.pre_dataset_dir, args.pre_dataset_name, args.pre_dataset, args.pre_dataset_ver, 'Reference')
        #     TrainSetSamplePath = os.path.join(self.filepath, 'TrainSetSample.h5')
            
        with h5py.File(TestSetInputPath, 'r') as f:
            self.Inputdata = f['Input_testdata'][:]
        
        # with h5py.File(TrainSetLabelPath, 'r') as f:
        #     self.Labeldata = f['label_testdata'][:]

       
        self.samplenum = self.Inputdata.shape[0]
 

    def __len__(self):
        return self.samplenum

    def __getitem__(self, i):

    ##### 测试改这里
        Inputdata = self.Inputdata[i]
        Inputdata = np.expand_dims(Inputdata, axis=0)
        
        
        return Inputdata

# class TestSetLoader(Dataset):
#     """The class to load the dataset"""
#     def __init__(self, args, start, samplenum):
#         self.args=args
#         self.samplenum = samplenum
#         self.start = start

#         self.TestSetNumber = np.arange(0, args.test_dataset_totalnum, 1)
#         self.TestSetPath = os.path.join(args.test_dataset_dir, args.test_dataset_name, args.test_dataset, 'total', 'Constant')
        
#         self.TrainSetPath = os.path.join(args.test_dataset_dir, args.test_dataset_name, args.test_dataset, args.test_trainset_ver, 'Constant')
#         self.TrainSetNumber = np.load(os.path.join(self.TrainSetPath, 'TrainSetNumber.npy'))

#         self.ValidSetPath = os.path.join(args.test_dataset_dir, args.test_dataset_name, args.test_dataset, args.test_trainset_ver, 'Constant')
#         self.ValidSetNumber = np.load(os.path.join(self.ValidSetPath, 'ValidSetNumber.npy'))

#         self.TestSetNumber = list(set(self.TestSetNumber) - set(self.TrainSetNumber) - set(self.ValidSetNumber))

#         self.samplenum_total = min(len(self.TestSetNumber), self.samplenum)

#     def __len__(self):
#         return self.samplenum_total

#     def __getitem__(self, i):

#         fileindex = self.TestSetNumber[self.start+i]

#         TestSetSamplePath = os.path.join(self.TestSetPath, 'DataSetSample'+str(fileindex)+'.h5')

#         with h5py.File(TestSetSamplePath, 'r') as f:
#             pwr = f['Feature'][:]
#             label = f['Label'][:]
#             velcurve = f['VelCurve'][:]
#             Vrange = f['Vrange'][:]
#             LSG = f['LSG'][:]
#             RefVel = f['RefVel'][:]
#             cdp = f['cdp'][()]
#             inline = f['inline'][()]
#             crossline = f['crossline'][()]

#         return pwr, label, velcurve, Vrange, LSG, RefVel, cdp, inline, crossline, fileindex
