
""" Trainer for pretrain phase. """
import os.path as osp
import os
import numpy as np
import h5py
import tqdm
import datetime
import psutil
import torch
from torch.utils.data import DataLoader 
import torch.nn as nn
from dataset_loader import *
from utils.misc import Averager, Resize_Feature,Timer
#from utils.Tool import *
from Network_torch import *
from scipy import io

class TestingLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.train_name = args.test_dataset_name +'_' + time_str+ '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 

        log_base_dir = './logs_fortest/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        test_base_dir = osp.join(log_base_dir, 'test')
        if not osp.exists(test_base_dir):
            os.mkdir(test_base_dir)

        self.save_path = osp.join(test_base_dir, self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        # basicFile = ['model']        
        # for file in basicFile:
        #     Path = os.path.join(self.save_path, file)
        #     if not os.path.exists(Path):
        #         os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'test' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        # Set args to be shareable in the class
        self.args = args

        # Build dataset and dataloader
        self.TestSet = LocTestSetLoader(args)

        self.Test_Loader = DataLoader(self.TestSet, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        self.sample_num = len(self.TestSet)
        self.channels = self.args.Out_channels
        self.height = self.args.test_height
        self.width = self.args.test_width
        print('The number of testing samples is %d' % (self.sample_num))
        print('The size of testing samples is %d x %d' % (self.height, self.width))


        # Build pretrain model
        if args.model_type == 'VGG16Attention':
            self.model = VGG16Attention(args.In_channels,args.Out_channels)
        elif args.model_type == 'UNet':
            self.model = UNet(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth':
            self.model = VGGDepth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1024':
            self.model = VGGDepth1024(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1':
            self.model = VGGDepth1(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGG19Depth':
            self.model = VGG19Depth(args.In_channels,args.Out_channels)
        
        # Load pretrain model
        if self.args.test_init_weights is not None:
            pretrained_dict = torch.load(self.args.test_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
         

    def test(self):



        # Set the test log
        testlog = {}
        testlog['args'] = vars(self.args)
        output = np.zeros((self.sample_num, self.channels,self.height,self.width), dtype=np.float32)


        test_result = {}

        self.model.eval()
        tqdm_test = tqdm.tqdm(self.Test_Loader)

        with torch.no_grad():
            
            for i, batch in enumerate(tqdm_test):
                Input_test = batch
                Input_test = np.transpose(Input_test, (0, 3, 1, 2))

                if torch.cuda.is_available():
                    Input_test = Input_test.cuda()
                q=i*self.args.test_batch_size
                #output[q:q+self.args.test_batch_size] = self.model(Input_test) # [0:args.test_batch_size]    
                output1 = self.model(Input_test).cpu().numpy()
                output[q:q+self.args.test_batch_size,:,:,:] = output1


                


        test_result['output'] = output#.cpu().numpy()
        #output = output.cpu().numpy()
        # Save the result and logs
        np.save(osp.join(self.save_path, 'test_result'), test_result)
        np.save(osp.join(self.save_path, 'test_trlog'), testlog)
        #io.savemat(osp.join(self.save_path + '/test_result.mat'), 'test_result', output)
        io.savemat(osp.join(self.save_path, 'test_result.mat'), {'test_result': output})
        print('Test is completed!')
        print('The result is saved in %s' % (self.save_path))
 


class DTestingLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.train_name = args.test_dataset_name +'_' + time_str+ '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 

        log_base_dir = './logs_fortest/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        test_base_dir = osp.join(log_base_dir, 'test')
        if not osp.exists(test_base_dir):
            os.mkdir(test_base_dir)

        self.save_path = osp.join(test_base_dir, self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        # basicFile = ['model']        
        # for file in basicFile:
        #     Path = os.path.join(self.save_path, file)
        #     if not os.path.exists(Path):
        #         os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'test' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        # Set args to be shareable in the class
        self.args = args

        # Build dataset and dataloader
        self.TestSet = LocTestSetLoader(args)

        self.Test_Loader = DataLoader(self.TestSet, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        self.sample_num = len(self.TestSet)
        self.channels = self.args.Out_channels
        self.height = self.args.test_height
        self.width = self.args.test_width
        print('The number of testing samples is %d' % (self.sample_num))
        print('The size of testing samples is %d x %d' % (self.height, self.width))


        # Build pretrain model
        if args.model_type == 'UNet_d':
            self.model = UNet_d(args.In_channels,192)
        else:
            self.model = UNet_d(args.In_channels,args.Out_channels)
        
        # Load pretrain model
        if self.args.test_init_weights is not None:
            pretrained_dict = torch.load(self.args.test_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
         

    def test(self):



        # Set the test log
        testlog = {}
        testlog['args'] = vars(self.args)
        output = np.zeros((self.sample_num, 192,96,64), dtype=np.float32)


        test_result = {}

        self.model.eval()
        tqdm_test = tqdm.tqdm(self.Test_Loader)

        with torch.no_grad():
            
            for i, batch in enumerate(tqdm_test):
                Input_test = batch
                Input_test = np.transpose(Input_test, (0, 3, 1, 2))

                if torch.cuda.is_available():
                    Input_test = Input_test.cuda()
                q=i*self.args.test_batch_size
                #output[q:q+self.args.test_batch_size] = self.model(Input_test) # [0:args.test_batch_size]    
                output1 = self.model(Input_test).cpu().numpy()
                output[q:q+self.args.test_batch_size,:,:,:] = output1


                


        test_result['output'] = output#.cpu().numpy()
        #output = output.cpu().numpy()
        # Save the result and logs
        np.save(osp.join(self.save_path, 'test_result'), test_result)
        np.save(osp.join(self.save_path, 'test_trlog'), testlog)
        #io.savemat(osp.join(self.save_path + '/test_result.mat'), 'test_result', output)
        io.savemat(osp.join(self.save_path, 'test_result.mat'), {'test_result': output})
        print('Test is completed!')
        print('The result is saved in %s' % (self.save_path))






class DepthTestingLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.train_name = args.test_dataset_name +'_' + time_str+ '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 

        log_base_dir = './logs_fortest/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        test_base_dir = osp.join(log_base_dir, 'test')
        if not osp.exists(test_base_dir):
            os.mkdir(test_base_dir)

        self.save_path = osp.join(test_base_dir, self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        # basicFile = ['model']        
        # for file in basicFile:
        #     Path = os.path.join(self.save_path, file)
        #     if not os.path.exists(Path):
        #         os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'test' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        # Set args to be shareable in the class
        self.args = args

        # Build dataset and dataloader
        self.TestSet = LocDepthTestSetLoader(args)

        self.Test_Loader = DataLoader(self.TestSet, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        self.sample_num = len(self.TestSet)
        self.channels = self.args.Out_channels
        self.height = self.args.test_height
        self.width = self.args.test_width
        print('The number of testing samples is %d' % (self.sample_num))
        print('The size of testing samples is %d x %d' % (self.height, self.width))


        # Build pretrain model
        if args.model_type == 'VGG16Attention':
            self.model = VGG16Attention(args.In_channels,args.Out_channels)
        elif args.model_type == 'UNet':
            self.model = UNet(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth':
            self.model = VGGDepth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1024':
            self.model = VGGDepth1024(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1':
            self.model = VGGDepth1(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGG19Depth':
            self.model = VGG19Depth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth256':
            self.model = VGGDepth256(args.In_channels,args.Out_channels)
        # Load pretrain model
        if self.args.test_init_weights is not None:
            pretrained_dict = torch.load(self.args.test_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
         

    def test(self):



        # Set the test log
        testlog = {}
        testlog['args'] = vars(self.args)
        output = np.zeros((self.sample_num, self.args.Out_channels), dtype=np.float32)


        test_result = {}

        self.model.eval()
        tqdm_test = tqdm.tqdm(self.Test_Loader)

        with torch.no_grad():
            
            for i, batch in enumerate(tqdm_test):
                Input_test = batch
                Input_test = np.transpose(Input_test, (0, 3, 1, 2))

                if torch.cuda.is_available():
                    Input_test = Input_test.cuda()
                q=i*self.args.test_batch_size
                #output[q:q+self.args.test_batch_size] = self.model(Input_test) # [0:args.test_batch_size]    
                output1 = self.model(Input_test).cpu().numpy()
                output[q:q+self.args.test_batch_size,:] = output1


                


        test_result['output'] = output#.cpu().numpy()
        #output = output.cpu().numpy()
        # Save the result and logs
        np.save(osp.join(self.save_path, 'test_result'), test_result)
        np.save(osp.join(self.save_path, 'test_trlog'), testlog)
        #io.savemat(osp.join(self.save_path + '/test_result.mat'), 'test_result', output)
        io.savemat(osp.join(self.save_path, 'test_result.mat'), {'test_result': output})
        print('Test is completed!')
        print('The result is saved in %s' % (self.save_path))



class DepthTestingLearner_withepixy(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.train_name = args.test_dataset_name +'_' + time_str+ '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 

        log_base_dir = './logs_fortest/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        test_base_dir = osp.join(log_base_dir, 'test')
        if not osp.exists(test_base_dir):
            os.mkdir(test_base_dir)

        self.save_path = osp.join(test_base_dir, self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        # basicFile = ['model']        
        # for file in basicFile:
        #     Path = os.path.join(self.save_path, file)
        #     if not os.path.exists(Path):
        #         os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'test' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        # Set args to be shareable in the class
        self.args = args

        # Build dataset and dataloader
        self.TestSet = LocDepthTestSetLoader_epi_xy(args)

        self.Test_Loader = DataLoader(self.TestSet, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        self.sample_num = len(self.TestSet)
        self.channels = self.args.Out_channels
        self.height = self.args.test_height
        self.width = self.args.test_width
        print('The number of testing samples is %d' % (self.sample_num))
        print('The size of testing samples is %d x %d' % (self.height, self.width))


        # Build pretrain model
        if args.model_type == 'VGG16Attention':
            self.model = VGG16Attention(args.In_channels,args.Out_channels)
        elif args.model_type == 'UNet':
            self.model = UNet(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth':
            self.model = VGGDepth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1024':
            self.model = VGGDepth1024(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1':
            self.model = VGGDepth1(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGG19Depth':
            self.model = VGG19Depth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth256':
            self.model = VGGDepth256(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth256_epi_xy':
            self.model = VGGDepth256_epi_xy(args.In_channels,args.Out_channels)
        # Load pretrain model
        if self.args.test_init_weights is not None:
            pretrained_dict = torch.load(self.args.test_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
         

    def test(self):



        # Set the test log
        testlog = {}
        testlog['args'] = vars(self.args)
        output = np.zeros((self.sample_num, self.args.Out_channels), dtype=np.float32)


        test_result = {}

        self.model.eval()
        tqdm_test = tqdm.tqdm(self.Test_Loader)

        with torch.no_grad():
            
            for i, batch in enumerate(tqdm_test):
                Input_test, Epicentral_distance, Station_X, Station_Y = batch
                Input_test = np.transpose(Input_test, (0, 3, 1, 2))
                Epicentral_distance = np.transpose(Epicentral_distance, (0,1))
                Station_X = np.transpose(Station_X, (0,1))
                Station_Y = np.transpose(Station_Y, (0,1))

                if torch.cuda.is_available():
                    Input_test = Input_test.cuda()
                    Epicentral_distance = Epicentral_distance.cuda()
                    Station_X = Station_X.cuda()
                    Station_Y = Station_Y.cuda()

                q=i*self.args.test_batch_size
                #output[q:q+self.args.test_batch_size] = self.model(Input_test) # [0:args.test_batch_size]    
                output1 = self.model(Input_test, Epicentral_distance, Station_X, Station_Y).cpu().numpy()
                output[q:q+self.args.test_batch_size,:] = output1


                


        test_result['output'] = output#.cpu().numpy()
        #output = output.cpu().numpy()
        # Save the result and logs
        np.save(osp.join(self.save_path, 'test_result'), test_result)
        np.save(osp.join(self.save_path, 'test_trlog'), testlog)
        #io.savemat(osp.join(self.save_path + '/test_result.mat'), 'test_result', output)
        io.savemat(osp.join(self.save_path, 'test_result.mat'), {'test_result': output})
        print('Test is completed!')
        print('The result is saved in %s' % (self.save_path))



class SingleStationDepthTestingLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.train_name = args.test_dataset_name + '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 

        log_base_dir = './logs_fortest/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        test_base_dir = osp.join(log_base_dir, 'test')
        if not osp.exists(test_base_dir):
            os.mkdir(test_base_dir)

        self.save_path = osp.join(test_base_dir, self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        # basicFile = ['model']        
        # for file in basicFile:
        #     Path = os.path.join(self.save_path, file)
        #     if not os.path.exists(Path):
        #         os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'test' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        # Set args to be shareable in the class
        self.args = args

        # Build dataset and dataloader
        self.TestSet = LocSingleStationDepthTestSetLoader(args)

        self.Test_Loader = DataLoader(self.TestSet, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        self.sample_num = len(self.TestSet)
        self.channels = self.args.Out_channels
        self.height = self.args.test_height
        self.width = self.args.test_width
        print('The number of testing samples is %d' % (self.sample_num))
        print('The size of testing samples is %d x %d' % (self.height, self.width))


        # Build pretrain model
        if args.model_type == 'VGGDepthSingle':
            self.model = VGGDepthSingle(args.In_channels,args.Out_channels)
        elif args.model_type == 'UNet':
            self.model = UNet(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth':
            self.model = VGGDepth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1024':
            self.model = VGGDepth1024(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1':
            self.model = VGGDepth1(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGG19Depth':
            self.model = VGG19Depth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepthSingle256':
            self.model = VGGDepthSingle256(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepthSingle256_epi':
            self.model = VGGDepthSingle256_epi(args.In_channels,args.Out_channels)
        
        # Load pretrain model
        if self.args.test_init_weights is not None:
            pretrained_dict = torch.load(self.args.test_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
         

    def test(self):



        # Set the test log
        testlog = {}
        testlog['args'] = vars(self.args)
        output = np.zeros((self.sample_num, 256), dtype=np.float32)


        test_result = {}

        self.model.eval()
        tqdm_test = tqdm.tqdm(self.Test_Loader)

        with torch.no_grad():
            
            for i, batch in enumerate(tqdm_test):
                Input_test = batch
                Input_test = np.transpose(Input_test, (0, 1, 3, 2))

                if torch.cuda.is_available():
                    Input_test = Input_test.cuda()
                q=i*self.args.test_batch_size
                #output[q:q+self.args.test_batch_size] = self.model(Input_test) # [0:args.test_batch_size]    
                output1 = self.model(Input_test).cpu().numpy()
                output[q:q+self.args.test_batch_size,:] = output1


                


        test_result['output'] = output#.cpu().numpy()
        #output = output.cpu().numpy()
        # Save the result and logs
        np.save(osp.join(self.save_path, 'test_result'), test_result)
        np.save(osp.join(self.save_path, 'test_trlog'), testlog)
        #io.savemat(osp.join(self.save_path + '/test_result.mat'), 'test_result', output)
        io.savemat(osp.join(self.save_path, 'test_result.mat'), {'test_result': output})
        print('Test is completed!')
        print('The result is saved in %s' % (self.save_path))
                
