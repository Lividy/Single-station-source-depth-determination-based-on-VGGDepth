
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
from utils.misc import Averager, Resize_Feature,Timer
#from utils.Tool import *
from Network_torch import *
from scipy import io

from dataset_loader import *

#from loss.loss import Dice_Loss, Focal_Loss
from utils.misc import Averager, Timer
from Network_torch import *
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt 

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 将输入和目标展平为一维
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集和并集
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        # 计算 Dice 系数
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 返回 Dice Loss
        return 1 - dice_score

class PreDTrainLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m-%d_%H-%M")
        self.train_name = args.pre_dataset_name + '_' + time_str + '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 
        log_base_dir = './logs/'  # The base dir for logs
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        pre_base_dir = osp.join(log_base_dir, 'train')    # The base dir for pretrain logs
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        self.save_path = osp.join(pre_base_dir,  self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        basicFile = ['model']        
        for file in basicFile:
            Path = os.path.join(self.save_path, file)
            if not os.path.exists(Path):
                os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'pre' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        self.TBlog_path = os.path.join('./TBlog', self.train_name) # The path to save tensorboard logs
        if not os.path.exists(self.TBlog_path):
            os.makedirs(self.TBlog_path)

        # Set args to be shareable in the class
        self.args = args

        # initialize some parameters
        #self.t0Int = np.arange(0, args.pre_nt * args.pre_dt, args.pre_dt)

        # Build dataset and dataloader
        self.TrainSet = LocTrainSetLoader(args)

        self.Train_Loader = DataLoader(self.TrainSet, batch_size=args.pre_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.ValidSet = LocValidSetLoader(args)

        self.Valid_Loader = DataLoader(self.ValidSet, batch_size=args.pre_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        self.ValidSampleNum = len(self.ValidSet)

        # Set the samples to visualize
        if self.ValidSampleNum > 10:
            self.VisualSampleNum = 10
        else:
            self.VisualSampleNum = self.ValidSampleNum

        self.VisualSample = list(np.random.choice(self.ValidSampleNum, self.VisualSampleNum, replace=False)) # Randomly choose 10 samples to visualize

        # Build pretrain model
        if args.model_type == 'UNet_d':
            self.model = UNet_d(args.In_channels,args.Out_channels)
        else:
            self.model = UNet(args.In_channels,args.Out_channels)
        
        # Load pretrain model
        if self.args.pre_init_weights is not None:
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set loss functions
        if args.loss_type == 'BCE':
            self.criterion = nn.BCELoss()
        elif args.loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif args.loss_type == 'Focal':
            self.criterion = Focal_Loss()
        elif args.loss_type == 'Dice':
            self.criterion = Dice_Loss()
        elif args.loss_type == 'BCEWithLogits':
            self.criterion = nn.BCEWithLogitsLoss()

    

        # Set optimizer 
        if args.pre_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.pre_lr)
        elif args.pre_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.pre_lr, momentum=args.pre_custom_momentum, weight_decay=args.pre_custom_weight_decay)

        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, gamma=self.args.pre_gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()


    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.save_path, 'model', name + '.pth'))
    
    def train(self):

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['val_VMAE'] = []        
        trlog['min_loss'] = 1e10
        trlog['min_VMAE'] = 1e10
        trlog['max_VMAE_epoch'] = []
        trlog['min_VMAE_epoch'] = []
        
        # Set the timer
        timer = Timer()  # Timer for the training time

        # Set iteration count to zero
        iter_count = 0
        # Set tensorboardX
        now = datetime.datetime.now()  # Get the current time
        datestr = now.strftime('%Y-%m-%d_%H:%M:%S')  # Get the current time in the format of string
        writer = SummaryWriter(log_dir=self.TBlog_path, comment=datestr)  # Set the tensorboard writer

        # Start training
        for epoch in range(1, self.args.pre_max_epoch + 1):
            
            # Set the model to train mode
            self.model.train()

            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_train = tqdm.tqdm(self.Train_Loader)      

            for i, batch in enumerate(tqdm_train, 1):
                # Update global count number 
                iter_count = iter_count + 1

                input, label = batch

                input = np.transpose(input, (0, 3, 1, 2))
                label = np.transpose(label, (0, 2, 1, 3))

                ############### 条件判断，是否需要对数据进行处理 ###############
                #可以在这里重复赋值，以便于在这里进行数据的处理
                # if self.args.ref_type == 'Reference':
                #     input = input[0]    
                #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])


                if torch.cuda.is_available():
                    input = input.cuda()
                    label = label.cuda()


                self.optimizer.zero_grad()

                output = self.model(input)
                #output = torch.squeeze(output, dim=1) # Remove the channel dimension

                # Calculate train loss
                loss = self.criterion(output, label)

                # Save loss to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Train-Loss(iter)', loss.item(), global_step=iter_count)                

                # Add loss for the averagers
                train_loss_averager.add(loss.item()) # Add the loss to the averager
                #print(train_loss_averager.item())

                tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, loss))
                loss.backward()
                self.optimizer.step()

            train_loss_averager = train_loss_averager.item()  ## Get the loss value

            # Save loss to tensorboard
            writer.add_scalar(self.train_name+'_Pre-Train-Loss(epoch)', train_loss_averager, global_step=epoch)
            
            print('Epoch {}/{}, Train: Loss={:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager))
            
            # Update learning rate
            self.lr_scheduler.step()

            # Start validation for specific epoch
            if epoch % self.args.pre_val_epoch == 0:
                # Set the model to valid mode
                self.model.eval()

                # Set averager classes to record validation losses and accuracies
                val_loss_averager = Averager()


                max_VMAE_epoch = 0
                min_VMAE_epoch = 1e101
                with torch.no_grad():
                    for i, batch in enumerate(self.Valid_Loader, 1):
                        val_input, val_label = batch

                        # if self.args.ref_type == 'Reference':

                        #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])
                        val_input = np.transpose(val_input, (0, 3, 1, 2))
                        val_label = np.transpose(val_label, (0, 2, 1, 3))

                        if torch.cuda.is_available():
                            val_input = val_input.cuda()
                            val_label = val_label.cuda()
                    
        

                        output = self.model(val_input)
                        #output = torch.squeeze(output, dim=1)

                        # Calculate valid loss
                        loss = self.criterion(output, val_label)

                        # Calculate valid VMAE
                        #APCurve,_ = GetResult(output.cpu().numpy(), self.t0Int, Vrange, threshold=self.args.Predthre)
                        
                        # VMAE_valid, VMAE_list = VMAE(APCurve, velcurve.numpy())  ## Get the VMAE value
                        
                        # if min(VMAE_list) < min_VMAE_epoch:
                        #     min_VMAE_epoch = min(VMAE_list)

                        # if max(VMAE_list) > max_VMAE_epoch:
                        #     max_VMAE_epoch = max(VMAE_list)
                       
                        # Add loss and VMAE for the averagers
                        val_loss_averager.add(loss.item())
                        # val_VMAE_averager.add(VMAE_valid)

                        # Save seg map to tensorboard
                        # if self.args.ref_type == 'Constant':
                        #     index = index.numpy()
                        #     for ind, name_ind in enumerate(index):
                        #         if name_ind in self.VisualSample:
                        #             temp = torch.zeros((3,output.shape[1], output.shape[2]))
                        #             temp[0,:,:] = output[ind,:,:].cpu()
                        #             temp[1,:,:] = label[ind,:,:]
                        #             writer.add_image(self.train_name+'_Pre-Valid-SegProbMap-{}'.format(name_ind), temp, global_step=epoch, dataformats='CHW')
                        # elif self.args.ref_type == 'Reference':
                        #     index = index.numpy()
                        #     for ind, name_ind in enumerate(index):
                        #         if name_ind in self.VisualSample:
                        #             temp = torch.zeros((3,output.shape[1], output.shape[2]))
                        #             temp[0,:,:] = output[ind,:,:].cpu()
                        #             temp[1,:,:] = label[ind,:,:]
                        #             writer.add_image(self.train_name+'_Pre-Train-SegProbMap-{}'.format(name_ind), temp, global_step=epoch, dataformats='CHW')

                # Update the averagers
                val_loss_averager = val_loss_averager.item()
            
                
                # Save loss and VMAE to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Valid-Loss', val_loss_averager, global_step=epoch)


    


                # Print loss and accuracy for this epoch
                print('Epoch {}/{}, Val: Loss={:.6f} '.format(epoch, self.args.pre_max_epoch, val_loss_averager))
            #####    
            # Save model every 10 epochs
            if epoch % 2 == 0:
                self.save_model('pre_epoch_{}'.format(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            # trlog['val_loss'].append(val_loss_averager)
            # trlog['val_VMAE'].append(val_VMAE_averager)#print(train_loss_averager  )
            np.save(osp.join(self.save_path, 'pre_trlog'), trlog)

            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.pre_max_epoch)))

        print('Training Finished! Running Time: {}'.format(timer.measure()))

        # Close tensorboardX writer
        writer.close()


class PreTrainLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m-%d_%H-%M")
        self.train_name = args.pre_dataset_name + '_' + time_str + '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 
        log_base_dir = './logs/'  # The base dir for logs
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        pre_base_dir = osp.join(log_base_dir, 'train')    # The base dir for pretrain logs
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        self.save_path = osp.join(pre_base_dir,  self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        basicFile = ['model']        
        for file in basicFile:
            Path = os.path.join(self.save_path, file)
            if not os.path.exists(Path):
                os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'pre' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        self.TBlog_path = os.path.join('./TBlog', self.train_name) # The path to save tensorboard logs
        if not os.path.exists(self.TBlog_path):
            os.makedirs(self.TBlog_path)

        # Set args to be shareable in the class
        self.args = args

        # initialize some parameters
        #self.t0Int = np.arange(0, args.pre_nt * args.pre_dt, args.pre_dt)

        # Build dataset and dataloader
        self.TrainSet = LocTrainSetLoader(args)

        self.Train_Loader = DataLoader(self.TrainSet, batch_size=args.pre_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.ValidSet = LocValidSetLoader(args)

        self.Valid_Loader = DataLoader(self.ValidSet, batch_size=args.pre_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        self.ValidSampleNum = len(self.ValidSet)

        # Set the samples to visualize
        if self.ValidSampleNum > 10:
            self.VisualSampleNum = 10
        else:
            self.VisualSampleNum = self.ValidSampleNum

        self.VisualSample = list(np.random.choice(self.ValidSampleNum, self.VisualSampleNum, replace=False)) # Randomly choose 10 samples to visualize

        # Build pretrain model
        if args.model_type == 'UNet':
            self.model = UNet(args.In_channels,args.Out_channels)
        else:
            self.model = UNet(args.In_channels,args.Out_channels)
        
        # Load pretrain model
        if self.args.pre_init_weights is not None:
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set loss functions
        if args.loss_type == 'BCE':
            self.criterion = nn.BCELoss()
        elif args.loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif args.loss_type == 'Focal':
            self.criterion = Focal_Loss()
        elif args.loss_type == 'Dice':
            self.criterion = Dice_Loss()
        elif args.loss_type == 'BCEWithLogits':
            self.criterion = nn.BCEWithLogitsLoss()

    

        # Set optimizer 
        if args.pre_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.pre_lr)
        elif args.pre_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.pre_lr, momentum=args.pre_custom_momentum, weight_decay=args.pre_custom_weight_decay)

        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, gamma=self.args.pre_gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()


    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.save_path, 'model', name + '.pth'))
    
    def train(self):

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['val_VMAE'] = []        
        trlog['min_loss'] = 1e10
        trlog['min_VMAE'] = 1e10
        trlog['max_VMAE_epoch'] = []
        trlog['min_VMAE_epoch'] = []
        
        # Set the timer
        timer = Timer()  # Timer for the training time

        # Set iteration count to zero
        iter_count = 0
        # Set tensorboardX
        now = datetime.datetime.now()  # Get the current time
        datestr = now.strftime('%Y-%m-%d_%H:%M:%S')  # Get the current time in the format of string
        writer = SummaryWriter(log_dir=self.TBlog_path, comment=datestr)  # Set the tensorboard writer

        # Start training
        for epoch in range(1, self.args.pre_max_epoch + 1):
            
            # Set the model to train mode
            self.model.train()

            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_train = tqdm.tqdm(self.Train_Loader)      

            for i, batch in enumerate(tqdm_train, 1):
                # Update global count number 
                iter_count = iter_count + 1

                input, label = batch

                input = np.transpose(input, (0, 3, 1, 2))
                label = np.transpose(label, (0, 3, 1, 2))

                ############### 条件判断，是否需要对数据进行处理 ###############
                #可以在这里重复赋值，以便于在这里进行数据的处理
                # if self.args.ref_type == 'Reference':
                #     input = input[0]    
                #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])


                if torch.cuda.is_available():
                    input = input.cuda()
                    label = label.cuda()


                self.optimizer.zero_grad()

                output = self.model(input)
                #output = torch.squeeze(output, dim=1) # Remove the channel dimension

                # Calculate train loss
                loss = self.criterion(output, label)

                # Save loss to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Train-Loss(iter)', loss.item(), global_step=iter_count)                

                # Add loss for the averagers
                train_loss_averager.add(loss.item()) # Add the loss to the averager
                #print(train_loss_averager.item())from dataset_loader import LocTrainSetLoader, LocValidSetLoader
                tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, loss))

                
                loss.backward()
                self.optimizer.step()

            train_loss_averager = train_loss_averager.item()  ## Get the loss value

            # Save loss to tensorboard
            writer.add_scalar(self.train_name+'_Pre-Train-Loss(epoch)', train_loss_averager, global_step=epoch)
            
            print('Epoch {}/{}, Train: Loss={:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager))
            
            # Update learning rate
            self.lr_scheduler.step()

            # Start validation for specific epoch
            if epoch % self.args.pre_val_epoch == 0:
                # Set the model to valid mode
                self.model.eval()

                # Set averager classes to record validation losses and accuracies
                val_loss_averager = Averager()


                max_VMAE_epoch = 0
                min_VMAE_epoch = 1e101
                with torch.no_grad():
                    for i, batch in enumerate(self.Valid_Loader, 1):
                        val_input, val_label = batch

                        # if self.args.ref_type == 'Reference':

                        #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])
                        val_input = np.transpose(val_input, (0, 3, 1, 2))
                        val_label = np.transpose(val_label, (0, 3, 1, 2))

                        if torch.cuda.is_available():
                            val_input = val_input.cuda()
                            val_label = val_label.cuda()
                    
        

                        output = self.model(val_input)
                        #output = torch.squeeze(output, dim=1)

                        # Calculate valid loss
                        loss = self.criterion(output, val_label)

                        # Calculate valid VMAE
                        #APCurve,_ = GetResult(output.cpu().numpy(), self.t0Int, Vrange, threshold=self.args.Predthre)
                        
                        # VMAE_valid, VMAE_list = VMAE(APCurve, velcurve.numpy())  ## Get the VMAE value
                        
                        # if min(VMAE_list) < min_VMAE_epoch:
                        #     min_VMAE_epoch = min(VMAE_list)

                        # if max(VMAE_list) > max_VMAE_epoch:
                        #     max_VMAE_epoch = max(VMAE_list)
                       
                        # Add loss and VMAE for the averagers
                        val_loss_averager.add(loss.item())
                        # val_VMAE_averager.add(VMAE_valid)

                        # Save seg map to tensorboard
                        # if self.args.ref_type == 'Constant':
                        #     index = index.numpy()
                        #     for ind, name_ind in enumerate(index):
                        #         if name_ind in self.VisualSample:
                        #             temp = torch.zeros((3,output.shape[1], output.shape[2]))
                        #             temp[0,:,:] = output[ind,:,:].cpu()
                        #             temp[1,:,:] = label[ind,:,:]
                        #             writer.add_image(self.train_name+'_Pre-Valid-SegProbMap-{}'.format(name_ind), temp, global_step=epoch, dataformats='CHW')
                        # elif self.args.ref_type == 'Reference':
                        #     index = index.numpy()
                        #     for ind, name_ind in enumerate(index):
                        #         if name_ind in self.VisualSample:
                        #             temp = torch.zeros((3,output.shape[1], output.shape[2]))
                        #             temp[0,:,:] = output[ind,:,:].cpu()
                        #             temp[1,:,:] = label[ind,:,:]
                        #             writer.add_image(self.train_name+'_Pre-Train-SegProbMap-{}'.format(name_ind), temp, global_step=epoch, dataformats='CHW')

                # Update the averagers
                val_loss_averager = val_loss_averager.item()
            
                
                # Save loss and VMAE to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Valid-Loss', val_loss_averager, global_step=epoch)


    


                # Print loss and accuracy for this epoch
                print('Epoch {}/{}, Val: Loss={:.6f} '.format(epoch, self.args.pre_max_epoch, val_loss_averager))
            #####    
            # Save model every 10 epochs
            if epoch % 2 == 0:
                self.save_model('pre_epoch_{}'.format(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            # trlog['val_loss'].append(val_loss_averager)
            # trlog['val_VMAE'].append(val_VMAE_averager)#print(train_loss_averager  )
            np.save(osp.join(self.save_path, 'pre_trlog'), trlog)

            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.pre_max_epoch)))

        print('Training Finished! Running Time: {}'.format(timer.measure()))

        # Close tensorboardX writer
        writer.close()
                

            

class XYTrainLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m-%d_%H-%M")
        self.train_name = args.pre_dataset_name + '_' + time_str + '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 
        log_base_dir = './logs/'  # The base dir for logs
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        pre_base_dir = osp.join(log_base_dir, 'train')    # The base dir for pretrain logs
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        self.save_path = osp.join(pre_base_dir,  self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        basicFile = ['model']        
        for file in basicFile:
            Path = os.path.join(self.save_path, file)
            if not os.path.exists(Path):
                os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'pre' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        self.TBlog_path = os.path.join('./TBlog', self.train_name) # The path to save tensorboard logs
        if not os.path.exists(self.TBlog_path):
            os.makedirs(self.TBlog_path)

        # Set args to be shareable in the class
        self.args = args

        # initialize some parameters
        #self.t0Int = np.arange(0, args.pre_nt * args.pre_dt, args.pre_dt)

        # Build dataset and dataloader
        self.TrainSet = LocXYTrainSetLoader(args)

        self.Train_Loader = DataLoader(self.TrainSet, batch_size=args.pre_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.ValidSet = LocXYValidSetLoader(args)

        self.Valid_Loader = DataLoader(self.ValidSet, batch_size=args.pre_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        self.ValidSampleNum = len(self.ValidSet)

        # Set the samples to visualize
        if self.ValidSampleNum > 10:
            self.VisualSampleNum = 10
        else:
            self.VisualSampleNum = self.ValidSampleNum

        self.VisualSample = list(np.random.choice(self.ValidSampleNum, self.VisualSampleNum, replace=False)) # Randomly choose 10 samples to visualize

        # Build pretrain model
        if args.model_type == 'UNet':
            self.model = UNet(args.In_channels,args.Out_channels)
        else:
            self.model = UNet(args.In_channels,args.Out_channels)
        
        # Load pretrain model
        if self.args.pre_init_weights is not None:
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set loss functions
        if args.loss_type == 'BCE':
            self.criterion = nn.BCELoss()
        elif args.loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif args.loss_type == 'Focal':
            self.criterion = Focal_Loss()
        elif args.loss_type == 'Dice':
            self.criterion = Dice_Loss()
        elif args.loss_type == 'BCEWithLogits':
            self.criterion = nn.BCEWithLogitsLoss()

    

        # Set optimizer 
        if args.pre_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.pre_lr)
        elif args.pre_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.pre_lr, momentum=args.pre_custom_momentum, weight_decay=args.pre_custom_weight_decay)

        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, gamma=self.args.pre_gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()


    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.save_path, 'model', name + '.pth'))
    
    def train(self):

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['val_VMAE'] = []        
        trlog['min_loss'] = 1e10
        trlog['min_VMAE'] = 1e10
        trlog['max_VMAE_epoch'] = []
        trlog['min_VMAE_epoch'] = []
        
        # Set the timer
        timer = Timer()  # Timer for the training time

        # Set iteration count to zero
        iter_count = 0
        # Set tensorboardX
        now = datetime.datetime.now()  # Get the current time
        datestr = now.strftime('%Y-%m-%d_%H:%M:%S')  # Get the current time in the format of string
        writer = SummaryWriter(log_dir=self.TBlog_path, comment=datestr)  # Set the tensorboard writer

        # Start training
        for epoch in range(1, self.args.pre_max_epoch + 1):
            
            # Set the model to train mode
            self.model.train()

            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_train = tqdm.tqdm(self.Train_Loader)      

            for i, batch in enumerate(tqdm_train, 1):
                # Update global count number 
                iter_count = iter_count + 1

                input, label = batch

                input = np.transpose(input, (0, 3, 1, 2))
                label = np.transpose(label, (0, 3, 1, 2))

                ############### 条件判断，是否需要对数据进行处理 ###############
                #可以在这里重复赋值，以便于在这里进行数据的处理
                # if self.args.ref_type == 'Reference':
                #     input = input[0]    
                #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])


                if torch.cuda.is_available():
                    input = input.cuda()
                    label = label.cuda()


                self.optimizer.zero_grad()

                output = self.model(input)
                #output = torch.squeeze(output, dim=1) # Remove the channel dimension

                # Calculate train loss
                loss = self.criterion(output, label)

                # Save loss to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Train-Loss(iter)', loss.item(), global_step=iter_count)                

                # Add loss for the averagers
                train_loss_averager.add(loss.item()) # Add the loss to the averager
                #print(train_loss_averager.item())

                # Print loss till this step
                #tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager.item()))
                tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, loss))
                # Loss backwards and optimizer updates
                loss.backward()
                self.optimizer.step()

            train_loss_averager = train_loss_averager.item()  ## Get the loss value

            # Save loss to tensorboard
            writer.add_scalar(self.train_name+'_Pre-Train-Loss(epoch)', train_loss_averager, global_step=epoch)
            
            print('Epoch {}/{}, Train: Loss={:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager))
            
            # Update learning rate
            self.lr_scheduler.step()


            if epoch % self.args.pre_val_epoch == 0:
                # Set the model to valid mode
                self.model.eval()
                val_loss_averager = Averager()
                max_VMAE_epoch = 0
                min_VMAE_epoch = 1e101
                with torch.no_grad():
                    for i, batch in enumerate(self.Valid_Loader, 1):
                        val_input, val_label = batch

                        # if self.args.ref_type == 'Reference':

                        #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])
                        val_input = np.transpose(val_input, (0, 3, 1, 2))
                        val_label = np.transpose(val_label, (0, 3, 1, 2))

                        if torch.cuda.is_available():
                            val_input = val_input.cuda()
                            val_label = val_label.cuda()
                    
        

                        output = self.model(val_input)
                        #output = torch.squeeze(output, dim=1)

                        # Calculate valid loss
                        loss = self.criterion(output, val_label)

                        # Calculate valid VMAE
                        #APCurve,_ = GetResult(output.cpu().numpy(), self.t0Int, Vrange, threshold=self.args.Predthre)
                        
                        # VMAE_valid, VMAE_list = VMAE(APCurve, velcurve.numpy())  ## Get the VMAE value
                        
                        # if min(VMAE_list) < min_VMAE_epoch:
                        #     min_VMAE_epoch = min(VMAE_list)

                        # if max(VMAE_list) > max_VMAE_epoch:
                        #     max_VMAE_epoch = max(VMAE_list)
                       
                        # Add loss and VMAE for the averagers
                        val_loss_averager.add(loss.item())
                        # val_VMAE_averager.add(VMAE_valid)

                        # Save seg map to tensorboard
                        # if self.args.ref_type == 'Constant':
                        #     index = index.numpy()
                        #     for ind, name_ind in enumerate(index):
                        #         if name_ind in self.VisualSample:
                        #             temp = torch.zeros((3,output.shape[1], output.shape[2]))
                        #             temp[0,:,:] = output[ind,:,:].cpu()
                        #             temp[1,:,:] = label[ind,:,:]
                        #             writer.add_image(self.train_name+'_Pre-Valid-SegProbMap-{}'.format(name_ind), temp, global_step=epoch, dataformats='CHW')
                        # elif self.args.ref_type == 'Reference':
                        #     index = index.numpy()
                        #     for ind, name_ind in enumerate(index):
                        #         if name_ind in self.VisualSample:
                        #             temp = torch.zeros((3,output.shape[1], output.shape[2]))
                        #             temp[0,:,:] = output[ind,:,:].cpu()
                        #             temp[1,:,:] = label[ind,:,:]
                        #             writer.add_image(self.train_name+'_Pre-Train-SegProbMap-{}'.format(name_ind), temp, global_step=epoch, dataformats='CHW')

                # Update the averagers
                val_loss_averager = val_loss_averager.item()
            
                
                # Save loss and VMAE to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Valid-Loss', val_loss_averager, global_step=epoch)


    


                # Print loss and accuracy for this epoch
                print('Epoch {}/{}, Val: Loss={:.6f} '.format(epoch, self.args.pre_max_epoch, val_loss_averager))
            #####    
            # Save model every 10 epochs
            if epoch % 2 == 0:
                self.save_model('pre_epoch_{}'.format(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            # trlog['val_loss'].append(val_loss_averager)
            # trlog['val_VMAE'].append(val_VMAE_averager)#print(train_loss_averager  )
            np.save(osp.join(self.save_path, 'pre_trlog'), trlog)

            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.pre_max_epoch)))

        print('Training Finished! Running Time: {}'.format(timer.measure()))

        # Close tensorboardX writer
        writer.close()



        

                
class DepthTrainLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m-%d_%H-%M")
        self.train_name = args.pre_dataset_name + '_' + time_str + '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 
        log_base_dir = './logs/'  # The base dir for logs
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        pre_base_dir = osp.join(log_base_dir, 'train')    # The base dir for pretrain logs
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        self.save_path = osp.join(pre_base_dir,  self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        basicFile = ['model']        
        for file in basicFile:
            Path = os.path.join(self.save_path, file)
            if not os.path.exists(Path):
                os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'pre' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        self.TBlog_path = os.path.join('./TBlog', self.train_name) # The path to save tensorboard logs
        if not os.path.exists(self.TBlog_path):
            os.makedirs(self.TBlog_path)

        # Set args to be shareable in the class
        self.args = args

        # initialize some parameters
        #self.t0Int = np.arange(0, args.pre_nt * args.pre_dt, args.pre_dt)

        # Build dataset and dataloader
        self.TrainSet = LocDepthTrainSetLoader(args)

        self.Train_Loader = DataLoader(self.TrainSet, batch_size=args.pre_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.ValidSet = LocDepthValidSetLoader(args)

        self.Valid_Loader = DataLoader(self.ValidSet, batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        self.ValidSampleNum = len(self.ValidSet)

        # Set the samples to visualize
        if self.ValidSampleNum > 10:
            self.VisualSampleNum = 10
        else:
            self.VisualSampleNum = self.ValidSampleNum

        self.VisualSample = list(np.random.choice(self.ValidSampleNum, self.VisualSampleNum, replace=False)) # Randomly choose 10 samples to visualize

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
        if self.args.pre_init_weights is not None:
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set loss functions
        if args.loss_type == 'BCE':
            self.criterion = nn.BCELoss()
        elif args.loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif args.loss_type == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif args.loss_type == 'Dice':
            self.criterion = DiceLoss()
        elif args.loss_type == 'BCEWithLogits':
            self.criterion = nn.BCEWithLogitsLoss()

    

        # Set optimizer 
        if args.pre_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.pre_lr)
        elif args.pre_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.pre_lr, momentum=args.pre_custom_momentum, weight_decay=args.pre_custom_weight_decay)

        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, gamma=self.args.pre_gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()


    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.save_path, 'model', name + '.pth'))
    
    def train(self):

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['val_VMAE'] = []        
        trlog['min_loss'] = 1e10
        trlog['min_VMAE'] = 1e10
        trlog['max_VMAE_epoch'] = []
        trlog['min_VMAE_epoch'] = []
        
        # Set the timer
        timer = Timer()  # Timer for the training time

        # Set iteration count to zero
        iter_count = 0
        # Set tensorboardX
        now = datetime.datetime.now()  # Get the current time
        datestr = now.strftime('%Y-%m-%d_%H:%M:%S')  # Get the current time in the format of string
        writer = SummaryWriter(log_dir=self.TBlog_path, comment=datestr)  # Set the tensorboard writer

        # Start training
        for epoch in range(1, self.args.pre_max_epoch + 1):
            
            # Set the model to train mode
            self.model.train()

            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_train = tqdm.tqdm(self.Train_Loader)      

            for i, batch in enumerate(tqdm_train, 1):
                # Update global count number 
                iter_count = iter_count + 1

                input, label = batch
                # 取input的channel个数
                input = input[:, :, :, 0:self.args.In_channels]
                input = np.transpose(input, (0, 3, 1, 2))
                label = np.transpose(label, (0,1))

                ############### 条件判断，是否需要对数据进行处理 ###############
                #可以在这里重复赋值，以便于在这里进行数据的处理
                # if self.args.ref_type == 'Reference':
                #     input = input[0]    
                #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])


                if torch.cuda.is_available():
                    input = input.cuda()
                    label = label.cuda()


                self.optimizer.zero_grad()

                output = self.model(input)
                
                #print(output)
                # Calculate train loss
                loss = self.criterion(output, label)

                # Save loss to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Train-Loss(iter)', loss.item(), global_step=iter_count)                

                # Add loss for the averagers
                train_loss_averager.add(loss.item()) # Add the loss to the averager
                #print(train_loss_averager.item())

                # Print loss till this step
                #tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager.item()))
                tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, loss))
                # Loss backwards and optimizer updates
                loss.backward()
                self.optimizer.step()

            train_loss_averager = train_loss_averager.item()  ## Get the loss value

            # Save loss to tensorboard
            writer.add_scalar(self.train_name+'_Pre-Train-Loss(epoch)', train_loss_averager, global_step=epoch)
            
            print('Epoch {}/{}, Train: Loss={:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager))
            
            # Update learning rate
            self.lr_scheduler.step()

            # Start validation for specific epoch
            if epoch % self.args.pre_val_epoch == 0:
                # Set the model to valid mode
                self.model.eval()

                # Set averager classes to record validation losses and accuracies
                val_loss_averager = Averager()

                total_loss = 0.0
                total_samples = 0

                max_VMAE_epoch = 0
                min_VMAE_epoch = 1e101
                with torch.no_grad():
                    for i, batch in enumerate(self.Valid_Loader, 1):
                        val_input, val_label = batch
                        val_input = val_input[:, :, :, 0:self.args.In_channels]
                        # if self.args.ref_type == 'Reference':

                        #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])
                        val_input = np.transpose(val_input, (0, 3, 1, 2))
                        val_label = np.transpose(val_label, (0, 1))

                        if torch.cuda.is_available():
                            val_input = val_input.cuda()
                            val_label = val_label.cuda()
                    
        

                        output = self.model(val_input)
                        #output = torch.squeeze(output, dim=1)
                        #print(output.shape)
                        # Calculate valid loss
                        loss = self.criterion(output, val_label)

                        # Calculate valid VMAE
                        #APCurve,_ = GetResult(output.cpu().numpy(), self.t0Int, Vrange, threshold=self.args.Predthre)
                        
                        # VMAE_valid, VMAE_list = VMAE(APCurve, velcurve.numpy())  ## Get the VMAE value
                        
                        # if min(VMAE_list) < min_VMAE_epoch:
                        #     min_VMAE_epoch = min(VMAE_list)

                        # if max(VMAE_list) > max_VMAE_epoch:
                        #     max_VMAE_epoch = max(VMAE_list)
                       
                        # Add loss and VMAE for the averagers
                        val_loss_averager.add(loss.item())
                        # val_VMAE_averager.add(VMAE_valid)
                        indices_label = np.argmax(val_label.cpu(), axis=1)
                        indices_output = np.argmax(output.cpu(), axis=1)
                        absolute_differences = torch.abs(indices_label - indices_output)
                        depth_error = torch.sum(absolute_differences)
                        total_loss += depth_error
                        total_samples += absolute_differences.size(0)
                        #print('Total loss: {}, Total samples: {}'.format(total_loss, total_samples))
                        #Save seg map to tensorboard
                    #这段是把每张图片的预测值和真实值画出来    
                        # if self.args.ref_type == 'Constant':
                        #     predictions = torch.zeros((output.shape[0], output.shape[1]))
                        #     true_values = torch.zeros((output.shape[0], output.shape[1]))
                        #     for ind in range(output.shape[0]):
                        #         predictions[ind,:] = output[ind, :].cpu()
                        #         true_values[ind,:] = val_label[ind, :].cpu()
                        #         plt.figure(figsize=(10, 5))
                        #         plt.plot(predictions[ind,:].numpy(), label='Predictions', color='blue')
                        #         plt.plot(true_values[ind,:].numpy(), label='True_val_depth', color='orange')
                        #         plt.title('Predictions vs True,{}'.format(ind))
                        #         plt.legend()
                        #         plt.grid()

                        #         # 保存图像并将其添加到 TensorBoard
                        #         writer.add_figure('Predictions_vs_True_Values-{}'.format(ind), plt.gcf(), global_step=epoch)
                        #         plt.close()

                        if self.args.ref_type == 'Constant':
                            predictions = torch.zeros((output.shape[0], output.shape[1]))
                            true_values = torch.zeros((output.shape[0], output.shape[1]))
                            
                            # 创建多个子图，列数为5，行数根据样本数动态计算
                            num_samples = output.shape[0]
                            num_columns = 5
                            num_rows = (num_samples + num_columns - 1) // num_columns  # 向上取整计算行数
                            
                            fig, axes = plt.subplots(num_rows, num_columns, figsize=(7*num_columns, 5 * num_rows))
                            
                            # 将 axes 扁平化为一维数组（方便索引）
                            axes = axes.flatten()
                            
                            # 在每个子图中绘制每个样本的预测值与真实值
                            for ind in range(num_samples):
                                predictions[ind, :] = output[ind, :].cpu()
                                true_values[ind, :] = val_label[ind, :].cpu()
                                
                                # 在对应的子图中绘制预测值曲线
                                axes[ind].plot(predictions[ind, :].numpy(), label='Predictions', color='blue')
                                
                                # 在对应的子图中绘制真实值曲线
                                axes[ind].plot(true_values[ind, :].numpy(), label='True Values', color='orange')

                                # 设置每个子图的标题和图例
                                axes[ind].set_title(f'Sample {ind}')
                                axes[ind].legend()
                                axes[ind].grid()

                            # 如果多余的子图存在（在列数不足时），隐藏它们
                            for ind in range(num_samples, len(axes)):
                                axes[ind].axis('off')  # 隐藏多余的子图

                            # 调整布局，使子图不重叠
                            plt.tight_layout()

                            # 将图像添加到 TensorBoard
                            writer.add_figure('Predictions_vs_True_Values_Per_Sample', fig, global_step=epoch)
                            
                            # 关闭图形
                            plt.close()    
                    



                    






                # 找到每一行中在第二维度（64维度）的最大值的索引
                # indices_label = np.argmax(val_label.cpu(), axis=1)
                # indices_output = np.argmax(output.cpu(), axis=1)
                # absolute_differences = torch.abs(indices_label - indices_output)
                # # print(output.shape)
                # # print(indices_output.shape)  
                # # print(absolute_differences.shape)   
                # # 将这些索引加起来
                # val_error = torch.sum(absolute_differences)/absolute_differences.shape[0]*0.02246094
                #print('Total loss: {}, Total samples: {}'.format(total_loss, total_samples))
                val_error = total_loss/total_samples*0.08984375
                # Save val-error to tensorboard
                writer.add_scalar(self.train_name+'_val_error', val_error, global_step=epoch)
                print('Epoch {}/{}, Val: Depth_error={:.6f} '.format(epoch, self.args.pre_max_epoch, val_error))

                # Update the averagers
                val_loss_averager = val_loss_averager.item()
               
                # Save loss and VMAE to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Valid-Loss', val_loss_averager, global_step=epoch)


                # Print loss and accuracy for this epoch
                print('Epoch {}/{}, Val: Loss={:.6f} '.format(epoch, self.args.pre_max_epoch, val_loss_averager))
            #####    
            # Save model every 10 epochs
            if epoch % 5 == 0:
                self.save_model('pre_epoch_{}'.format(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            # trlog['val_loss'].append(val_loss_averager)
            # trlog['val_VMAE'].append(val_VMAE_averager)#print(train_loss_averager  )
            np.save(osp.join(self.save_path, 'pre_trlog'), trlog)

            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.pre_max_epoch)))

        print('Training Finished! Running Time: {}'.format(timer.measure()))

        # Close tensorboardX writer
        writer.close()

class DepthTrainLearner_with_epixy(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m-%d_%H-%M")
        self.train_name = args.pre_dataset_name + '_' + time_str + '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 
        log_base_dir = './logs/'  # The base dir for logs
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        pre_base_dir = osp.join(log_base_dir, 'train')    # The base dir for pretrain logs
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        self.save_path = osp.join(pre_base_dir,  self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        basicFile = ['model']        
        for file in basicFile:
            Path = os.path.join(self.save_path, file)
            if not os.path.exists(Path):
                os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'pre' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        self.TBlog_path = os.path.join('./TBlog', self.train_name) # The path to save tensorboard logs
        if not os.path.exists(self.TBlog_path):
            os.makedirs(self.TBlog_path)

        # Set args to be shareable in the class
        self.args = args

        # initialize some parameters
        #self.t0Int = np.arange(0, args.pre_nt * args.pre_dt, args.pre_dt)

        # Build dataset and dataloader
        self.TrainSet = LocDepthTrainSetLoader_epi_xy(args)

        self.Train_Loader = DataLoader(self.TrainSet, batch_size=args.pre_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.ValidSet = LocDepthValidSetLoader_epi_xy(args)

        self.Valid_Loader = DataLoader(self.ValidSet, batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        self.ValidSampleNum = len(self.ValidSet)

        # Set the samples to visualize
        if self.ValidSampleNum > 10:
            self.VisualSampleNum = 10
        else:
            self.VisualSampleNum = self.ValidSampleNum

        self.VisualSample = list(np.random.choice(self.ValidSampleNum, self.VisualSampleNum, replace=False)) # Randomly choose 10 samples to visualize

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
        if self.args.pre_init_weights is not None:
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set loss functions
        if args.loss_type == 'BCE':
            self.criterion = nn.BCELoss()
        elif args.loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif args.loss_type == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif args.loss_type == 'Dice':
            self.criterion = DiceLoss()
        elif args.loss_type == 'BCEWithLogits':
            self.criterion = nn.BCEWithLogitsLoss()

    

        # Set optimizer 
        if args.pre_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.pre_lr)
        elif args.pre_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.pre_lr, momentum=args.pre_custom_momentum, weight_decay=args.pre_custom_weight_decay)

        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, gamma=self.args.pre_gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()


    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.save_path, 'model', name + '.pth'))
    
    def train(self):

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['val_VMAE'] = []        
        trlog['min_loss'] = 1e10
        trlog['min_VMAE'] = 1e10
        trlog['max_VMAE_epoch'] = []
        trlog['min_VMAE_epoch'] = []
        
        # Set the timer
        timer = Timer()  # Timer for the training time

        # Set iteration count to zero
        iter_count = 0
        # Set tensorboardX
        now = datetime.datetime.now()  # Get the current time
        datestr = now.strftime('%Y-%m-%d_%H:%M:%S')  # Get the current time in the format of string
        writer = SummaryWriter(log_dir=self.TBlog_path, comment=datestr)  # Set the tensorboard writer

        # Start training
        for epoch in range(1, self.args.pre_max_epoch + 1):
            
            # Set the model to train mode
            self.model.train()

            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_train = tqdm.tqdm(self.Train_Loader)      

            for i, batch in enumerate(tqdm_train, 1):
                # Update global count number 
                iter_count = iter_count + 1

                input, label, Epicentral_distance, Station_X, Station_Y = batch

                input = np.transpose(input, (0, 3, 1, 2))
                label = np.transpose(label, (0,1))
                Epicentral_distance = np.transpose(Epicentral_distance, (0,1))
                Station_X = np.transpose(Station_X, (0,1))
                Station_Y = np.transpose(Station_Y, (0,1))

                ############### 条件判断，是否需要对数据进行处理 ###############
                #可以在这里重复赋值，以便于在这里进行数据的处理
                # if self.args.ref_type == 'Reference':
                #     input = input[0]    
                #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])


                if torch.cuda.is_available():
                    input = input.cuda()
                    label = label.cuda()
                    Epicentral_distance = Epicentral_distance.cuda()
                    Station_X = Station_X.cuda()
                    Station_Y = Station_Y.cuda()


                self.optimizer.zero_grad()

                output = self.model(input, Epicentral_distance, Station_X, Station_Y)
                
                #print(output)
                # Calculate train loss
                loss = self.criterion(output, label)

                # Save loss to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Train-Loss(iter)', loss.item(), global_step=iter_count)                

                # Add loss for the averagers
                train_loss_averager.add(loss.item()) # Add the loss to the averager
                #print(train_loss_averager.item())

                # Print loss till this step
                #tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager.item()))
                tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, loss))
                # Loss backwards and optimizer updates
                loss.backward()
                self.optimizer.step()

            train_loss_averager = train_loss_averager.item()  ## Get the loss value

            # Save loss to tensorboard
            writer.add_scalar(self.train_name+'_Pre-Train-Loss(epoch)', train_loss_averager, global_step=epoch)
            
            print('Epoch {}/{}, Train: Loss={:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager))
            
            # Update learning rate
            self.lr_scheduler.step()

            # Start validation for specific epoch
            if epoch % self.args.pre_val_epoch == 0:
                # Set the model to valid mode
                self.model.eval()

                # Set averager classes to record validation losses and accuracies
                val_loss_averager = Averager()

                total_loss = 0.0
                total_samples = 0

                max_VMAE_epoch = 0
                min_VMAE_epoch = 1e101
                with torch.no_grad():
                    for i, batch in enumerate(self.Valid_Loader, 1):
                        val_input, val_label, Epicentral_distance, Station_X, Station_Y = batch

                        # if self.args.ref_type == 'Reference':

                        #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])
                        val_input = np.transpose(val_input, (0, 3, 1, 2))
                        val_label = np.transpose(val_label, (0, 1))
                        Epicentral_distance = np.transpose(Epicentral_distance, (0,1))
                        Station_X = np.transpose(Station_X, (0,1))
                        Station_Y = np.transpose(Station_Y, (0,1))


                        if torch.cuda.is_available():
                            val_input = val_input.cuda()
                            val_label = val_label.cuda()
                            Epicentral_distance = Epicentral_distance.cuda()
                            Station_X = Station_X.cuda()
                            Station_Y = Station_Y.cuda()
                    
        

                        output = self.model(val_input, Epicentral_distance, Station_X, Station_Y)
                        #output = torch.squeeze(output, dim=1)
                        #print(output.shape)
                        # Calculate valid loss
                        loss = self.criterion(output, val_label)

                        # Calculate valid VMAE
                        #APCurve,_ = GetResult(output.cpu().numpy(), self.t0Int, Vrange, threshold=self.args.Predthre)
                        
                        # VMAE_valid, VMAE_list = VMAE(APCurve, velcurve.numpy())  ## Get the VMAE value
                        
                        # if min(VMAE_list) < min_VMAE_epoch:
                        #     min_VMAE_epoch = min(VMAE_list)

                        # if max(VMAE_list) > max_VMAE_epoch:
                        #     max_VMAE_epoch = max(VMAE_list)
                       
                        # Add loss and VMAE for the averagers
                        val_loss_averager.add(loss.item())
                        # val_VMAE_averager.add(VMAE_valid)
                        indices_label = np.argmax(val_label.cpu(), axis=1)
                        indices_output = np.argmax(output.cpu(), axis=1)
                        absolute_differences = torch.abs(indices_label - indices_output)
                        depth_error = torch.sum(absolute_differences)
                        total_loss += depth_error
                        total_samples += absolute_differences.size(0)
                        #print('Total loss: {}, Total samples: {}'.format(total_loss, total_samples))
                        #Save seg map to tensorboard
                    #这段是把每张图片的预测值和真实值画出来    
                        # if self.args.ref_type == 'Constant':
                        #     predictions = torch.zeros((output.shape[0], output.shape[1]))
                        #     true_values = torch.zeros((output.shape[0], output.shape[1]))
                        #     for ind in range(output.shape[0]):
                        #         predictions[ind,:] = output[ind, :].cpu()
                        #         true_values[ind,:] = val_label[ind, :].cpu()
                        #         plt.figure(figsize=(10, 5))
                        #         plt.plot(predictions[ind,:].numpy(), label='Predictions', color='blue')
                        #         plt.plot(true_values[ind,:].numpy(), label='True_val_depth', color='orange')
                        #         plt.title('Predictions vs True,{}'.format(ind))
                        #         plt.legend()
                        #         plt.grid()

                        #         # 保存图像并将其添加到 TensorBoard
                        #         writer.add_figure('Predictions_vs_True_Values-{}'.format(ind), plt.gcf(), global_step=epoch)
                        #         plt.close()

                        if self.args.ref_type == 'Constant':
                            predictions = torch.zeros((output.shape[0], output.shape[1]))
                            true_values = torch.zeros((output.shape[0], output.shape[1]))
                            
                            # 创建多个子图，列数为5，行数根据样本数动态计算
                            num_samples = output.shape[0]
                            num_columns = 5
                            num_rows = (num_samples + num_columns - 1) // num_columns  # 向上取整计算行数
                            
                            fig, axes = plt.subplots(num_rows, num_columns, figsize=(7*num_columns, 5 * num_rows))
                            
                            # 将 axes 扁平化为一维数组（方便索引）
                            axes = axes.flatten()
                            
                            # 在每个子图中绘制每个样本的预测值与真实值
                            for ind in range(num_samples):
                                predictions[ind, :] = output[ind, :].cpu()
                                true_values[ind, :] = val_label[ind, :].cpu()
                                
                                # 在对应的子图中绘制预测值曲线
                                axes[ind].plot(predictions[ind, :].numpy(), label='Predictions', color='blue')
                                
                                # 在对应的子图中绘制真实值曲线
                                axes[ind].plot(true_values[ind, :].numpy(), label='True Values', color='orange')

                                # 设置每个子图的标题和图例
                                axes[ind].set_title(f'Sample {ind}')
                                axes[ind].legend()
                                axes[ind].grid()

                            # 如果多余的子图存在（在列数不足时），隐藏它们
                            for ind in range(num_samples, len(axes)):
                                axes[ind].axis('off')  # 隐藏多余的子图

                            # 调整布局，使子图不重叠
                            plt.tight_layout()

                            # 将图像添加到 TensorBoard
                            writer.add_figure('Predictions_vs_True_Values_Per_Sample', fig, global_step=epoch)
                            
                            # 关闭图形
                            plt.close()    
                    



                    






                # 找到每一行中在第二维度（64维度）的最大值的索引
                # indices_label = np.argmax(val_label.cpu(), axis=1)
                # indices_output = np.argmax(output.cpu(), axis=1)
                # absolute_differences = torch.abs(indices_label - indices_output)
                # # print(output.shape)
                # # print(indices_output.shape)  
                # # print(absolute_differences.shape)   
                # # 将这些索引加起来
                # val_error = torch.sum(absolute_differences)/absolute_differences.shape[0]*0.02246094
                #print('Total loss: {}, Total samples: {}'.format(total_loss, total_samples))
                val_error = total_loss/total_samples*0.08984375
                # Save val-error to tensorboard
                writer.add_scalar(self.train_name+'_val_error', val_error, global_step=epoch)
                print('Epoch {}/{}, Val: Depth_error={:.6f} '.format(epoch, self.args.pre_max_epoch, val_error))

                # Update the averagers
                val_loss_averager = val_loss_averager.item()
               
                # Save loss and VMAE to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Valid-Loss', val_loss_averager, global_step=epoch)


                # Print loss and accuracy for this epoch
                print('Epoch {}/{}, Val: Loss={:.6f} '.format(epoch, self.args.pre_max_epoch, val_loss_averager))
            #####    
            # Save model every 10 epochs
            if epoch % 5 == 0:
                self.save_model('pre_epoch_{}'.format(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            # trlog['val_loss'].append(val_loss_averager)
            # trlog['val_VMAE'].append(val_VMAE_averager)#print(train_loss_averager  )
            np.save(osp.join(self.save_path, 'pre_trlog'), trlog)

            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.pre_max_epoch)))

        print('Training Finished! Running Time: {}'.format(timer.measure()))

        # Close tensorboardX writer
        writer.close()

        
class SingleStationDepthTrainLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m-%d_%H-%M")
        self.train_name = args.pre_dataset_name + '_' + time_str + '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 
        log_base_dir = './logs/'  # The base dir for logs
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        pre_base_dir = osp.join(log_base_dir, 'train')    # The base dir for pretrain logs
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        self.save_path = osp.join(pre_base_dir,  self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        basicFile = ['model']        
        for file in basicFile:
            Path = os.path.join(self.save_path, file)
            if not os.path.exists(Path):
                os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'pre' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        self.TBlog_path = os.path.join('./TBlog', self.train_name) # The path to save tensorboard logs
        if not os.path.exists(self.TBlog_path):
            os.makedirs(self.TBlog_path)

        # Set args to be shareable in the class
        self.args = args

        # initialize some parameters
        #self.t0Int = np.arange(0, args.pre_nt * args.pre_dt, args.pre_dt)

        # Build dataset and dataloader
        self.TrainSet = LocSingleStationDepthTrainSetLoader(args)

        self.Train_Loader = DataLoader(self.TrainSet, batch_size=args.pre_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.ValidSet = LocSingleStationDepthValidSetLoader(args)

        self.Valid_Loader = DataLoader(self.ValidSet, batch_size=100, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        self.ValidSampleNum = len(self.ValidSet)

        # Set the samples to visualize
        if self.ValidSampleNum > 10:
            self.VisualSampleNum = 10
        else:
            self.VisualSampleNum = self.ValidSampleNum

        self.VisualSample = list(np.random.choice(self.ValidSampleNum, self.VisualSampleNum, replace=False)) # Randomly choose 10 samples to visualize

        # Build pretrain model
        if args.model_type == 'VGG16Attention':
            self.model = VGG16Attention(args.In_channels,args.Out_channels)
        elif args.model_type == 'UNet':
            self.model = UNet(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth':
            self.model = VGGDepth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1024':
            self.model = VGGDepth1024(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepthSingle':
            self.model = VGGDepthSingle(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepthSingle256':
            self.model = VGGDepthSingle256(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGG19Depth':
            self.model = VGG19Depth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepthSingle256_epi':
            self.model = VGGDepthSingle256_epi(args.In_channels,args.Out_channels)
    
        
        # Load pretrain model
        if self.args.pre_init_weights is not None:
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set loss functions
        if args.loss_type == 'BCE':
            self.criterion = nn.BCELoss()
        elif args.loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif args.loss_type == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif args.loss_type == 'Dice':
            self.criterion = DiceLoss()
        elif args.loss_type == 'BCEWithLogits':
            self.criterion = nn.BCEWithLogitsLoss()

    

        # Set optimizer 
        if args.pre_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.pre_lr)
        elif args.pre_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.pre_lr, momentum=args.pre_custom_momentum, weight_decay=args.pre_custom_weight_decay)

        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, gamma=self.args.pre_gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()


    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.save_path, 'model', name + '.pth'))
    
    def train(self):

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['val_VMAE'] = []        
        trlog['min_loss'] = 1e10
        trlog['min_VMAE'] = 1e10
        trlog['max_VMAE_epoch'] = []
        trlog['min_VMAE_epoch'] = []
        
        # Set the timer
        timer = Timer()  # Timer for the training time

        # Set iteration count to zero
        iter_count = 0
        # Set tensorboardX
        now = datetime.datetime.now()  # Get the current time
        datestr = now.strftime('%Y-%m-%d_%H:%M:%S')  # Get the current time in the format of string
        writer = SummaryWriter(log_dir=self.TBlog_path, comment=datestr)  # Set the tensorboard writer

        # Start training
        for epoch in range(1, self.args.pre_max_epoch + 1):
            
            # Set the model to train mode
            self.model.train()

            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_train = tqdm.tqdm(self.Train_Loader)      

            for i, batch in enumerate(tqdm_train, 1):
                # Update global count number 
                iter_count = iter_count + 1

                input, label = batch

                input = np.transpose(input, (0, 1, 3, 2))
                label = np.transpose(label, (0,1))

                ############### 条件判断，是否需要对数据进行处理 ###############
                #可以在这里重复赋值，以便于在这里进行数据的处理
                # if self.args.ref_type == 'Reference':
                #     input = input[0]    
                #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])


                if torch.cuda.is_available():
                    input = input.cuda()
                    label = label.cuda()


                self.optimizer.zero_grad()

                output = self.model(input)
                
                #print(output)
                # Calculate train loss
                loss = self.criterion(output, label)

                # Save loss to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Train-Loss(iter)', loss.item(), global_step=iter_count)                

                # Add loss for the averagers
                train_loss_averager.add(loss.item()) # Add the loss to the averager
                #print(train_loss_averager.item())

                # Print loss till this step
                #tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager.item()))
                tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, loss))
                # Loss backwards and optimizer updates
                loss.backward()
                self.optimizer.step()

            train_loss_averager = train_loss_averager.item()  ## Get the loss value

            # Save loss to tensorboard
            writer.add_scalar(self.train_name+'_Pre-Train-Loss(epoch)', train_loss_averager, global_step=epoch)
            
            print('Epoch {}/{}, Train: Loss={:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager))
            
            # Update learning rate
            self.lr_scheduler.step()

            # Start validation for specific epoch
            if epoch % self.args.pre_val_epoch == 0:
                # Set the model to valid mode
                self.model.eval()

                # Set averager classes to record validation losses and accuracies
                val_loss_averager = Averager()

                total_loss = 0.0
                total_samples = 0

                max_VMAE_epoch = 0
                min_VMAE_epoch = 1e101
                with torch.no_grad():
                    for i, batch in enumerate(self.Valid_Loader, 1):
                        val_input, val_label = batch

                        # if self.args.ref_type == 'Reference':

                        #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])
                        val_input = np.transpose(val_input, (0, 1, 3, 2))
                        #print(val_input.shape)
                        val_label = np.transpose(val_label, (0, 1))

                        if torch.cuda.is_available():
                            val_input = val_input.cuda()
                            val_label = val_label.cuda()
                    
        

                        output = self.model(val_input)
                        #output = torch.squeeze(output, dim=1)

                        # Calculate valid loss
                        loss = self.criterion(output, val_label)

                        # Calculate valid VMAE
                        #APCurve,_ = GetResult(output.cpu().numpy(), self.t0Int, Vrange, threshold=self.args.Predthre)
                        
                        # VMAE_valid, VMAE_list = VMAE(APCurve, velcurve.numpy())  ## Get the VMAE value
                        
                        # if min(VMAE_list) < min_VMAE_epoch:
                        #     min_VMAE_epoch = min(VMAE_list)

                        # if max(VMAE_list) > max_VMAE_epoch:
                        #     max_VMAE_epoch = max(VMAE_list)
                       
                        # Add loss and VMAE for the averagers
                        val_loss_averager.add(loss.item())
                        # val_VMAE_averager.add(VMAE_valid)
                        indices_label = np.argmax(val_label.cpu(), axis=1)
                        indices_output = np.argmax(output.cpu(), axis=1)
                        absolute_differences = torch.abs(indices_label - indices_output)
                        depth_error = torch.sum(absolute_differences)
                        total_loss += depth_error
                        total_samples += absolute_differences.size(0)
                        # #Save seg map to tensorboard
                        # if self.args.ref_type == 'Constant':
                        #     predictions = torch.zeros((output.shape[0], output.shape[1]))
                        #     true_values = torch.zeros((output.shape[0], output.shape[1]))
                        #     for ind in range(output.shape[0]):
                        #         predictions[ind,:] = output[ind, :].cpu()
                        #         true_values[ind,:] = val_label[ind, :].cpu()
                        #         plt.figure(figsize=(10, 5))
                        #         plt.plot(predictions[ind,:].numpy(), label='Predictions', color='blue')
                        #         plt.plot(true_values[ind,:].numpy(), label='True_val_depth', color='orange')
                        #         plt.title('Predictions vs True')
                        #         plt.legend()
                        #         plt.grid()

                        #         # 保存图像并将其添加到 TensorBoard
                        #         writer.add_figure('Predictions_vs_True_Values-{}'.format(ind), plt.gcf(), global_step=epoch)
                        #         plt.close()

                                
                                #writer.add_image(self.train_name+'_Pre-Valid-SegProbMap-{}', temp, global_step=epoch, dataformats='CH')
                                #writer.add_scalar(self.train_name+'Vaild_results(epoch)', temp, global_step=epoch)    
                        if self.args.ref_type == 'Constant':
                            predictions = torch.zeros((output.shape[0], output.shape[1]))
                            true_values = torch.zeros((output.shape[0], output.shape[1]))
                            
                            # 创建多个子图，列数为5，行数根据样本数动态计算
                            num_samples = output.shape[0]
                            num_columns = 5
                            num_rows = (num_samples + num_columns - 1) // num_columns  # 向上取整计算行数
                            
                            fig, axes = plt.subplots(num_rows, num_columns, figsize=(7*num_columns, 5 * num_rows))
                            
                            # 将 axes 扁平化为一维数组（方便索引）
                            axes = axes.flatten()
                            
                            # 在每个子图中绘制每个样本的预测值与真实值
                            for ind in range(num_samples):
                                predictions[ind, :] = output[ind, :].cpu()
                                true_values[ind, :] = val_label[ind, :].cpu()
                                
                                # 在对应的子图中绘制预测值曲线
                                axes[ind].plot(predictions[ind, :].numpy(), label='Predictions', color='blue')
                                
                                # 在对应的子图中绘制真实值曲线
                                axes[ind].plot(true_values[ind, :].numpy(), label='True Values', color='orange')

                                # 设置每个子图的标题和图例
                                axes[ind].set_title(f'Sample {ind}')
                                axes[ind].legend()
                                axes[ind].grid()

                            # 如果多余的子图存在（在列数不足时），隐藏它们
                            for ind in range(num_samples, len(axes)):
                                axes[ind].axis('off')  # 隐藏多余的子图

                            # 调整布局，使子图不重叠
                            plt.tight_layout()

                            # 将图像添加到 TensorBoard
                            writer.add_figure('Predictions_vs_True_Values_Per_Sample', fig, global_step=epoch)
                            
                            # 关闭图形
                            plt.close()    

                # 找到每一行中在第二维度（64维度）的最大值的索引
                # indices_label = np.argmax(val_label.cpu(), axis=1)
                # indices_output = np.argmax(output.cpu(), axis=1)
                # absolute_differences = torch.abs(indices_label - indices_output)
                # # 将这些索引加起来
                # val_error = torch.sum(absolute_differences)/output.shape[0]*0.2265625
                val_error = total_loss/total_samples*0.08984375
                # Save val-error to tensorboard
                writer.add_scalar(self.train_name+'_val_error', val_error, global_step=epoch)
                print('Epoch {}/{}, Val: Depth_error={:.6f} '.format(epoch, self.args.pre_max_epoch, val_error))
                # Update the averagers
                val_loss_averager = val_loss_averager.item()
               
                # Save loss and VMAE to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Valid-Loss', val_loss_averager, global_step=epoch)


                # Print loss and accuracy for this epoch
                print('Epoch {}/{}, Val: Loss={:.6f} '.format(epoch, self.args.pre_max_epoch, val_loss_averager))
            #####    
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('pre_epoch_{}'.format(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            # trlog['val_loss'].append(val_loss_averager)
            # trlog['val_VMAE'].append(val_VMAE_averager)#print(train_loss_averager  )
            np.save(osp.join(self.save_path, 'pre_trlog'), trlog)

            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.pre_max_epoch)))

        print('Training Finished! Running Time: {}'.format(timer.measure()))

        # Close tensorboardX writer
        writer.close()
        
class SingleStation12DepthTrainLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m-%d_%H-%M")
        self.train_name = args.pre_dataset_name + '_' + time_str + '_lr' + str(args.pre_lr) + '_batch' + str(args.pre_batch_size) 
        log_base_dir = './logs/'  # The base dir for logs
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        pre_base_dir = osp.join(log_base_dir, 'train')    # The base dir for pretrain logs
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        self.save_path = osp.join(pre_base_dir,  self.train_name)
        if not osp.exists(self.save_path):
            os.mkdir(self.save_path)

        basicFile = ['model']        
        for file in basicFile:
            Path = os.path.join(self.save_path, file)
            if not os.path.exists(Path):
                os.makedirs(Path)
    
        with open(osp.join(self.save_path, 'args.txt'), 'w') as f:
            for arg in vars(args):
                if 'pre' in arg:
                    f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")

        self.TBlog_path = os.path.join('./TBlog', self.train_name) # The path to save tensorboard logs
        if not os.path.exists(self.TBlog_path):
            os.makedirs(self.TBlog_path)

        # Set args to be shareable in the class
        self.args = args

        # initialize some parameters
        #self.t0Int = np.arange(0, args.pre_nt * args.pre_dt, args.pre_dt)

        # Build dataset and dataloader
        self.TrainSet = LocSingleStation12DepthTrainSetLoader(args)

        self.Train_Loader = DataLoader(self.TrainSet, batch_size=args.pre_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.ValidSet = LocSingleStation12DepthValidSetLoader(args)

        self.Valid_Loader = DataLoader(self.ValidSet, batch_size=200, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        self.ValidSampleNum = len(self.ValidSet)

        # Set the samples to visualize
        if self.ValidSampleNum > 10:
            self.VisualSampleNum = 10
        else:
            self.VisualSampleNum = self.ValidSampleNum

        self.VisualSample = list(np.random.choice(self.ValidSampleNum, self.VisualSampleNum, replace=False)) # Randomly choose 10 samples to visualize

        # Build pretrain model
        if args.model_type == 'VGG16Attention':
            self.model = VGG16Attention(args.In_channels,args.Out_channels)
        elif args.model_type == 'UNet':
            self.model = UNet(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth':
            self.model = VGGDepth(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepth1024':
            self.model = VGGDepth1024(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGGDepthSingle':
            self.model = VGGDepthSingle(args.In_channels,args.Out_channels)
        elif args.model_type == 'VGG19Depth':
            self.model = VGG19Depth(args.In_channels,args.Out_channels)
    
        
        # Load pretrain model
        if self.args.pre_init_weights is not None:
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set loss functions
        if args.loss_type == 'BCE':
            self.criterion = nn.BCELoss()
        elif args.loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif args.loss_type == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif args.loss_type == 'Dice':
            self.criterion = DiceLoss()
        elif args.loss_type == 'BCEWithLogits':
            self.criterion = nn.BCEWithLogitsLoss()

    

        # Set optimizer 
        if args.pre_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.pre_lr)
        elif args.pre_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.pre_lr, momentum=args.pre_custom_momentum, weight_decay=args.pre_custom_weight_decay)

        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, gamma=self.args.pre_gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()


    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), osp.join(self.save_path, 'model', name + '.pth'))
    
    def train(self):

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['val_VMAE'] = []        
        trlog['min_loss'] = 1e10
        trlog['min_VMAE'] = 1e10
        trlog['max_VMAE_epoch'] = []
        trlog['min_VMAE_epoch'] = []
        
        # Set the timer
        timer = Timer()  # Timer for the training time

        # Set iteration count to zero
        iter_count = 0
        # Set tensorboardX
        now = datetime.datetime.now()  # Get the current time
        datestr = now.strftime('%Y-%m-%d_%H:%M:%S')  # Get the current time in the format of string
        writer = SummaryWriter(log_dir=self.TBlog_path, comment=datestr)  # Set the tensorboard writer

        # Start training
        for epoch in range(1, self.args.pre_max_epoch + 1):
            
            # Set the model to train mode
            self.model.train()

            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_train = tqdm.tqdm(self.Train_Loader)      

            for i, batch in enumerate(tqdm_train, 1):
                # Update global count number 
                iter_count = iter_count + 1

                input, label = batch

                input = np.transpose(input, (0, 1, 3, 2))
                label = np.transpose(label, (0,1))

                ############### 条件判断，是否需要对数据进行处理 ###############
                #可以在这里重复赋值，以便于在这里进行数据的处理
                # if self.args.ref_type == 'Reference':
                #     input = input[0]    
                #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])


                if torch.cuda.is_available():
                    input = input.cuda()
                    label = label.cuda()


                self.optimizer.zero_grad()

                output = self.model(input)
                
                #print(output)
                # Calculate train loss
                loss = self.criterion(output, label)

                # Save loss to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Train-Loss(iter)', loss.item(), global_step=iter_count)                

                # Add loss for the averagers
                train_loss_averager.add(loss.item()) # Add the loss to the averager
                #print(train_loss_averager.item())

                # Print loss till this step
                #tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager.item()))
                tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, loss))
                # Loss backwards and optimizer updates
                loss.backward()
                self.optimizer.step()

            train_loss_averager = train_loss_averager.item()  ## Get the loss value

            # Save loss to tensorboard
            writer.add_scalar(self.train_name+'_Pre-Train-Loss(epoch)', train_loss_averager, global_step=epoch)
            
            print('Epoch {}/{}, Train: Loss={:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager))
            
            # Update learning rate
            self.lr_scheduler.step()

            # Start validation for specific epoch
            if epoch % self.args.pre_val_epoch == 0:
                # Set the model to valid mode
                self.model.eval()

                # Set averager classes to record validation losses and accuracies
                val_loss_averager = Averager()


                max_VMAE_epoch = 0
                min_VMAE_epoch = 1e101
                with torch.no_grad():
                    for i, batch in enumerate(self.Valid_Loader, 1):
                        val_input, val_label = batch

                        # if self.args.ref_type == 'Reference':

                        #     label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])
                        val_input = np.transpose(val_input, (0, 1, 3, 2))
                        val_label = np.transpose(val_label, (0, 1))

                        if torch.cuda.is_available():
                            val_input = val_input.cuda()
                            val_label = val_label.cuda()
                    
        

                        output = self.model(val_input)
                        #output = torch.squeeze(output, dim=1)

                        # Calculate valid loss
                        loss = self.criterion(output, val_label)

                        # Calculate valid VMAE
                        #APCurve,_ = GetResult(output.cpu().numpy(), self.t0Int, Vrange, threshold=self.args.Predthre)
                        
                        # VMAE_valid, VMAE_list = VMAE(APCurve, velcurve.numpy())  ## Get the VMAE value
                        
                        # if min(VMAE_list) < min_VMAE_epoch:
                        #     min_VMAE_epoch = min(VMAE_list)

                        # if max(VMAE_list) > max_VMAE_epoch:
                        #     max_VMAE_epoch = max(VMAE_list)
                       
                        # Add loss and VMAE for the averagers
                        val_loss_averager.add(loss.item())
                        # val_VMAE_averager.add(VMAE_valid)

                        #Save seg map to tensorboard
                        if self.args.ref_type == 'Constant':
                            predictions = torch.zeros((output.shape[0], output.shape[1]))
                            true_values = torch.zeros((output.shape[0], output.shape[1]))
                            for ind in range(output.shape[0]):
                                predictions[ind,:] = output[ind, :].cpu()
                                true_values[ind,:] = val_label[ind, :].cpu()
                                plt.figure(figsize=(10, 5))
                                plt.plot(predictions[ind,:].numpy(), label='Predictions', color='blue')
                                plt.plot(true_values[ind,:].numpy(), label='True_val_depth', color='orange')
                                plt.title('Predictions vs True')
                                plt.legend()
                                plt.grid()

                                # 保存图像并将其添加到 TensorBoard
                                writer.add_figure('Predictions_vs_True_Values-{}'.format(ind), plt.gcf(), global_step=epoch)
                                plt.close()

                                
                                #writer.add_image(self.train_name+'_Pre-Valid-SegProbMap-{}', temp, global_step=epoch, dataformats='CH')
                                #writer.add_scalar(self.train_name+'Vaild_results(epoch)', temp, global_step=epoch)    
                        
                        elif self.args.ref_type == 'Reference':
                            index = index.numpy()
                            for ind, name_ind in enumerate(index):
                                if name_ind in self.VisualSample:
                                    temp = torch.zeros((3,output.shape[1], output.shape[2]))
                                    temp[0,:,:] = output[ind,:,:].cpu()
                                    temp[1,:,:] = label[ind,:,:]
                                    writer.add_image(self.train_name+'_Pre-Train-SegProbMap-{}'.format(name_ind), temp, global_step=epoch, dataformats='CHW')

                # 找到每一行中在第二维度（64维度）的最大值的索引
                indices_label = np.argmax(val_label.cpu(), axis=1)
                indices_output = np.argmax(output.cpu(), axis=1)
                absolute_differences = torch.abs(indices_label - indices_output)
                # 将这些索引加起来
                val_error = torch.sum(absolute_differences)/output.shape[0]*0.2265
                # Save val-error to tensorboard
                writer.add_scalar(self.train_name+'_val_error', val_error, global_step=epoch)

                # Update the averagers
                val_loss_averager = val_loss_averager.item()
               
                # Save loss and VMAE to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Valid-Loss', val_loss_averager, global_step=epoch)


                # Print loss and accuracy for this epoch
                print('Epoch {}/{}, Val: Loss={:.6f} '.format(epoch, self.args.pre_max_epoch, val_loss_averager))
            #####    
            # Save model every 10 epochs
            if epoch % 2 == 0:
                self.save_model('pre_epoch_{}'.format(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            # trlog['val_loss'].append(val_loss_averager)
            # trlog['val_VMAE'].append(val_VMAE_averager)#print(train_loss_averager  )
            np.save(osp.join(self.save_path, 'pre_trlog'), trlog)

            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.pre_max_epoch)))

        print('Training Finished! Running Time: {}'.format(timer.measure()))

        # Close tensorboardX writer
        writer.close()
                