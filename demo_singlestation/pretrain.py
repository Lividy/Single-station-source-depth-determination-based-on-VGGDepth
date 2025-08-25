
""" Trainer for pretrain phase. """
import os.path as osp
import os
import numpy as np
import tqdm
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader.dataset_loader import TrainSetLoader, ValidSetLoader

from models.MIFNet import MIFNet
from models.MFFVPNet import MultiFeatureFusionVelPickNet
from loss.loss import Dice_Loss, Focal_Loss
from utils.PastProcess import GetResult, GetResult_new
from utils.metrics import VMAE
from utils.misc import Averager, Timer
from tensorboardX import SummaryWriter
# from utils.losses import FocalLoss,CE_DiceLoss,LovaszSoftmax

class PreTrainLearner(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%m-%d_%H-%M")
        self.train_name = args.pre_dataset_name + '_' + args.pre_dataset_ver + '_' + args.model_type + '_' + args.ref_type + '_' + time_str

        log_base_dir = './logs/'  # The base dir for logs
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)

        pre_base_dir = osp.join(log_base_dir, 'pre')    # The base dir for pretrain logs
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)

        self.save_path = osp.join(pre_base_dir, self.train_name)
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
        self.t0Int = np.arange(0, args.pre_nt * args.pre_dt, args.pre_dt)

        # Build dataset and dataloader
        self.TrainSet = TrainSetLoader(args)

        self.Train_Loader = DataLoader(self.TrainSet, batch_size=args.pre_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        self.ValidSet = ValidSetLoader(args)

        self.Valid_Loader = DataLoader(self.ValidSet, batch_size=args.pre_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        self.ValidSampleNum = len(self.ValidSet)

        # Set the samples to visualize
        if self.ValidSampleNum > 10:
            self.VisualSampleNum = 10
        else:
            self.VisualSampleNum = self.ValidSampleNum

        self.VisualSample = list(np.random.choice(self.ValidSampleNum, self.VisualSampleNum, replace=False))

        # Build pretrain model
        if args.model_type == 'MIFNet':
            self.model = MIFNet(self.t0Int, args.resize)
        else:
            self.model = MultiFeatureFusionVelPickNet(self.t0Int, resize=args.resize, NetType=args.model_type)
        
        # Load pretrain model
        if self.args.pre_init_weights is not None:
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']            
            self.model.load_state_dict(pretrained_dict)
        
        # Set loss functions
        if args.loss_type == 'BCE':
            self.criterion = nn.BCELoss()
        elif args.loss_type == 'Dice':
            self.criterion = Dice_Loss()
        elif args.loss_type == 'Focal':
            self.criterion = Focal_Loss()

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
        timer = Timer()

        # Set iteration count to zero
        iter_count = 0
        # Set tensorboardX
        now = datetime.datetime.now()
        datestr = now.strftime('%Y-%m-%d_%H:%M:%S')
        writer = SummaryWriter(log_dir=self.TBlog_path, comment=datestr)

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

                pwr, label, velcurve, Vrange, LSG, RefVel = batch

                if self.args.ref_type == 'Reference':
                    pwr = pwr.reshape(pwr.shape[0]*pwr.shape[1],pwr.shape[2],pwr.shape[3],pwr.shape[4])
                    label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])
                    velcurve = velcurve.reshape(velcurve.shape[0]*velcurve.shape[1],velcurve.shape[2],velcurve.shape[3])
                    Vrange = Vrange.reshape(Vrange.shape[0]*Vrange.shape[1],Vrange.shape[2])
                    LSG = LSG.reshape(LSG.shape[0]*LSG.shape[1],LSG.shape[2],LSG.shape[3],LSG.shape[4])
                    RefVel = RefVel.reshape(RefVel.shape[0]*RefVel.shape[1],RefVel.shape[2],RefVel.shape[3])

                if torch.cuda.is_available():
                    pwr = pwr.cuda()
                    LSG = LSG.cuda()
                    label = label.cuda()
                    RefVel = RefVel.numpy()
                    Vrange = Vrange.numpy()

                self.optimizer.zero_grad()

                output = self.model(pwr, LSG, RefVel, Vrange)
                output = torch.squeeze(output, dim=1)

                # Calculate train loss
                loss = self.criterion(output, label)

                # Save loss to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Train-Loss(iter)', loss.item(), global_step=iter_count)                

                # Add loss for the averagers
                train_loss_averager.add(loss.item())

                # Print loss till this step
                tqdm_train.set_description('Epoch {}/{}, Loss {:.6f}'.format(epoch, self.args.pre_max_epoch, train_loss_averager.item()))

                # Loss backwards and optimizer updates
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()

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
                val_VMAE_averager = Averager()

                max_VMAE_epoch = 0
                min_VMAE_epoch = 1e10

                with torch.no_grad():
                    for i, batch in enumerate(self.Valid_Loader, 1):
                        pwr, label, velcurve, Vrange, LSG, RefVel, index = batch

                        if self.args.ref_type == 'Reference':
                            pwr = pwr.reshape(pwr.shape[0]*pwr.shape[1],pwr.shape[2],pwr.shape[3],pwr.shape[4])
                            label = label.reshape(label.shape[0]*label.shape[1],label.shape[2],label.shape[3])
                            velcurve = velcurve.reshape(velcurve.shape[0]*velcurve.shape[1],velcurve.shape[2],velcurve.shape[3])
                            Vrange = Vrange.reshape(Vrange.shape[0]*Vrange.shape[1],Vrange.shape[2])
                            LSG = LSG.reshape(LSG.shape[0]*LSG.shape[1],LSG.shape[2],LSG.shape[3],LSG.shape[4])
                            RefVel = RefVel.reshape(RefVel.shape[0]*RefVel.shape[1],RefVel.shape[2],RefVel.shape[3])

                        if torch.cuda.is_available():
                            pwr = pwr.cuda()
                            LSG = LSG.cuda()
                            label = label.cuda()
                            RefVel = RefVel.numpy()
                            Vrange = Vrange.numpy()

                        output = self.model(pwr, LSG, RefVel, Vrange)
                        output = torch.squeeze(output, dim=1)

                        # Calculate valid loss
                        loss = self.criterion(output, label)

                        # Calculate valid VMAE
                        APCurve,_ = GetResult(output.cpu().numpy(), self.t0Int, Vrange, threshold=self.args.Predthre)
                        
                        VMAE_valid, VMAE_list = VMAE(APCurve, velcurve.numpy())
                        
                        if min(VMAE_list) < min_VMAE_epoch:
                            min_VMAE_epoch = min(VMAE_list)

                        if max(VMAE_list) > max_VMAE_epoch:
                            max_VMAE_epoch = max(VMAE_list)
                       
                        # Add loss and VMAE for the averagers
                        val_loss_averager.add(loss.item())
                        val_VMAE_averager.add(VMAE_valid)

                        # Save seg map to tensorboard
                        if self.args.ref_type == 'Constant':
                            index = index.numpy()
                            for ind, name_ind in enumerate(index):
                                if name_ind in self.VisualSample:
                                    temp = torch.zeros((3,output.shape[1], output.shape[2]))
                                    temp[0,:,:] = output[ind,:,:].cpu()
                                    temp[1,:,:] = label[ind,:,:]
                                    writer.add_image(self.train_name+'_Pre-Valid-SegProbMap-{}'.format(name_ind), temp, global_step=epoch, dataformats='CHW')
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
                val_VMAE_averager = val_VMAE_averager.item()
                
                # Save loss and VMAE to tensorboard
                writer.add_scalar(self.train_name+'_Pre-Valid-Loss', val_loss_averager, global_step=epoch)
                writer.add_scalar(self.train_name+'_Pre-Valid-VMAE', val_VMAE_averager, global_step=epoch)
                writer.add_scalar(self.train_name+'_Pre-Valid-min_VMAE_epoch', min_VMAE_epoch, global_step=epoch)
                writer.add_scalar(self.train_name+'_Pre-Valid-max_VMAE_epoch', max_VMAE_epoch, global_step=epoch)

                # Save loss and VMAE to trlog, and save model
                if self.args.pre_early_stop_type == 'loss':
                    if val_loss_averager < trlog['min_loss']:
                        trlog['min_loss'] = val_loss_averager
                        # self.save_model('pre_MinLoss_epoch_{}'.format(epoch))
                        EarlyStopCount = 0
                    # Early stopping
                    else:
                        EarlyStopCount = EarlyStopCount + 1
                        if EarlyStopCount >= self.args.pre_early_stop:
                            print('Early Stop at Epoch {}'.format(epoch))
                            break

                    if val_VMAE_averager < trlog['min_VMAE']:
                        trlog['min_VMAE'] = val_VMAE_averager

                elif self.args.pre_early_stop_type == 'VMAE':
                    if val_VMAE_averager < trlog['min_VMAE']:
                        trlog['min_VMAE'] = val_VMAE_averager
                        # self.save_model('pre_MinVMAE_epoch_{}'.format(epoch))
                        EarlyStopCount = 0
                    # Early stopping
                    else:
                        EarlyStopCount = EarlyStopCount + 1
                        if EarlyStopCount >= self.args.pre_early_stop:
                            print('Early Stop at Epoch {}'.format(epoch))
                            break

                    if val_loss_averager < trlog['min_loss']:
                        trlog['min_loss'] = val_loss_averager

                trlog['max_VMAE_epoch'].append(max_VMAE_epoch)
                trlog['min_VMAE_epoch'].append(min_VMAE_epoch)

                # Print loss and accuracy for this epoch
                print('Epoch {}/{}, Val: Loss=     {:.6f} VMAE=     {:.4f} Min Loss=     {:.6f} Min VMAE=     {:.4f} max_VMAE_epoch=     {:.4f} min_VMAE_epoch=     {:.4f}'.format(epoch, self.args.pre_max_epoch, \
                    val_loss_averager, val_VMAE_averager, trlog['min_loss'], trlog['min_VMAE'], max_VMAE_epoch, min_VMAE_epoch))
                
            # Save model every 10 epochs
            if epoch % 1 == 0:
                self.save_model('pre_epoch_{}'.format(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_VMAE'].append(val_VMAE_averager)

            # Save the logs
            np.save(osp.join(self.save_path, 'pre_trlog'), trlog)

            print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.pre_max_epoch)))

        print('Training Finished! Running Time: {}'.format(timer.measure()))

        # Close tensorboardX writer
        writer.close()
                
