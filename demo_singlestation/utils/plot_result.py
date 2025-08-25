
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import torch

def Plot_Loss(Loss_dict, SavePath=None, figsize=[5,5], mode='diff'):
    Colors = ['blue', 'red', 'green', 'black', 'darkorange', 'magenta', 'brown', 'cyan', 'yellow']

    fig, ax = plt.subplots(figsize=figsize)
    for idx, (filename, file_dict) in enumerate(Loss_dict.items()):
        train_loss_name = list(file_dict.keys())[0]
        valid_loss_name = list(file_dict.keys())[1]
        train_loss = file_dict[train_loss_name]
        valid_loss = file_dict[valid_loss_name]

        if mode == 'same':
            ax.plot(range(len(train_loss)), train_loss, label=train_loss_name, c=Colors[idx], linewidth=2,alpha=0.8)
            ax.plot(range(len(valid_loss)), valid_loss, label=valid_loss_name, c=Colors[idx], linewidth=2,alpha=0.8, linestyle='--')
        elif mode == 'diff':
            ax.plot(range(len(train_loss)), train_loss, label=train_loss_name, c=Colors[idx*2], linewidth=2,alpha=0.8)
            ax.plot(range(len(valid_loss)), valid_loss, label=valid_loss_name, c=Colors[idx*2+1], linewidth=2,alpha=0.8, linestyle='--')

    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Loss', fontsize=20)
    ax.set_yscale('log', base=10)
    plt.legend(loc=1, fontsize=12)
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    ax.grid(True)
    plt.show()
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=300, bbox_inches='tight')


def Plot_VMAE(VMAE_dict, SavePath=None, figsize=[5,5]):
    Colors = ['blue', 'red', 'green', 'black', 'darkorange', 'magenta', 'brown', 'cyan', 'yellow']

    fig, ax = plt.subplots(figsize=figsize)
    for idx, (filename, file_dict) in enumerate(VMAE_dict.items()):
        valid_VMAE_name = list(file_dict.keys())[0]
        max_VMAE_name = list(file_dict.keys())[1]
        valid_VMAE = file_dict[valid_VMAE_name]
        max_VMAE = file_dict[max_VMAE_name]

        ax.plot(range(len(valid_VMAE)), valid_VMAE, label=valid_VMAE_name, c=Colors[idx*2], linewidth=2,alpha=0.8)
           
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('VMAE', fontsize=20)
    
    plt.legend(loc=1, fontsize=12)
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    ax.grid(True)
    plt.show()
    if SavePath is None:
        plt.show()
    else:
        plt.savefig(SavePath, dpi=300, bbox_inches='tight')