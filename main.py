import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataloader import get_data_loader
from train import train
from test import test
from model import Net

if __name__ == '__main__':

	train_loader, test_loader = get_data_loader(128)
	
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(device)
	model = Net().to(device)
	epochs = 85

	optimizer = optim.SGD(model.parameters(), lr=0.1, momentum = 0.9)
	scheduler = optim.lr_scheduler.OneCycleLR(
	        optimizer,
	        max_lr=4.93E-02,
	        steps_per_epoch=len(train_loader),
	        epochs=epochs,
	        pct_start=5/epochs,
	        div_factor=100,
	        three_phase=False,
	        final_div_factor=100,
	        anneal_strategy='linear'
	    )
	criterion = F.nll_loss

	for epoch in range(1, epochs):
	  train(model, device, train_loader, optimizer, criterion, scheduler)
	  test(model, device, test_loader, criterion)
	  print(f"epoch {epoch} Done")