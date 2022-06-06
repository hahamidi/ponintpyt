
import argparse
import os

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from fastprogress import master_bar, progress_bar

from datasets import ShapeNetDataset, PointMNISTDataset
from model.pointnet import ClassificationPointNet, SegmentationPointNet
from utils import plot_losses, plot_accuracies
from tqdm import tqdm

class Trainer():
    def __init__(self,model,
                        train_data_loader, 
                        val_data_loader , 
                        optimizer ,
                        epochs,
                        number_of_classes,
                        loss_function,
                        scheduler,
                        device):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.number_of_classes = number_of_classes
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.device = device




    def train_one_epoch(self,epoch_num):

                epoch_train_loss = []
                epoch_train_acc = []
                batch_number = 0
                batch_iter = tqdm(enumerate(self.train_data_loader), 'Training', total=len(self.train_data_loader),
                                position=0)
                for idx,data in batch_iter:
                    batch_number += 1
                    points, targets = data
 
                    points, targets = points.to(self.device), targets.to(self.device)
                    if points.shape[0] <= 1:
                        continue
                    optimizer.zero_grad()
                    self.model = self.model.train()
                    preds, feature_transform = self.model(points)
  
                    preds = preds.view(-1, train_dataset.NUM_SEGMENTATION_CLASSES)
                    targets = targets.view(-1)

                    identity = torch.eye(feature_transform.shape[-1]).to(self.device)
                    regularization_loss = torch.norm(
                        identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1))
                    )
                    loss = F.nll_loss(preds, targets) + 0.001 * regularization_loss
                    epoch_train_loss.append(loss.cpu().item())
                    loss.backward()
                    optimizer.step()
                    preds = preds.data.max(1)[1]
                    corrects = preds.eq(targets.data).cpu().sum()

                    accuracy = corrects.item() / float(self.train_data_loader.batch_size*2500)
                    epoch_train_acc.append(accuracy)
                    batch_iter.set_description('train loss: %f, train accuracy: %f' % (np.mean(epoch_train_loss),
                                                                            np.mean(epoch_train_acc)))

    

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['shapenet', 'mnist'], help='dataset to train on')
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')
    parser.add_argument('output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2500, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--number_of_workers', type=int, default=1, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')

    args = parser.parse_args()
    MODELS = {
                'classification': ClassificationPointNet,
                'segmentation': SegmentationPointNet
            }

    DATASETS = {
                'shapenet': ShapeNetDataset,
                'mnist': PointMNISTDataset
            }
    train_dataset = DATASETS[args.dataset](args.dataset_folder,
                                      task=args.task,
                                      number_of_points=args.number_of_points)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.number_of_workers)
    test_dataset = DATASETS[args.dataset](args.dataset_folder,
                                            task=args.task,
                                            train=False,
                                            number_of_points=args.number_of_points)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.number_of_workers)

    if args.task == 'classification':
            model = ClassificationPointNet(num_classes=train_dataset.NUM_CLASSIFICATION_CLASSES,
                                       point_dimension=train_dataset.POINT_DIMENSION)
    elif args.task == 'segmentation':
            model = SegmentationPointNet(num_classes=train_dataset.NUM_SEGMENTATION_CLASSES,
                                     point_dimension=train_dataset.POINT_DIMENSION)
    else:
                raise Exception('Unknown task !')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    trainer = Trainer(model = model,
                        train_data_loader = train_dataloader, 
                        val_data_loader = test_dataloader, 
                        optimizer = optimizer,
                        epochs=args.epochs,
                        number_of_classes = train_dataset.NUM_CLASSIFICATION_CLASSES,
                        loss_function = F.cross_entropy,
                        scheduler = None,
                        device =device)
    trainer.train_one_epoch(0)




