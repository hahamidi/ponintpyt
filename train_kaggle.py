
import argparse
import os

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from fastprogress import master_bar, progress_bar

from datasets import PointMNISTDataset
from part import ShapeNetDataset
from model.pointnet import ClassificationPointNet, SegmentationPointNet
from utils import plot_losses, plot_accuracies
from tqdm import tqdm
import tensorflow
from tensorflow.keras.metrics import MeanIoU

from sklearn.manifold import TSNE as sklearnTSNE
import matplotlib.pyplot as plt
import random
from pylab import cm

dire = __file__.split('/')[:-1]
dire = '/'.join(dire)


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

        self.load_model = False
        self.load_epoch = 0

        self.blue= lambda x: '\033[94m' + x + '\033[0m'
        self.red = lambda x: '\033[91m' + x + '\033[0m'




    def train_one_epoch(self,epoch_num):

                epoch_train_loss = []
                epoch_train_acc = []
                batch_number = 0
                # batch_iter = tqdm(enumerate(self.train_data_loader), 'Training', total=len(self.train_data_loader),
                #                 position=0)
                self.model = self.model.train()
                for data in self.train_data_loader:
                    batch_number += 1
                    points, targets = data
                    # print(targets)
 
                    points, targets = points.to(self.device), targets.to(self.device)
  

                    if points.shape[0] <= 1:
                        continue
                    self.optimizer.zero_grad()
                    
                    preds, feature_transform = self.model(points)
                    # if idx == 0:
                    #     self.show_embedding_sklearn((preds).cpu().detach().numpy(),targets.cpu().detach().numpy(),title = "train"+str(epoch_num))
                    preds = preds.view(-1, self.number_of_classes)
                    targets = targets.view(-1)

                    identity = torch.eye(feature_transform.shape[-1]).to(self.device)
                    regularization_loss = torch.norm(
                        identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1))
                    )
                    loss = F.nll_loss(preds, targets) + 0.001 * regularization_loss
                    # print(loss)
                    epoch_train_loss.append(loss.cpu().item())
                    loss.backward()
                    self.optimizer.step()
                    preds = preds.data.max(1)[1]
                    corrects = preds.eq(targets.data).cpu().sum()

                    accuracy = corrects.item() / float(self.train_data_loader.batch_size*2500)
                    epoch_train_acc.append(accuracy)
                    # batch_iter.set_description(self.blue('train loss: %f, train accuracy: %f' % (loss.cpu().item(),accuracy)))
                print("Loss",np.mean(epoch_train_loss))
                print("Accuracy",np.mean(epoch_train_acc))

                                                                        
    def val_one_epoch(self,epoch_num):
        epoch_val_loss = []
        epoch_val_acc = []
        batch_number = 0
        shape_ious = []
        batch_iter = tqdm(enumerate(self.val_data_loader), 'Validation', total=len(self.val_data_loader),position=0)
        self.model = self.model.eval()
        with tensorflow.device('/cpu:0'):
            m = MeanIoU(self.number_of_classes, name=None, dtype=None)
            for idx,data in batch_iter:
                        

                        batch_number += 1
                        points, targets = data
                        # print(targets)
    
                        points, targets = points.to(self.device), targets.to(self.device)

                        if points.shape[0] <= 1:
                            continue

                        
                        preds, feature_transform = self.model(points)
    
                        preds = preds.view(-1, self.number_of_classes)
                        targets = targets.view(-1)

                        identity = torch.eye(feature_transform.shape[-1]).to(self.device)
                        regularization_loss = torch.norm(
                            identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1))
                        )
                        loss = F.nll_loss(preds, targets) + 0.001 * regularization_loss
                        epoch_val_loss.append(loss.cpu().item())
                        preds = preds.data.max(1)[1]
                        pred_np = preds.cpu().data.numpy()
                        target_np = targets.cpu().data.numpy()
                        m.update_state(pred_np, target_np)
                        part_ious = m.result().numpy()
                        shape_ious.append(np.mean(part_ious))

                        corrects = preds.eq(targets.data).cpu().sum()

                        accuracy = corrects.item() / float(self.val_data_loader.batch_size*2500)
                        epoch_val_acc.append(accuracy)
                        batch_iter.set_description(self.red('val loss: %f, val accuracy: %f,MIou %f' % (loss.cpu().item(),
                                                                                accuracy,np.mean(part_ious))))
        print("Loss",np.mean(epoch_val_loss))
        print("Accuracy",np.mean(epoch_val_acc))
        print("Mean IOU: ", np.mean(shape_ious))


    def save_model_optimizer(self,epoch_num):
        torch.save(self.model.state_dict(), dire+'/checkpoints/model_epoch_' + str(epoch_num) + '.pth')
        torch.save(self.optimizer.state_dict(),  dire+'/checkpoints/optimizer_epoch_' + str(epoch_num) + '.pth')
        print('Model and optimizer saved!')

    def load_model_optimizer(self,epoch_num):
        self.model.load_state_dict(torch.load( dire+'/checkpoints/model_epoch_' + str(epoch_num) + '.pth'))
        self.optimizer.load_state_dict(torch.load( dire+'/checkpoints/optimizer_epoch_' + str(epoch_num) + '.pth'))
        print('Model and optimizer loaded!')

    def show_embedding_sklearn(self,tsne_embs_i, lbls,title = "", cmap=plt.cm.tab20,highlight_lbls = None):
            
            labels = lbls.flatten()
            print(labels.shape)
            print(tsne_embs_i.shape)
            feat = np.zeros((tsne_embs_i.shape[1],tsne_embs_i.shape[2])).T

            for b in tsne_embs_i:
                feat= np.concatenate((feat, b.T), axis=0)

            feat= feat[tsne_embs_i.shape[2]: , :]
            number_of_labels = np.amax(labels) + 1
            selected = np.zeros((tsne_embs_i.shape[1],1)).T
            labels_s = []
            print(feat.shape)

            for i in range(number_of_labels):
                selected= np.concatenate((selected,feat[labels == i][0:100]), axis=0)
                labels_s= np.concatenate((labels_s,labels[labels == i][0:100]), axis=0)
            selected = selected[1:]

            tsne = sklearnTSNE(n_components=2, random_state=0)  # n_components means you mean to plot your dimensional data to 2D
            x_test_2d = tsne.fit_transform(selected)

            fig,ax = plt.subplots(figsize=(10,10))
            ax.scatter(x_test_2d[:,0], x_test_2d[:,1], c=labels_s, cmap=cmap, alpha=1 if highlight_lbls is None else 0.1)
            random_str = str(random.randint(0,1000000))
            plt.savefig("/./content/embed"+random_str+"-"+str(title)+'.png')



    def train(self):
            if self.load_model == True:
                self.load_model_optimizer(self.load_epoch)

            for epoch in range(self.epochs):
                
                self.train_one_epoch(epoch)
                # self.val_one_epoch(epoch)
                if epoch % 20 == 0:
                   self.save_model_optimizer(epoch)

                
                # self.scheduler.step()
                # torch.save(self.model.state_dict(), 'model_%d.pkl' % epoch)

    

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['shapenet', 'mnist'], help='dataset to train on')
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')
    parser.add_argument('output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2500, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
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
    print(device)
    trainer = Trainer(model = model,
                        train_data_loader = train_dataloader, 
                        val_data_loader = test_dataloader, 
                        optimizer = optimizer,
                        epochs=args.epochs,
                        number_of_classes = train_dataset.NUM_SEGMENTATION_CLASSES,
                        loss_function = F.cross_entropy,
                        scheduler = None,
                        device =device)
    print(train_dataset.NUM_SEGMENTATION_CLASSES)
    trainer.train()








