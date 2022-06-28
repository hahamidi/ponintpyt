
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from datasets import ShapeNetDataset, PointMNISTDataset
from trainer_me import Trainer
from model.DGCNN import DGCNN_partseg




    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['shapenet', 'mnist'], help='dataset to train on')
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')
    parser.add_argument('output_folder', type=str, help='output folder')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--number_of_workers', type=int, default=1, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--num_points', type=int, default=2048,help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',help='Num of nearest neighbors to use')
    print(1)
    args = parser.parse_args()

    print(1)
    DATASETS = {
                'shapenet': ShapeNetDataset,
                'mnist': PointMNISTDataset
            }
    train_dataset = DATASETS[args.dataset](args.dataset_folder,
                                      task=args.task,
                                      number_of_points=args.num_points)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.number_of_workers)
    test_dataset = DATASETS[args.dataset](args.dataset_folder,
                                            task=args.task,
                                            train=False,
                                            number_of_points=args.num_points)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.number_of_workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN_partseg(args, train_dataset.NUM_SEGMENTATION_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    
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





