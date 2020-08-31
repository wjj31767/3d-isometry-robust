from __future__ import print_function

import argparse
import os
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_class import ModelNet40, ShapeNetPart
from data.transforms_3d import *
from models.dgcnn import DGCNN
from models.pointnet import PointNetCls, get_loss_v2
from models.pointnet2 import PointNet2ClsMsg
from utils import progress_bar


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss




def drop_points( points, target,model,criterion,numdrop,numstep):
    points_adv = points.clone()
    for i in range(numstep):
        points_adv = torch.tensor(points_adv, dtype=torch.float32).to(device)
        points_adv.requires_grad_(True)
        pred,trans_feat=model(points_adv)
        loss = criterion(pred,target.long())
        model.zero_grad()
        loss.backward()
        grad = points_adv.grad.cpu().detach().numpy()
        points_adv = points_adv.cpu().detach().numpy()
        # change the grad into spherical axis and compute r*dL/dr
        ## mean value
        # sphere_core = np.sum(pointclouds_pl_adv, axis=1, keepdims=True)/float(pointclouds_pl_adv.shape[1])
        ## median value
        sphere_core = np.median(points_adv, axis=2, keepdims=True)

        sphere_r = np.sqrt(np.sum(np.square(points_adv - sphere_core), axis=1))  ## BxN

        sphere_axis = points_adv - sphere_core  ## BxNx3

        #if args.drop_neg:
        #    sphere_map = np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=1),
        #                             np.power(sphere_r, args.power))
        #else:
        sphere_map = -np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=1),
                                      np.power(sphere_r, 1))

        drop_indice = np.argpartition(sphere_map, kth=sphere_map.shape[1] - numdrop, axis=1)[:, -numdrop:]
        tmp = np.zeros((points_adv.shape[0], 3,points_adv.shape[2] - numdrop), dtype=float)
        for j in range(points_adv.shape[0]):
            tmp[j] = np.delete(points_adv[j], drop_indice[j], axis=1)  # along N points to delete
        points_adv = tmp.copy()
    points_adv = torch.tensor(points_adv, dtype=torch.float32)
    return points_adv

if __name__ == '__main__':
    ########################################
    ## Set hypeparameters
    ########################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pointnet', help='choose model type')
    parser.add_argument('--data', type=str, default='modelnet40', help='choose data set')
    parser.add_argument('--seed', type=int, default=0, help='manual random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--step', nargs='+', default= [50, 80, 120, 150], 
                        help='epochs when to change lr, for example type "--adj_step 50 80 120 150" in command line')
    parser.add_argument('--dr', nargs='+', default=[0.1, 0.1, 0.2, 0.2], help='decay rates of learning rate' )
    parser.add_argument('--resume', type=str, default='example', help='resume path')
    parser.add_argument('--feature_transform', type=int, default=1, help="use feature transform")    
    parser.add_argument('--lambda_ft',type=float, default=0.001, help="lambda for feature transform")    
    parser.add_argument('--augment', type=int, default= 1, help='data argment to increase robustness')
    parser.add_argument('--name', type=str, default='train', help='name of the experiment')
    parser.add_argument('--note', type=str, default='', help='notation of the experiment')
    parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
    parser.add_argument('--num_steps', type=int, default=10, help='num of steps to drop each step')
    parser.add_argument('--num_drop', type=int, default=10, help='num of points to drop each step')
    parser.add_argument('--drop_neg', action='store_true', help='drop negative points')

    args = parser.parse_args()
    print(args)
    args.adj_lr = {'steps' : [int(temp) for temp in args.step], 
                   'decay_rates' : [float(temp) for temp in args.dr]}
    args.feature_transform , args.augment = bool(args.feature_transform), bool(args.augment)
    ### Set random seed
    args.seed = args.seed if args.seed > 0 else random.randint(1, 10000)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    SHAPE_NAMES = [line.rstrip() for line in \
                   open(os.path.join(BASE_DIR, 'data/modelnet40_data/modelnet40_ply_hdf5_2048/shape_names.txt'))]
    num_drop, num_steps = args.num_drop, args.num_steps
    DUMP_DIR = args.dump_dir
    if not os.path.exists(DUMP_DIR):
        os.mkdir(DUMP_DIR)
    ########################################
    ## Intiate model
    ########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data == 'modelnet40':
        num_classes = 40
    elif args.data == 'shapenetpart':
        num_classes = 16


    if args.model == 'pointnet':
        model = PointNetCls(num_classes, args.feature_transform)  
        model = model.to(device)  
    elif args.model == 'pointnet2':
        model = PointNet2ClsMsg(num_classes)
        model = model.to(device)
        model = nn.DataParallel(model)
    elif args.model == 'dgcnn':
        model = DGCNN(num_classes)
        model = model.to(device) 
        model = nn.DataParallel(model)



    print('=====> Loading from checkpoint...')
    checkpoint = torch.load('checkpoints/%s.pth' % args.resume)
    args = checkpoint['args']

    torch.manual_seed(args.seed)
    print("Random Seed: ", args.seed)

    model.load_state_dict(checkpoint['model_state_dict'])
    loss = get_loss_v2()
    print('Successfully resumed!')
    
    ########################################
    ## Load data
    ########################################
    print('======> Loading data')
    #print(args.augment, args.feature_transform)

    test_tfs = normalize()
    if args.data == 'modelnet40':
        test_data = ModelNet40(partition='test', num_points=args.num_points, transforms=test_tfs)
    elif args.data == 'shapenetpart':
        test_data = ShapeNetPart(partition='test', num_points=args.num_points, transforms=test_tfs)


    test_loader = DataLoader(test_data, num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    print('======> Successfully loaded!')



    ########################################
    ## Train
    ########################################
    if args.model == 'dgcnn':
        criterion = cal_loss
    else:
        criterion = F.cross_entropy #nn.CrossEntropyLoss()



        ### Test in batch 

    model.eval()
    correct = 0
    correct_adv = 0
    total = 0
    class_acc = np.zeros((num_classes, 3))
    class_acc_adv = np.zeros((num_classes, 3))
    for j, data in enumerate(test_loader, 0):
        points, label = data
        points, label = points.to(device), label.to(device)[:, 0]
        model.eval()

        points = points.transpose(2, 1)  # to be shape batch_size*3*N
        pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(label.cpu()):
            classacc = pred_choice[label == cat].eq(label[label == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[label == cat].size()[0])
            class_acc[cat, 1] += 1
        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(label.data).cpu().sum()
        total += label.size(0)


        cur_batch_data_adv = drop_points(points, label, model, loss, num_drop, num_steps)
        cur_batch_data_adv = cur_batch_data_adv.cuda()
        pred_adv, _ = model(cur_batch_data_adv)
        pred_choice_adv = pred_adv.data.max(1)[1]
        for cat in np.unique(label.cpu()):
            classacc_adv = pred_choice_adv[label == cat].eq(label[label == cat].long().data).cpu().sum()
            class_acc_adv[cat, 0] += classacc_adv.item() / float(cur_batch_data_adv[label == cat].size()[0])
            class_acc_adv[cat, 1] += 1
        correct_adv += pred_choice_adv.eq(label.long().data).cpu().sum()
        progress_bar(j, len(test_loader), 'Test Acc: %.3f%% (%d/%d)|Adv test acc: %.3f%%(%d/%d)'% (100. * correct.item() / total, correct,total,100.*correct_adv/total,correct_adv,total))


    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc_adv[:, 2] = class_acc_adv[:, 0] / class_acc_adv[:, 1]

    for i, name in enumerate(SHAPE_NAMES):
        print('%10s:\t%0.3f' % (name, class_acc[:,2][i]))
        print('%10s:\t%0.3f' % (name, class_acc_adv[:,2][i]))
