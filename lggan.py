import os
import csv
import argparse
import torch
from torch.nn import functional as F
from torch import nn
from utils import IOStream
import random
from data.data_class import ModelNet40
from torch.utils.data import Dataset, DataLoader
from data.transforms_3d import *
from provider import load_h5
import sys
import h5py
from models.point_resnet import Discriminator
from models.lgnet import Generator
from models.pointnet import PointNetCls
from utils import progress_bar, adjust_lr_steep, log_row
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import plotly.graph_objs as go

########################################
## Set hypeparameters
########################################
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='pointnet', help='choose model type')
parser.add_argument('--data', type=str, default='modelnet40', help='choose data set')
parser.add_argument('--seed', type=int, default=0, help='manual random seed')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate in training')
parser.add_argument('--step', nargs='+', default=[50, 80, 120, 150],
                    help='epochs when to change lr, for example type "--adj_step 50 80 120 150" in command line')
parser.add_argument('--dr', nargs='+', default=[0.1, 0.1, 0.2, 0.2], help='decay rates of learning rate')
parser.add_argument('--resume', type=str, default='example', help='resume path')
parser.add_argument('--feature_transform', type=int, default=1, help="use feature transform")
parser.add_argument('--lambda_ft', type=float, default=0.001, help="lambda for feature transform")
parser.add_argument('--augment', type=int, default=1, help='data argment to increase robustness')
parser.add_argument('--name', type=str, default='train', help='name of the experiment')
parser.add_argument('--note', type=str, default='', help='notation of the experiment')
parser.add_argument('--adv_path', default='LGGAN', help='output adversarial example path [default: LGGAN]')
parser.add_argument('--lggan', type=str,default='lggan', help='run file store place')
parser.add_argument('--tau', type=float, default=1e2, help='balancing weight for loss function [default: 1e2]')
args = parser.parse_args()
args.adj_lr = {'steps': [int(temp) for temp in args.step],
               'decay_rates': [float(temp) for temp in args.dr]}
args.feature_transform, args.augment = bool(args.feature_transform), bool(args.augment)
### Set random seed
args.seed = args.seed if args.seed > 0 else random.randint(1, 10000)
if not os.path.exists('checkpoints/'+args.lggan):
    os.mkdir('checkpoints/'+args.lggan)
io = IOStream('checkpoints/' + args.lggan + '/run.log')
io.cprint(str(args))
TAU = args.tau
ITERATION = 100

# create adversarial example path
ADV_PATH = args.adv_path
if not os.path.exists('results'): os.mkdir('results')
ADV_PATH = os.path.join('results', ADV_PATH)
if not os.path.exists(ADV_PATH): os.mkdir(ADV_PATH)
ADV_PATH = os.path.join(ADV_PATH, 'test')


NUM_CLASSES = 40


def write_h5(data, data_orig, label, label_orig, num_batches):

    h5_filename = ADV_PATH+str(num_batches)+'.h5'
    h5f = h5py.File(h5_filename, 'w')
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('orig_data', data=data_orig)
    h5f.create_dataset('label', data=label)
    h5f.create_dataset('orig_label', data=label_orig)
    h5f.close()
def generate_labels(labels):
    targets = np.zeros(np.size(labels))
    for i in range(len(labels)):
        rand_v = random.randint(0, NUM_CLASSES-1)
        while labels[i]==rand_v:
            rand_v = random.randint(0, NUM_CLASSES-1)
        targets[i] = rand_v
    targets = targets.astype(np.int32)

    return targets

def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res
def draw(points):
    trace = go.Scatter3d(
        x=points[0, 0, :], y=points[0, 1, :], z=points[0, 2, :], mode='markers', marker=dict(
            size=12,
            color=points[0, 2, :],  # set color to an array/list of desired values
            colorscale='Viridis'
        )
    )
    layout = go.Layout(title='3D Scatter plot')
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
if __name__ == '__main__':
    g_lr = 1e-3
    d_lr= 1e-5
    train_tfs = compose([rotate_y(),
                         rand_scale(),
                         rand_translate(),
                         jitter(),
                         normalize()
                         ])
    test_tfs = normalize()
    train_data = ModelNet40(partition='train', num_points=args.num_points, transforms=train_tfs)
    test_data = ModelNet40(partition='test', num_points=args.num_points, transforms=test_tfs)
    train_loader = DataLoader(train_data, num_workers=4,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, num_workers=4,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    ########################################
    ## Intiate model
    ########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack_model = PointNetCls(NUM_CLASSES, args.feature_transform).to(device)
    attack_model=attack_model.to(device)
    if len(args.resume) > 1:
        print('=====> Loading from checkpoint...')
        checkpoint = torch.load('checkpoints/%s.pth' % args.resume)
        args = checkpoint['args']

        torch.manual_seed(args.seed)
        print("Random Seed: ", args.seed)

        attack_model.load_state_dict(checkpoint['model_state_dict'])

        print('Successfully resumed!')
    else:
        print("there's no checkpoint")
    generator = Generator(up_ratio=1).to(device)
    discriminator = Discriminator(torch.nn.functional.leaky_relu,num_point=args.num_points).to(device)

    attack_model_criterion = F.cross_entropy
    generator_criterion = F.mse_loss
    d_optim = torch.optim.Adam(list(discriminator.parameters()),lr=d_lr,betas=(0.5, 0.599))
    g_optim = torch.optim.Adam(list(generator.parameters()),lr=g_lr,betas=(0.9, 0.999))
    attack_model.eval()
    for my_iter in range(ITERATION):
        print("ITERATION:",my_iter)
        correct = 0
        correct_adv = 0
        total = 0
        # testing phase
        generator.eval()
        discriminator.eval()
        print("TEST")
        for i, data in enumerate(test_loader, 0):
            points, label = data
            target_labels = generate_labels(label[:, 0].numpy())
            target_labels = torch.from_numpy(target_labels)
            points, label = points.to(device), label.to(device)[:, 0]
            points = points.transpose(2, 1)
            # draw(points.cpu())
            # print(points.shape)
            pred_original, _ = attack_model(points)
            pred_choice = pred_original.data.max(1)[1]
            correct += pred_choice.eq(label.data).cpu().sum()
            # g_optim.zero_grad()
            target_labelsg = one_hot(target_labels,40).to(device)
            # generator.zero_grad()
            points_adv = generator(points,target_labelsg,device)
            # print(points_adv.shape)

            # draw(points_adv.cpu().detach().numpy())
            pred, _ = attack_model(points_adv)
            pred_choice = pred.data.max(1)[1]
            correct_adv += pred_choice.eq(label.data).cpu().sum()
            total += label.size(0)
            generator_loss = ((pred_choice - label)**2).float().mean()

            d_real = discriminator(points)
            d_fake = discriminator(points_adv)
            d_loss_fake = torch.mean(d_fake**2)
            d_loss_real = torch.mean((d_real-1)**2)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            g_loss = torch.mean((d_fake-1)**2)
            pred_loss = attack_model_criterion(pred_original,label)
            g_loss = generator_loss+TAU*pred_loss+g_loss
            progress_bar(i, len(test_loader), 'Generator Loss: %.3f | Discrimitor Loss: %.3f| Test Acc: %.3f%% (%d/%d) | Test After Gen Acc: %.3f%% (%d/%d)'
                         % (g_loss.item() / (i + 1), d_loss.item() / (i + 1),100. * correct.item() / total, correct, total,100. * correct_adv.item() / total, correct_adv, total))

        correct = 0
        correct_adv = 0
        total = 0
        generator.train()
        discriminator.train()
        print("TRAIN")
        for i, data in enumerate(train_loader, 0):
            points, label = data
            target_labels = generate_labels(label[:, 0].numpy())
            target_labels = torch.from_numpy(target_labels)
            points, label = points.to(device), label.to(device)[:, 0]
            points = points.transpose(2, 1)
            # draw(points.cpu())
            # print(points.shape)
            g_optim.zero_grad()
            d_optim.zero_grad()
            labelg = one_hot(label,40).to(device)
            target_labelsg = one_hot(target_labels, 40).to(device)
            points_adv = generator(points, target_labelsg, device)


            d_fake = discriminator(points_adv.detach())
            d_loss_fake = torch.mean(d_fake ** 2)
            d_loss_fake.backward()
            d_real = discriminator(points)
            d_loss_real = torch.mean((d_real - 1) ** 2)
            d_loss_real.backward()
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_optim.step()

            d_fake = discriminator(points_adv)
            g_loss = torch.mean((d_fake - 1) ** 2)
            g_loss.backward(retain_graph=True)
            pred, _ = attack_model(points_adv)
            pred_choice = pred.data.max(1)[1]
            correct_adv += pred_choice.eq(label.data).cpu().sum()
            total += label.size(0)
            pred_loss = attack_model_criterion(pred, label)
            generator_loss = torch.mean((pred-labelg)**2)
            g_loss = generator_loss + TAU * pred_loss + g_loss

            g_loss.backward()
            g_optim.step()

            progress_bar(i, len(train_loader),
                         'Generator Loss: %.3f | Discrimitor Loss: %.3f| Train GAN Acc: %.3f%% (%d/%d)'
                         % (g_loss.item() / (i + 1), d_loss.item() / (i + 1),
                          100. * correct_adv/ total, correct_adv, total))
