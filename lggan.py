import os
import csv
import argparse
import torch
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


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
parser.add_argument('--resume', type=str, default='/', help='resume path')
parser.add_argument('--feature_transform', type=int, default=1, help="use feature transform")
parser.add_argument('--lambda_ft', type=float, default=0.001, help="lambda for feature transform")
parser.add_argument('--augment', type=int, default=1, help='data argment to increase robustness')
parser.add_argument('--name', type=str, default='train', help='name of the experiment')
parser.add_argument('--note', type=str, default='', help='notation of the experiment')
parser.add_argument('--adv_path', default='LGGAN', help='output adversarial example path [default: LGGAN]')
parser.add_argument('--lggan', type=str,default='lggan', help='run file store place')

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
    train_loader = DataLoader(train_data, num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    ########################################
    ## Intiate model
    ########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator()
    discriminator = Discriminator(torch.nn.functional.leaky_relu,batch=args.batch_size).to(device)
    attack_model = PointNetCls(NUM_CLASSES, args.feature_transform).to(device)
    attack_model=attack_model.to(device)

    pred_loss = torch.nn.functional.cross_entropy
    generator_loss = torch.nn.MSELoss()
    d_optim = torch.optim.Adam(list(discriminator.parameters()),lr=d_lr,betas=(0.5, 0.599))
    g_optim = torch.optim.Adam(list(generator.parameters()),lr=g_lr,betas=(0.9, 0.999))
    for my_iter in range(ITERATION):
        error_cnt = 0
        is_training = False
        total_correct_adv = 0
        total_seen = 0
        total_attack_adv = 0
        total_seen_class_adv = [0 for _ in range(NUM_CLASSES)]
        total_correct_class_adv = [0 for _ in range(NUM_CLASSES)]
        # testing phase
        is_training_ae = False
        for i, data in enumerate(train_loader, 0):
            points, label = data
            print(points.shape)
            target_labels = generate_labels(label.numpy())
            target_labels = torch.from_numpy(target_labels).to(device)
            points, label = points.to(device), label.to(device)[:, 0]
            points = points.transpose(2, 1)
            points_adv = generator(points,label)
            print(points_adv.shape)
            break
        break