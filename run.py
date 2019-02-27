import argparse
import json
from train import Model
from utils import prepare_data_lists, restore_pkl_object, split_dataset_to_train_val
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise parser.ArgumentTypeError('Boolean value expected.')


def str2list(s):
    return eval(s)


# Training settings
parser = argparse.ArgumentParser(description='Cough detection')
parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint', help="directory where summary will be written")
parser.add_argument('--dataset_pkl_path', type=str, default='/home/sam/Desktop/first_model_with_data/labels.pckl', help="data directories")
parser.add_argument("--checkpoint_to_restore", type=str, default='', help="checkpoint to restore")
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.001')
parser.add_argument('--lr_decay', type=float, default=1, help='Learning Rate decay')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--num_epochs', type=int, default=15000, help='number of epochs to train for')
parser.add_argument("--checkpoint_freq", type=int, default=7, help='when output a graph, frequency as iteration')
parser.add_argument("--use_gpu", type=str2bool, default=True, help='run graph on GPU')

opt = parser.parse_args()

print "Running with following options ..."
print '-----------------------------------'
print json.dumps(vars(opt), indent=2)
print '-----------------------------------\n'
print 'Loading datasets'
print '-----------------------------------\n'

os.chdir(os.path.dirname(os.path.realpath(__file__)))


# train_data, val_data = prepare_data_lists(opt.dataset_pkl_path)

dataset = restore_pkl_object(opt.dataset_pkl_path)

train_data, val_data = split_dataset_to_train_val(dataset, train_dataset_percent=0.8)

model = Model(opt, train_data, val_data)
model.train(opt)
