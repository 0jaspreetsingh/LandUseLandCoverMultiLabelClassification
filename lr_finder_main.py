import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
import logging
import numpy as np
import random
import warnings
from datetime import datetime
from pathlib import Path
import utils
import loader
from lr_finder import LRFinder
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
        print('Using AMP..')
except AttributeError:
    pass
warnings.filterwarnings("ignore")


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='lclu')

    parser.add_argument(
        '--root-dir',
        type=dir_path,
        help='Path to images folder and label.csv file')
    parser.add_argument(
        '--rgb',
        type=str2bool,
        default=False,
        help='set true to load rgb images')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='/netscratch/jaspreet/rgb_out',
        help='Path to save models')
    parser.add_argument(
        '--model-choice',
        type=str,
        default='alexnet',
        help='The model name to train. Choose from: alexnet')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=19,
        help='Number of classes')
    parser.add_argument(
        '--saved-model',
        type=str,
        default=None,
        help='Path to saved pytorch model to evaluate or resume training')
    parser.add_argument(
        '--train-csv',
        type=str,
        default='train.csv',
        help='Training split')
    parser.add_argument(
        '--val-csv',
        type=str,
        default='val.csv',
        help='Validation split')
    parser.add_argument(
        '--test-csv',
        type=str,
        default='test.csv',
        help='Testing split')
    parser.add_argument(
        '--test',
        type=str2bool,
        default=False,
        help='set true to test the model')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Initial learning rate')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.0001,
        help='Weight decay')
    parser.add_argument(
        '--loss',
        type=str,
        default='bce',
        help='Loss function to optimize. focal or bce')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        help='Optimizer for weights update. sgd or adam')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Optimizer momentum')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=50,
        help='Number of training epochs')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Input batch size')
    parser.add_argument(
        '--notes',
        type=str,
        default=None,
        help='Notes about experiment')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=os.cpu_count(),
        # default=int(os.environ['SLURM_CPUS_ON_NODE']),
        help='The number of workers for data loaders')
    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='patience for early stopping')
    parser.add_argument(
        '--train-countries',
        '--list',
        nargs='+',
        help='Set of countries to be in training dataset',
        required=True)
    parser.add_argument(
        '--test-countries',
        '--names-list',
        nargs='+',
        help='Set of countries to be in testing dataset',
        required=True)
    parser.add_argument(
        '--chunk-size',
        type=str,
        default=None,
        help='Size of the target dataset to be used for retraining')
    return parser


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = get_parser()
    args = parser.parse_args()
    init_seed(args.seed)
    log.info(f"Cuda Available: {torch.cuda.is_available()}")
    num_gpus = torch.cuda.device_count()
    log.info(f"Cuda device count: {num_gpus}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f'Device is: {device}')

    model_dir = os.path.join(args.out_dir, "lr_finder_plot_" + datetime.today().strftime('%Y%m%d_%H%M%S'))
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    plot_file = os.path.join(model_dir, 'lr_plot.png')

    lbl_corr_vec = None
    with open('./corr_mats/all_bands.npy', 'rb') as f:
        lbl_corr_vec = torch.from_numpy(np.load(f).flatten()).float()
    lbl_corr_vec = lbl_corr_vec.to(device)

    model = loader.load_model(lbl_corr_vec, args=args, device=device)

    num_params = utils.count_params(model)
    log.info(f'Total number of model parameters: {num_params}')

    train_loader, val_loader, _ = loader.load_data(args)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=100, num_iter=100)
    lr_finder.plot(save_file=plot_file, suggest_lr=True)  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state


if __name__ == '__main__':
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.addHandler(TqdmLoggingHandler())
    main()
