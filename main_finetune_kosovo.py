import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
import mlflow
import mlflow.pytorch
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score, multilabel_confusion_matrix, \
    coverage_error, label_ranking_average_precision_score, label_ranking_loss, hamming_loss
import warnings
from datetime import datetime
from pathlib import Path
import utils
import loss_fns
import loader
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
        print('Using AMP..')
except AttributeError:
    pass
warnings.filterwarnings("ignore")


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
        '--model-name',
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
        # default=os.cpu_count(),
        default=int(os.environ['SLURM_CPUS_ON_NODE']),
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
    return parser


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_metrics(targets, predictions, prefix):
    with torch.no_grad():
        predictions = torch.sigmoid(predictions)
        predictions[predictions >= 0.5] = 1
        exact_match = (predictions == targets).sum() / (len(targets) * 19)
        mlflow.log_metric(prefix + "exact_match", exact_match.item())
        targets = targets.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy() > 0.5

        mlflow.log_metric(prefix + "acc_score", accuracy_score(targets, predictions))
        mlflow.log_metric(prefix + "f2_samples", fbeta_score(targets, predictions, average="samples", beta=2))
        mlflow.log_metric(prefix + "f2_macro", fbeta_score(targets, predictions, average="macro", beta=2))
        mlflow.log_metric(prefix + "f2_micro", fbeta_score(targets, predictions, average="micro", beta=2))
        mlflow.log_metric(prefix + "p_samples", precision_score(targets, predictions, average="samples"))
        mlflow.log_metric(prefix + "p_macro", precision_score(targets, predictions, average="macro"))
        mlflow.log_metric(prefix + "p_micro", precision_score(targets, predictions, average="micro"))
        mlflow.log_metric(prefix + "r_samples", recall_score(targets, predictions, average="samples"))
        mlflow.log_metric(prefix + "r_macro", recall_score(targets, predictions, average="macro"))
        mlflow.log_metric(prefix + "r_micro", recall_score(targets, predictions, average="micro"))
        mlflow.log_metric(prefix + "hl", hamming_loss(targets, predictions))
        mlflow.log_metric(prefix + "cov", coverage_error(targets, predictions))
        mlflow.log_metric(prefix + "lrap", label_ranking_average_precision_score(targets, predictions.astype(np.float)))
        mlflow.log_metric(prefix + "rl", label_ranking_loss(targets, predictions))

        if prefix != "train_":
            cmat = multilabel_confusion_matrix(y_true=targets, y_pred=predictions)
            classwise_f2 = utils.get_classwise_f2_score(cmat=cmat)
            mlflow.log_metrics(classwise_f2)


def train(model, train_loader, val_loader, args, criterion, optimizer, scheduler, device):
    patience = args.patience
    best_loss = np.inf

    out_dir = args.out_dir
    model_dir = os.path.join(out_dir, "model_" + datetime.today().strftime('%Y%m%d_%H%M%S'))
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_file = os.path.join(model_dir, 'model.pt')
    mlflow.log_param('model_dir', model_dir)

    if has_native_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Train
    for epoch in range(args.num_epoch):
        targets = None
        predictions = None
        print(f"Training epoch: {epoch + 1}")
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if has_native_amp:
                with torch.cuda.amp.autocast():
                    preds = model(inputs)
                    batch_loss = criterion(preds, labels.float())
                scaler.scale(batch_loss).backward()

                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(inputs)
                batch_loss = criterion(preds, labels.float())  # Refer: https://discuss.pytorch.org/t/multi-label-binary
                # -classification-result-type-float-cant-be-cast-to-the-desired-output-type-long/117915
                batch_loss.backward()
                optimizer.step()

            running_loss += batch_loss.item()
            if targets is None:
                targets = labels
                predictions = preds
            else:
                targets = torch.cat((targets, labels))
                predictions = torch.cat((predictions, preds))
        log_metrics(targets, predictions, prefix="train_")
        # noinspection PyTypeChecker
        mlflow.log_metric("train_loss", np.mean(running_loss))

        # Evaluate
        val_loss = validate(model, val_loader, criterion, epoch + 1, device)
        # Early Stopping
        if val_loss < best_loss:
            if os.path.exists(model_file):  # checking if there is a file with this name
                os.remove(model_file)  # deleting the file
                torch.save(model.state_dict(), model_file)
                print(f'Model state saved in epoch {epoch + 1}')
            else:
                torch.save(model.state_dict(), model_file)
            patience = args.patience
        else:
            patience -= 1

        if patience == 0:
            break

        scheduler.step(val_loss)


def validate(model, data_loader, criterion, epoch_number, device):
    print(f"Validation epoch {epoch_number}")

    with torch.no_grad():
        targets = None
        predictions = None
        running_loss = 0.0
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)
            batch_loss = criterion(preds, labels.float())
            running_loss += batch_loss.item()
            if targets is None:
                targets = labels
                predictions = preds
            else:
                targets = torch.cat((targets, labels))
                predictions = torch.cat((predictions, preds))
        epoch_loss = np.mean(running_loss)
        log_metrics(targets, predictions, prefix="val_")
        # noinspection PyTypeChecker
        mlflow.log_metric("val_loss", epoch_loss)
        return epoch_loss


def test(model, data_loader, device):
    with torch.no_grad():
        targets = None
        predictions = None
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)
            if targets is None:
                targets = labels
                predictions = preds
            else:
                targets = torch.cat((targets, labels))
                predictions = torch.cat((predictions, preds))
        log_metrics(targets, predictions, prefix="test_")


def main():
    parser = get_parser()
    args = parser.parse_args()
    init_seed(args.seed)
    print(f"Cuda Available: {torch.cuda.is_available()}")
    num_gpus = torch.cuda.device_count()
    print(f"Cuda device count: {num_gpus}")

    experiment = "LCLU"
    mlflow.set_experiment(experiment)
    mlflow.start_run()
    run = mlflow.active_run()
    print(f"run_id: {run.info.run_id}")
    mlflow.log_params(vars(args))
    mlflow.set_tag("notes", args.notes)
    mlflow.log_param("num-gpus", num_gpus)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device is: {device}')

    lbl_corr_vec = None
    with open('./corr_mats/kosovo.npy', 'rb') as f:
        lbl_corr_vec = torch.from_numpy(np.load(f).flatten()).float()
    lbl_corr_vec = lbl_corr_vec.to(device)

    model = loader.load_model(lbl_corr_vec, args=args, device=device)

    num_params = utils.count_params(model)
    print(f'Total number of model parameters: {num_params}')
    mlflow.log_param("num-params", num_params)

    for param in model.parameters():
        param.requires_grad = False
    model.module.model._fc.weight.requires_grad = True
    model.module.model._fc.bias.requires_grad = True
    model.module.lbl_layers[0].weight.requires_grad = True
    model.module.lbl_layers[0].bias.requires_grad = True
    model.module.lbl_layers[1].weight.requires_grad = True
    model.module.lbl_layers[1].bias.requires_grad = True
    model.module.lbl_layers[2].weight.requires_grad = True
    model.module.lbl_layers[2].bias.requires_grad = True

    if not args.test:
        train_loader, val_loader = loader.load_data(args, phase='train')
        if args.loss == 'focal':
            criterion = loss_fns.FocalLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        if args.optimizer == 'adam':
            optimizer = optim.Adam([
                {'params': model.module.model._fc.parameters()},
                {'params': model.module.lbl_layers[0].parameters()},
                {'params': model.module.lbl_layers[1].parameters()},
                {'params': model.module.lbl_layers[2].parameters()}
            ], lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = optim.SGD([
                {'params': model.module.model._fc.parameters()},
                {'params': model.module.lbl_layers[0].parameters()},
                {'params': model.module.lbl_layers[1].parameters()},
                {'params': model.module.lbl_layers[2].parameters()}
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        train(model, train_loader, val_loader, args, criterion, optimizer, scheduler, device)
    else:
        if args.saved_model is None:
            print("Use the argument --saved-model to test a saved model")
        else:
            test_loader = loader.load_data(args, phase='test')
            test(model, test_loader, device)

    mlflow.end_run()


if __name__ == '__main__':
    main()
