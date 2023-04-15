import torch
import torch.nn as nn
from datadings.reader import MsgpackReader
from datadings.torch import IterableDataset
from datadings.torch import Compose
import os
from datadings.torch import CompressedToPIL
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ConvertImageDtype
from tifffile import imread
from io import BytesIO
from torch.utils.data import ChainDataset
from model.resnet_esamble import ResnetEsamble
from model.resnet_intermediate_fusion import ResnetIntermediateFusion
from transforms import resize_transforms , merge_bands_transforms, intermediate_fusion_transforms

from model.efnet_lclu_extract import Model
from model.efnet_lclu_esamble import EsambleModel
import model.efnet_lclu_esamble_withoutB60 as without_B60
from model.efnet_lclu_intermediate_fusion import IntermediateFustionModel
from model.alexnet import AlexNet
from model.resnet import Resnet

class TiffToNDArray:
    def __call__(self, buf):
        img = imread(BytesIO(buf))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def load_data(args):
    batch_size = args.batch_size
    num_workers = args.num_workers
    name_prefix = 'set'
    if args.model_choice == 'rgb':
        transforms = {'data': Compose(
            CompressedToPIL(),
            ToTensor(),
        )}
        name_prefix = ''
    elif args.model_choice == 'resize' or args.model_choice == 'alexnet' or args.model_choice == 'resnet':
        transforms = resize_transforms
    elif args.model_choice == 'merge' or args.model_choice == 'merge_without_B60' or args.model_choice == 'resnet_esamble':
        transforms = merge_bands_transforms
    elif args.model_choice == 'intermediate_fusion' or args.model_choice == 'resnet_intermediate':
        transforms = intermediate_fusion_transforms
    else:
        transforms = {'data': Compose(
            TiffToNDArray(),
            ToTensor(),
            ConvertImageDtype(torch.float),
        )}
        

    train_countries = args.train_countries

    train_datasets = []
    for country in train_countries:
        if args.chunk_size is not None:
            train_file = os.path.join(args.root_dir, country, args.chunk_size+"_train_data.msgpack")
        else:
            train_file = os.path.join(args.root_dir, country, f"train_data{name_prefix}.msgpack")
        train_reader = MsgpackReader(train_file)
        train_ds = IterableDataset(train_reader, transforms=transforms, batch_size=batch_size, )
        train_datasets.append(train_ds)
    train_dataset = ChainDataset(train_datasets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
                              persistent_workers=True)

    val_datasets = []
    for country in train_countries:
        val_file = os.path.join(args.root_dir, country, f"val_data{name_prefix}.msgpack")
        val_reader = MsgpackReader(val_file)
        val_ds = IterableDataset(val_reader, transforms=transforms, batch_size=batch_size, )
        val_datasets.append(val_ds)
    val_dataset = ChainDataset(val_datasets)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers,
                            persistent_workers=True)

    test_countries = args.test_countries
    test_datasets = []
    for country in test_countries:
        test_file = os.path.join(args.root_dir, country, f"test_data{name_prefix}.msgpack")
        test_reader = MsgpackReader(test_file)
        test_ds = IterableDataset(test_reader, transforms=transforms, batch_size=batch_size, )
        test_datasets.append(test_ds)
    test_dataset = ChainDataset(test_datasets)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4,
                             persistent_workers=True)
    return train_loader, val_loader, test_loader


def load_model(lbl_corr_vec, args, device):
    if args.model_choice == 'rgb':
        model = Model(device, variant="efficientnet-b5", lbl_corr_vec=lbl_corr_vec, n_channels=3, n_classes=19)
    elif args.model_choice == 'resize':
        model = Model(device, variant="efficientnet-b5", lbl_corr_vec=lbl_corr_vec, n_channels=12, n_classes=19)
    elif args.model_choice == 'merge':
        model = EsambleModel(device, variant="efficientnet-b5", lbl_corr_vec=lbl_corr_vec, n_classes=19)
    elif args.model_choice == 'merge_without_B60':
        model = without_B60.EsambleModel(device, variant="efficientnet-b5", lbl_corr_vec=lbl_corr_vec, n_classes=19)
    elif args.model_choice == 'intermediate_fusion':
        model = IntermediateFustionModel(device, variant="efficientnet-b5", lbl_corr_vec=lbl_corr_vec, n_channels=4, n_classes=19)
    elif args.model_choice == 'alexnet':
        model = AlexNet(device, in_channels=12, num_classes = 19)
    elif args.model_choice == 'resnet':
        model = Resnet(device=device, n_channels=12, n_classes= 19)
    elif args.model_choice == 'resnet_esamble':
        model = ResnetEsamble(device=device, n_classes=19)
    elif args.model_choice == 'resnet_intermediate':
        model = ResnetIntermediateFusion(device=device, n_classes=19)
    else:
        model = Model(device, variant="efficientnet-b5", lbl_corr_vec=lbl_corr_vec, n_channels=4, n_classes=19)

    model = nn.DataParallel(model)

    if args.saved_model is not None:
        state_dict = torch.load(args.saved_model)
        if args.imagenet:
            del state_dict['module.model._fc.weight']
            del state_dict['module.model._fc.bias']
            model.load_state_dict(state_dict, strict=False)
            torch.nn.init.xavier_uniform(model.module.model._fc.weight)
            torch.nn.init.zeros_(model.module.model._fc.bias)
        else:
            model.load_state_dict(state_dict)
        print("loaded saved model")

    model.to(device)
    return model
