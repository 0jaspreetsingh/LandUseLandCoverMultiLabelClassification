import torch
import numpy as np
import torchvision.transforms as transforms

def normalize_B10(B10):
    nB10 = transforms.Normalize(
        mean=(0.0540, 0.0382, 0.0273, 0.0328), std=(3.5444, 2.6132, 1.8816, 6.0683))
    return nB10(B10)


def normalize_B20(B20):
    nB20 = transforms.Normalize(
        mean=(0.0296, 0.0328, 0.0353, 0.0323, 0.0182, 0.0143), std=(2.4376, 4.9182, 5.8950, 6.2497, 3.0722, 1.7604))
    return nB20(B20)


def normalize_B60(B60):
    nB60 = transforms.Normalize(
        mean=(0.0452, 0.0445), std=(2.9666, 6.2927))
    return nB60(B60)

def merge_bands_transforms(sample: dict):
    B60 = torch.from_numpy(np.stack((sample["B01"], sample["B09"])))
    B60 = normalize_B60(B60)
    B10 = torch.from_numpy(np.stack((sample["B02"], sample["B03"], sample["B04"], sample["B08"])))
    B10 = normalize_B10(B10)
    B20 = torch.from_numpy(
        np.stack((sample["B05"], sample["B06"], sample["B07"], sample["B8A"], sample["B11"], sample["B12"])))
    B20 = normalize_B20(B20)
    label = torch.from_numpy(sample["label"])
    data = (B10 , B20 , B60)
    # return {"B10": B10, "B20": B20, "B60": B60, "label": label}
    return {"data": data, "label": label}

def intermediate_fusion_transforms(sample: dict):
    resize_B60 = transforms.Resize(size=(30, 30))
    B60 = torch.from_numpy(np.stack((sample["B01"], sample["B09"])))
    B60 = normalize_B60(B60)
    B60 = resize_B60(B60)
    B10 = torch.from_numpy(np.stack((sample["B02"], sample["B03"], sample["B04"], sample["B08"])))
    B10 = normalize_B10(B10)
    B20 = torch.from_numpy(
        np.stack((sample["B05"], sample["B06"], sample["B07"], sample["B8A"], sample["B11"], sample["B12"])))
    B20 = normalize_B20(B20)
    label = torch.from_numpy(sample["label"])
    data = (B10 , B20 , B60)
    # return {"B10": B10, "B20": B20, "B60": B60, "label": label}
    return {"data": data, "label": label}

def resize_transforms(sample: dict):
    resize = transforms.Resize(size=(120, 120))
    B60 = torch.from_numpy(np.stack((sample["B01"], sample["B09"])))
    B60 = normalize_B60(B60)
    B60 = resize(B60)
    B10 = torch.from_numpy(np.stack((sample["B02"], sample["B03"], sample["B04"], sample["B08"])))
    B10 = normalize_B10(B10)
    B10 = resize(B10)
    B20 = torch.from_numpy(
        np.stack((sample["B05"], sample["B06"], sample["B07"], sample["B8A"], sample["B11"], sample["B12"])))
    B20 = normalize_B20(B20)
    B20 = resize(B20)
    data = torch.cat((B10, B20, B60), dim=0)
    label = torch.from_numpy(sample["label"])
    return {"data": data, "label": label}

    # return {"B10": B10, "B20": B20, "B60": B60, "label": label}


def normalize(sample: dict):
     ## normalize batch
    sample["B01"] = sample["B01"] +1e-6
    sample["B02"] = sample["B02"] +1e-6
    sample["B03"] = sample["B03"] +1e-6
    sample["B04"] = sample["B04"] +1e-6
    sample["B05"] = sample["B05"] +1e-6
    sample["B06"] = sample["B06"] +1e-6
    sample["B07"] = sample["B07"] +1e-6
    sample["B08"] = sample["B08"] +1e-6
    sample["B8A"] = sample["B8A"] +1e-6
    sample["B09"] = sample["B09"] +1e-6
    sample["B11"] = sample["B11"] +1e-6
    sample["B12"] = sample["B12"] +1e-6
    ##
    ## normalize batch
    sample["B01"] = (sample["B01"] - sample["B01"].min())/(sample["B01"].max() - sample["B01"].min() + 1e-6)
    sample["B02"] = (sample["B02"] - sample["B02"].min())/(sample["B02"].max() - sample["B02"].min() + 1e-6)
    sample["B03"] = (sample["B03"] - sample["B03"].min())/(sample["B03"].max() - sample["B03"].min() + 1e-6)
    sample["B04"] = (sample["B04"] - sample["B04"].min())/(sample["B04"].max() - sample["B04"].min() + 1e-6)
    sample["B04"] = (sample["B04"] - sample["B04"].min())/(sample["B04"].max() - sample["B04"].min() + 1e-6)
    sample["B05"] = (sample["B05"] - sample["B05"].min())/(sample["B05"].max() - sample["B05"].min() + 1e-6)
    sample["B06"] = (sample["B06"] - sample["B06"].min())/(sample["B06"].max() - sample["B06"].min() + 1e-6)
    sample["B07"] = (sample["B07"] - sample["B07"].min())/(sample["B07"].max() - sample["B07"].min() + 1e-6)
    sample["B08"] = (sample["B08"] - sample["B08"].min())/(sample["B08"].max() - sample["B08"].min() + 1e-6)
    sample["B8A"] = (sample["B8A"] - sample["B8A"].min())/(sample["B8A"].max() - sample["B8A"].min() + 1e-6)
    sample["B09"] = (sample["B09"] - sample["B09"].min())/(sample["B09"].max() - sample["B09"].min() + 1e-6)
    sample["B11"] = (sample["B11"] - sample["B11"].min())/(sample["B11"].max() - sample["B11"].min() + 1e-6)
    sample["B12"] = (sample["B12"] - sample["B12"].min())/(sample["B12"].max() - sample["B12"].min() + 1e-6)
    return sample


def normalized_transform(sample: dict):
    ## training data min max values
    sample["B01"] = (sample["B01"])/(19348)
    sample["B02"] = (sample["B02"])/(20566)
    sample["B03"] = (sample["B03"])/(18989)
    sample["B04"] = (sample["B04"])/(17881)
    sample["B05"] = (sample["B05"])/(17374)
    sample["B06"] = (sample["B06"])/(17160)
    sample["B07"] = (sample["B07"])/(16950)
    sample["B08"] = (sample["B08"])/(16708)
    sample["B8A"] = (sample["B8A"])/(16627)
    sample["B09"] = (sample["B09"])/(16204)
    sample["B11"] = (sample["B11"])/(15465)
    sample["B12"] = (sample["B12"])/(15273)
    return sample

def normalize_RGB(B10):
    nB10 = transforms.Normalize(
        mean=(0.0540, 0.0382, 0.0273), std=(3.5444, 2.6132, 1.8816))
    return nB10(B10)

def rgb_only_transforms(sample: dict):
    sample["B02"] = (sample["B02"])/(256)
    sample["B03"] = (sample["B03"])/(256)
    sample["B04"] = (sample["B04"])/(256)
    B10 = torch.from_numpy(np.stack((sample["B02"], sample["B03"], sample["B04"])))
    label = torch.from_numpy(sample["label"])
    ## invalid values in B20 and B60
    return {"B10": B10, "B20": torch.zeros(1), "B60": torch.zeros(1), "label": label}

def rgb_normalized_transforms(sample: dict):
    B10 = torch.from_numpy(np.stack((sample["B02"], sample["B03"], sample["B04"])))
    B10 = normalize_RGB(B10)
    label = torch.from_numpy(sample["label"])
    ## invalid values in B20 and B60
    return {"B10": B10, "B20": torch.zeros(1), "B60": torch.zeros(1), "label": label}
