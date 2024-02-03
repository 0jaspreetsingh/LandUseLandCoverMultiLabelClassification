# LandUseLandCoverMultiLabelClassification

This repository contains the code for [INTER-REGION TRANSFER LEARNING FOR LAND USE LAND COVER CLASSIFICATION](https://isprs-annals.copernicus.org/articles/X-1-W1-2023/881/2023/) published in the *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences*.

## Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@Article{isprs-annals-X-1-W1-2023-881-2023,
  AUTHOR = {Siddamsetty, J. and Stricker, M. and Charfuelan, M. and Nuske, M. and Dengel, A.},
  TITLE = {INTER-REGION TRANSFER LEARNING FOR LAND USE LAND COVER CLASSIFICATION},
  JOURNAL = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  VOLUME = {X-1/W1-2023},
  YEAR = {2023},
  PAGES = {881--888},
  URL = {https://isprs-annals.copernicus.org/articles/X-1-W1-2023/881/2023/},
  DOI = {10.5194/isprs-annals-X-1-W1-2023-881-2023}
}
```
## Motivation

Regular observation of the earth to tackle some of the following problems:   
1. Understanding land use dynamics  
2. Resource management  
3. Urban planning  
4. Environmental monitoring  
5. Disaster risk reduction  

## Dataset Description  
12 Bands  
Bands and pixel resolution in meters:  
60 Meter (20 x 20 pixels) - B01: Coastal aerosol | B09: Water vapor   
10 Meter (120 x 120 pixels) - B02: Blue | B03: Green | B04: Red | B08: NIR  
20 Meter (60 x 60 pixels) - B05: Vegetation red edge | B06: Vegetation red edge | B07: Vegetation red edge | B8A: Narrow NIR | B11: SWIR | B12: SWIR 

https://bigearth.net/static/documents/Description_BigEarthNet-S2.pdf  
https://www.tensorflow.org/datasets/catalog/bigearthnet  

## Few examples 
![plot](./images/samples.png)  

## Preprocessing using Datadings

Accessing tiny files in separate directories is slow.  
Accessing data over the network attached storage slows this further.  
Converting Train, Test and Val splits to datadings files for faster training.  
Link: https://datadings.readthedocs.io/en/latest/index.html  

## 3 Experiments

1. <b>Resizing all Bands to 120X120</b>  
![plot](./images/resized.png)  
2. <b>Intermediate Fusion</b>  
![plot](./images/intermediate_fusion.png)    
3. <b>Late Fusion</b>  
![plot](./images/late_fusion.png)  

## Classification Results
![plot](./images/results.png)  
Deep Multi-Attention Driven Approach paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9096309
