#!/bin/bash

python -m main --root-dir "/home/jaspreet/Documents/DFKI/netscratch/dataset" --num-workers 1 --batch-size 8 --out-dir "/home/jaspreet/Documents/DFKI/netscratch/jsingh/lclu_out" --model-choice "resize" --train-countries "All_Bands" --test-countries "All_Bands"