# Description
This is Pedestrian Detection with Deep Compression developed by  
Kevin Le ktl014@ucsd.edu ,
Jeanette Nguyen jnn004@ucsd.edu ,
Thomas An tan@eng.ucsd.edu ,
Christian Gunther cjgunthe@end.ucsd.edu

# Requirements
requires Python v3

requires PyTorch >=0.4  

install cupy, you can install via pip install cupy-cuda80 or(cupy-cuda90,cupy-cuda91, etc).

install other dependencies: pip install -r requirements.txt

Optional, but strongly recommended: build cython code nms_gpu_post:  
cd model/utils/nms/  
python build.py build_ext --inplace

# Code Organization
Training: train.ipynb

Demo: demo.ipynb



# Prepare Data
To download the dataset copy the github repository:

`$ git clone https://github.com/mitmul/caltech-pedestrian-dataset-converter.git`

To prepare the dataset, run prepare_dataset.py in the tools directory:

`$ python prepare_dataset.py --path=/path/to/output --data-dir=/datasets/ee285f-public/caltech_pedestrians_usa/ `

Output for example run:

`$ python prepare_dataset.py --path=../data --data-dir=/datasets/ee285f-public/caltech_pedestrians_usa/`

```
data/

-- data_train.csv

-- data_val.csv

-- data_test.csv

-- dataset.log
```

