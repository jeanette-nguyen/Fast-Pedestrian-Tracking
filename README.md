# Description
This is Pedestrian Detection with Deep Compression developed by  
Kevin Le ktl014@ucsd.edu ,
Jeanette Nguyen jnn004@ucsd.edu ,
Thomas An tan@eng.ucsd.edu ,
Christian Gunther cjgunthe@end.ucsd.edu

# Requirements
requires Python v3

requires PyTorch >=0.4  

install cupy:  
`$ pip install cupy-cuda80`

install other dependencies:   
`$ pip install -r requirements.txt`

Optional, but strongly recommended: build cython code nms_gpu_post:
```
$ cd model/utils/nms/
$ python build.py build_ext --inplace
```

# Code Organization
[demo.ipynb](demo.ipynb) - Run a demo of our code  
[Train.ipynb](Train.ipynb) - Run the training of our model  
<pre>
core/  
-- data/  
|  -- __init__.py - Faster RCNN data init  
|  -- d_util.py -  Faster RCNN data utils
|  -- dataloader.py - Faster RCNN dataloader
-- __init__.py - Faster RCNN init
-- logger.py - Faster RCNN logger for data and model

data/
-- __init__.py - data init
-- __caltech_dataset.py - Split the data into different sets
-- dataset.py - Preprocess data and get it in the necessary format
-- util.py - Utility functions to preprocess the data

dataset/
-- data_test.csv - Caltech Dataset Test sets 06-10
-- data_train.csv - Caltech Dataset Train sets 00-05
-- data_val.csv - Caltech Dataset Validation part of sets 00-05

misc/
-- convert_caffe_pretrain.py - Convert caffe pretrained weights to be usable by our model

model/
-- compression/
|  -- PruningClasses.py - Module to prune weights from the model
|  -- __init__.py - model compression init
|  -- prune_utils.py - Utils for pruning
|  -- quantization.py - Function to quantize the weights
|  -- vgg16.py - Modified VGG Network to support pruning and quantization of weights
-- model_deprecated/
|  -- FasterRCNN.py - Unused model
|  -- RPN.py - Unused region propasal network
|  -- VGG16.py - Unused model
-- utils/
|  -- nms/
|  |  -- build/temp.linux-x86_64-3.6/
|  |  |  -- _nms_gpu_post.o
|  |  -- __init__.py - Faster RCNN non-maximum suppression init
|  |  -- _nms_gpu_post.c - Faster RCNN nms C extension
|  |  -- _nms_gpu_post.cpython-36m-x86_64-linux-gnu.so
|  |  -- _nms_gpu_post.pyx
|  |  -- _nms_gpu_post_py.py - Faster RCNN cms code
|  |  -- build.py - Faster RCNN Build the cython code for nms
|  |  -- non_maximum_suppression.py - Faster RCNN Suppress bounding boxes according to their IoUs
|  -- __init__.py - Faster RCNN model utils init
|  -- bbox_tools.py - Generate bounding boxes and perform calculations on them
|  -- creator_tool.py - Generate proposal regions
|  -- roi_cupy.py - Faster RCNN generate regions of interest
-- utils/deprecated/
|  -- bbox.py - Unused bounding box file
|  -- config.py - Unused config file
|  -- network.py - Unused network file
|  -- proposal_layer.py - Unused proposal layer file
-- __init__.py - Faster RCNN model init
-- faster_rcnn.py - Faster RCNN model
-- faster_rcnn_vgg16.py - Faster RCNN model based on vgg16
-- region_proposal_network.py - Region Proposal Network introduced in Faster R-CNN
-- roi_module.py - Region of Interest Module

tools/
-- __init__.py - tools init
-- benchmark_model.py - Measures framerate of the evaluation
-- plot_annotations.py - Draw bounding box annotations on images
-- preparte_dataset.py - Generate data csv files
-- visualize_dataset.ipynb - Display images with bounding boxes

utils/
-- __init__.py - utils init
-- array_tool.py - Tools to convert specified type
-- config.py - Settings to configure the model 
-- constants.py - Declared constants
-- eval_tool.py - Tools to evaluate the accuracy of our detections
-- size_utils.py - Get size of our model
-- vis_tool.py - Tools to help visualize the images with bounding boxes

__init__.py - Faster RCNN init

demo.ipynb - Demo of our evaluation to detect some pedestrians

eval.py - Evaluate our model's MAP

prune.py - Trains a model with pruned weights

quantize.py - Quantizes the model weights

requirements.txt - Requirements that must be installed for the model to run

train.ipynb - Notebook to rerun the training if need be

train.py - Run the code to train our Faster RCNN model

trainer.py - Wrapper for conveniently training

</pre>

# Prepare Data
To download the dataset copy the github repository:
`$ git clone https://github.com/mitmul/caltech-pedestrian-dataset-converter.git`

To prepare the dataset, run prepare_dataset.py in the tools directory:
`$ python prepare_dataset.py --path=/path/to/output --data-dir=/datasets/ee285f-public/caltech_pedestrians_usa/ `

Output for example run:
`$ python prepare_dataset.py --path=../data --data-dir=/datasets/ee285f-public/caltech_pedestrians_usa/`

Note, preparing the data is not necessary. We have included the our own prepared csv files (train, val, and test) in the repository as well in the [dataset folder](https://github.com/ktl014/Fast-Pedestrian-Tracking/tree/master/dataset).

# Pretrained Model
Download the pretrained weights from our model [here](https://drive.google.com/open?id=1S2McCJo-od-BvVDGaOw8_NICjJvF9rDe).
