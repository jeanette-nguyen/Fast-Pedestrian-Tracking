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
[train.ipynb](train.ipynb) - Run the training of our model  
<pre>
core/  
-- data/  
|  -- __init__.py - Faster RCNN data init  
|  -- d_util.py -  Faster RCNN data utils
|  -- dataloader.py - Faster RCNN dataloader
-- __init__.py - Faster RCNN init
-- logger.py - Faster RCNN logger for data and model

data/
-- __init__.py - Faster RCNN 
-- __caltech_dataset.py - 
-- dataset.py - 
-- util.py - 

dataset/
-- data_test.csv - Caltech Dataset Test sets 06-10
-- data_train.csv - Caltech Dataset Train sets 00-05
-- data_val.csv - Caltech Dataset Validation part of sets 00-05

misc/
-- convert_caffe_pretrain.py - Convert caffe pretrained weights to be usable by our model ?

model/
-- compression/
|  -- PruningClasses.py - 
|  -- __init__.py - model compression init
|  -- prune_utils.py - 
|  -- quantization.py - 
|  -- vgg16.py - 
-- model_deprecated/
|  -- FasterRCNN.py -
|  -- RPN.py - 
|  -- VGG16.py -
-- utils/
|  -- nms/
|  |  -- build/temp.linux-x86_64-3.6/
|  |  |  -- _nms_gpu_post.o
|  |  -- __init__.py - Faster RCNN non-maximum suppression init
|  |  -- _nms_gpu_post.c - 
|  |  -- _nms_gpu_post.cpython-36m-x86_64-linux-gnu.so - 
|  |  -- _nms_gpu_post.pyx -
|  |  -- _nms_gpu_post_py.py - 
|  |  -- build.py - 
|  |  -- non_maximum_suppression.py - 
|  -- __init__.py - Faster RCNN model utils init
|  -- bbox_tools.py - 
|  -- creator_tool.py - 
|  -- roi_cupy.py - 
-- utils/deprecated/
|  -- bbox.py -
|  -- config.py -
|  -- network.py -
|  -- proposal_layer.py -
-- __init__.py - Faster RCNN model init
-- faster_rcnn.py - Faster RCNN model
-- faster_rcnn_vgg16.py - Faster RCNN model
-- region_proposal_network.py
-- roi_module.py

tools/
-- __init__.py - tools init
-- benchmark_model.py - 
-- plot_annotations.py - 
-- preparte_dataset.py - 
-- visualize_dataset.ipynb - Display images with bounding boxes (Figures ex)

utils/
-- __init__.py - utils init
-- array_tool.py - 
-- config.py - Settings to configure the model 
-- constants.py - 
-- eval_tool.py - 
-- size_utils.py -
-- vis_tool.py - 

__init__.py - Faster RCNN init

eval.py - 

prun.py -

quantize.py - 

requirements.txt - Requirements that must be installed for the model to run

train.ipynb - 

train.py - Run the code to train our model

trainer.py - 

</pre>

# Prepare Data
To download the dataset copy the github repository:
`$ git clone https://github.com/mitmul/caltech-pedestrian-dataset-converter.git`

To prepare the dataset, run prepare_dataset.py in the tools directory:
`$ python prepare_dataset.py --path=/path/to/output --data-dir=/datasets/ee285f-public/caltech_pedestrians_usa/ `

Output for example run:
`$ python prepare_dataset.py --path=../data --data-dir=/datasets/ee285f-public/caltech_pedestrians_usa/`


