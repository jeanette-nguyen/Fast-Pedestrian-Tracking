# Description
This is Pedestrian Detection with Deep Compression developed by 

Kevin Le, ktl014@ucsd.edu
Jeanette Nguyen, jnn004@ucsd.edu
Thomas An, tan@eng.ucsd.edu
Christian Gunther, cjgunthe@end.ucsd.edu

# Requirements
requires PyTorch >=0.4

install PyTorch >=0.4 with GPU (code are GPU-only), refer to official website

install cupy, you can install via pip install cupy-cuda80 or(cupy-cuda90,cupy-cuda91, etc).

install other dependencies: pip install -r requirements.txt

Optional, but strongly recommended: build cython code nms_gpu_post:
cd model/utils/nms/
python build.py build_ext --inplace
cd -

# Code Organization
