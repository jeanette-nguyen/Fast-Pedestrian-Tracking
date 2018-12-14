import os
from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    mask_lin = True
    mask_conv = False
    voc_data_dir = '/datasets/home/98/898/cjgunthe/Fast-Pedestrian-Tracking/dataset2/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 4
    test_num_workers = 4

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.
    mask = True
    use_simple = False
    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # visualization
    env = 'faster_rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14
    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'
    test_num = 10000
    # model
<<<<<<< HEAD
    load_path = None#'/datasets/home/98/898/cjgunthe/Fast-Pedestrian-Tracking/checkpoints/fasterrcnn_12131543_0_test_run'
    model_name = 'test_run'
=======
    load_path = None
>>>>>>> 35544e1a501a16254cb8ecfd78b4db4d33ae4ade

    # benchmark
    benchmark_path = None
    '''
    Pruning Configs
    '''
    sparse_dense = False
    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'


    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _parse_all(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
assert opt.mask_conv == False, "Only supports pruning FC layers now"
assert os.path.exists(opt.voc_data_dir)
