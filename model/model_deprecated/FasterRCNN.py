import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

import RPN


line = '===================='*2

class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN,self).__init__()

        self.features = self.rpn.features
        #utils.freeze(self.features)
        
def main():
    pass



if __name__== "__main__":
    main()

