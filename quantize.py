from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.config import opt
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
import torch
from trainer import FasterRCNNTrainer
from model.compression import quantization
from utils.size_utils import get_size
import argparse

parser = argparse.ArgumentParser(description="Arguments for quantization")
parser.add_argument("--bits", "-b", type=int, default=5, help="Number of bits to quantize")
parser.add_argument("--verbose", default=True, action='store_false', help="Print verbose or not")
parser.add_argument("--save_path", type=str, default="./checkpoints/quantized_model.model", help="Model save path")
parser.add_argument("--load_path", type=str, default="./checkpoints/pruned_model.model", help="Pruned model to quantize")
args = parser.parse_args()

def main():
    faster_rcnn = FasterRCNNVGG16(mask=opt.mask)
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    assert os.path.isfile(args.load_path), f"Need valid checkpoint, {args.load_path} not found"
    trainer.load(args.load_path)
    '''
    Check to make sure weights are dense
    '''
    for n, m in trainer.named_modules():
        if hasattr(m, 'sparse'):
            m.sparse = False
    for n, m in trainer.named_modules():
        if hasattr(m, 'weight'):
            if m.weight.is_sparse:
                print("Weights are already sparse")
                return
    print("\n\n=========SIZE BEFORE=============")
    get_size(trainer)
    trainer.quantize(bits=args.bits, verbose=args.verbose)
    print("\n\n=========SIZE AFTER==============")
    get_size(trainer)
    trainer.save(save_path=args.save_path)


if __name__ == "__main__":
    main()