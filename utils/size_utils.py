import numpy as np
import torch

def get_size(model, sparse=False):
    tot_size = 0
    for n, m in model.named_modules():
        if hasattr(m, 'weight'):
            if hasattr(m, 'sparse') and m.sparse:
                coalesced = m.weight.data.coalesce().cpu()
                if str(torch.__version__) <= "0.4.1":
                    indices_size = coalesced._indices().numpy().nbytes
                    values_size = coalesced._values().numpy().nbytes
                else:
                    indices_size = coalesced.indices().numpy().nbytes
                    values_size = coalesced.values().numpy().nbytes
                s = indices_size + values_size
                tot_size += s
            else:
                numpy_w = m.weight.data.cpu().numpy()
                s = numpy_w.nbytes
                tot_size += s
            print(f"Size of {n}: {s} bytes ({s / 1000000.} MBytes)")
    print(f"Total Size: {tot_size}")
    