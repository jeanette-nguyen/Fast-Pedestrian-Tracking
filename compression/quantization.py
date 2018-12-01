import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

def quantize(model, bits=5):
    """
    Performs quantization on the weights through the use of KMeans
    Args:
        model: PyTorch model to pass in
        bits: number of bits to encode the weights
    """
    for module in model.children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        if len(shape) == 2:
            if r < c:
                matrix = csr_matrix(weight)
            else:
                matrix = csc_matrix(weight)
            mmin = min(matrix.data)
            mmax = max(matrix.data)
            space = np.linspace(mmin, mmax, num=2**bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(matrix.data.reshape(-1, 1))
            new_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            matrix.data = new_weights
            module.weight.data = torch.from_numpy(matrix.toarray()).to(dev)
        else:
            assert len(shape) == 4, "must be 2 or 4 for FC layers or convolution filters"
            for in_c in range(shape[0]):
                for out_c in range(shape[1]):
                    k = weight[in_c, out_c, :, :]
                    shape_k = k.shape
                    matrix = csr_matrix(weight[in_c, out_c, :, :]) if shape_k[0] < shape_k[1] else csc_matrix(weight)
                    mmin = min(matrix.data)
                    mmax = max(matrix.data)
                    space = np.linspace(mmin, mmax, num=2**bits)
                    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True, algorithm="full")
                    kmeans.fit(matrix.data.reshape(-1, 1))
                    new_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                    matrix.data = new_weights
                    module.weight[in_c, out_c].data = torch.from_numpy(matrix.toarray()).to(dev)