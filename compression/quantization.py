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
        (r, c) = weight.shape
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