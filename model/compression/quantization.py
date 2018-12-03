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
    mask = model.mask
    for sequential in model.children():
        for module in sequential:
            try:
                dev = module.weight.device
                weight = module.weight.data.cpu().numpy() * module.mask.data.cpu() if mask else module.weight.data.cpu()
                shape = weight.shape
                print(f"Quantizing: {str(module)}")
                print(shape)
                if len(shape) == 2:
                    if shape[0] < shape[1]:
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
                    for out_c in range(shape[0]):
                        for in_c in range(shape[1]):
                            k = weight[out_c, in_c, :, :]
                            try:
                                matrix = csc_matrix(k)
                                mmin = min(matrix.data)
                                mmax = max(matrix.data)
                                space = np.linspace(mmin, mmax, num=2**bits)
                                reshaped_data = matrix.data.reshape(-1, 1)
                                clusters = min(len(space), len(reshaped_data))
                                kmeans = KMeans(n_clusters=clusters, init='k-means++', n_init=1, precompute_distances=True, algorithm="full")
                                kmeans.fit(matrix.data.reshape(-1, 1))
                                new_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                                matrix.data = new_weights
                                module.weight[out_c, in_c].data = torch.from_numpy(matrix.toarray()).to(dev)
                            except:
                                continue
                print("Done quantizingg {}".format(str(module)))
            except:
                print("No weights in module {}".format(str(module)))
                