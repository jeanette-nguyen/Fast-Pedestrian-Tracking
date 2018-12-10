import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

def sparse_mx_to_tensor(sparse_mx):
    print("Turning Sparse")
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def quantize(model, bits=5, verbose=False):
    """
    Performs quantization on the weights through the use of KMeans
    Args:
        model: PyTorch model to pass in
        bits: number of bits to encode the weights
    """
    for sequential in model.children():
        for module in sequential:
            try:
                dev = module.weight.device
                weight = module.weight.data.cpu().numpy()
                shape = weight.shape
                print(f"Quantizing: {str(module)}")
                print(shape)
                if len(shape) == 2:
                    matrix = coo_matrix(weight)
                    mmin = min(matrix.data)
                    mmax = max(matrix.data)
                    space = np.linspace(mmin, mmax, num=2**bits)
                    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True, algorithm="full")
                    kmeans.fit(matrix.data.reshape(-1, 1))
                    new_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                    matrix.data = new_weights
                    tensor = sparse_mx_to_tensor(matrix)
                    module.weight.data = tensor.to(dev)
                    # module.weight.data = torch.from_numpy(matrix.toarray()).to(dev)
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
                                if verbose:
                                    print("No weights in {}{} of {}".format(out_c, in_c, str(module)))
                print("Done quantizingg {}".format(str(module)))
            except:
                if verbose:
                    print("No weights in module {}".format(str(module)))
                