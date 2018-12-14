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
    print("\n=====!![NOTE]: Only support for quantization of linear layers supported=====\n")
    model.sparse = True
    for sequential in model.children():
        if "maskedlinear" in str(sequential).lower():
            for name, module in sequential.named_modules():
                if hasattr(module, 'sparse'):
                    module.sparse = True
                if name and "Sequential" not in str(module) and "masked" in str(module).lower():
                    dev = module.weight.device
                    weight = module.weight.data.cpu().numpy()
                    shape = weight.shape
                    print(f"Trying to quantize: {str(module)} of size: {shape} ({shape[0]*shape[1]} parameters)")
                    try:
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
                        module.sparse = True
                        print("Done quantizing {}".format(str(module)))
                    except:
                        if verbose:
                            print("No weights or mask in module {}".format(str(module)))
    return model