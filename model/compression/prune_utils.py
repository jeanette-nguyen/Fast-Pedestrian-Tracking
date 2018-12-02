import numpy as np

def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)
def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)

def print_nonzeros(model):
    nonzero = total = 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = param.data.cpu().numpy()
        nonzero_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nonzero_count
        total += total_params
        print(f"{name:20} | nonzeros = {nonzero_count:7} / {total_params:7} ({100. * nonzero_count / total_params:6.2f}%) | total_pruned = {total_params - nonzero_count : 7} | shape = {tensor.shape}")
    print(f"Active: {nonzero}, pruned : {total - nonzero}, total: {total}, Compressed: {100. * nonzero / total:6.2f}%")

                