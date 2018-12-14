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

def print_nonzeros_mask(model):
    nonzero = total = 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            tensor = param.data.cpu().numpy()
            nnz = np.count_nonzero(tensor)
            nonzero += nnz
            total_params = np.prod(tensor.shape)
            total += total_params
            print(f"""{name:20} | nonzeros = {nnz:7} / {total_params:7} """
                  f"""({100. * nnz / total_params:6.2f}%) | total_pruned = """
                  f"""{total_params - nnz : 7} | shape = {tensor.shape}""")
        else:
            continue
    print(f"Active: {nonzero}/{total}, pruned : {total - nonzero} ({100. * (total - nonzero)/total : 6.2f}%)")

def print_nonzeros(model, only_pruned=False):
    nonzero = total = 0
    if only_pruned:
        for modname, module in model.named_modules():
            if "Mask" in str(module) and modname and "Sequential" not in str(module):
                for name, param in module.named_parameters():
                    if "mask" in name:
                        continue
                    tensor = param.data.cpu().numpy()
                    nonzero_count = np.count_nonzero(tensor)
                    total_params = np.prod(tensor.shape)
                    nonzero += nonzero_count
                    total += total_params
                    if "Mask" in str(module) and modname and "Sequential" not in str(module):
                        print(f"""{name:20} | nonzeros = {nonzero_count:7} / {total_params:7} """
                              f"""({100. * nonzero_count / total_params:6.2f}%)"""
                              f""" | total_pruned = {total_params - nonzero_count : 7} | shape = {tensor.shape}""")
        print(f"Active: {nonzero}, pruned : {total - nonzero}, total: {total}, Compressed: {100. * nonzero / total:6.2f}%")
    else:
        for name, param in model.named_parameters():
            if 'mask' in name:
                continue
            tensor = param.data.cpu().numpy()
            nonzero_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nonzero_count
            total += total_params
            print(f"""{name:20} | nonzeros = {nonzero_count:7} / {total_params:7}"""
                  f""" ({100. * nonzero_count / total_params:6.2f}%) | """
                  f"""total_pruned = {total_params - nonzero_count : 7} |"""
                  f""" shape = {tensor.shape}""")
        print(f"""Active: {nonzero}, pruned : {total - nonzero}, total: {total},"""
              f""" Compressed: {100. * nonzero / total:6.2f}%""")

                