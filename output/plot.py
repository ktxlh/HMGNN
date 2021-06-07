
from os import getcwd, mkdir
from os.path import isdir, join

import matplotlib.pyplot as plt

def plot(exp_name, value_dict, scatter=False):
    save_path = join(getcwd(), f'{exp_name}.png')
    for label, values in value_dict.items():
        if scatter:
            iters = [i for i, v in enumerate(values) if v >= 0]
            values = [v for v in values if v >= 0]
            plt.scatter(iters, values, label=label)
        else:
            plt.plot(values, label=label)
    plt.xlabel('Iterations')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close() # Close a figure window

def process(kv):
    with open("results.txt", 'r') as f:
        lines = f.readlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("model_name"):
                model_name = lines[i].strip().split(' ')[-1]
                break
        for name in ["train_loss_list ", "val_loss_list   ", "val_acc_list    "]:
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith(name):
                    kv[name.replace(' ', '')] = lines[i].strip().replace(name, '')
                    break
    return model_name

kv = dict()
plot(process(kv), {k: [float(i) for i in v.split()] for k, v in kv.items()}, scatter=True)
