import argparse
import sys

import numpy as np
import torch

import options as option

sys.path.insert(0, "../../")
import utils as util

parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="options/setting1/train/train_setting1_x4.yml",
    help="Path to options YMAL file.",
)
args = parser.parse_args()
opt = option.parse(args.opt, is_train=True)

setting = opt["degradation"]

is_iso = True if setting["rate_iso"] == 1 else False

batch_ker = util.random_batch_kernel(
    batch=30000,
    l=setting["ksize"],
    sig_min=setting["sig_min"],
    sig_max=setting["sig_max"],
    rate_iso=setting["rate_iso"],
    random_disturb=setting["random_disturb"],
    tensor=False,
)
print("batch kernel shape: {}".format(batch_ker.shape))
b = np.size(batch_ker, 0)
batch_ker = batch_ker.reshape((b, -1))
pca_matrix = util.PCA(batch_ker, k=setting["code_length"]).float()

iso_name = "" if is_iso else "_aniso"
matrix_name = "pca_{}matrix_x{}".format(iso_name, opt["scale"])
print("{} shape: {}".format(matrix_name, pca_matrix.shape))

torch.save(pca_matrix, "./{}.pth".format(matrix_name))
print("Save PCA matrix at: ./{}.pth".format(matrix_name))
