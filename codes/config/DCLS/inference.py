import argparse
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="./options/setting1/test/test_setting1_x4.yml",
    help="Path to options YMAL file.",
)
parser.add_argument("-input_dir", type=str, default="samples")
parser.add_argument("-output_dir", type=str, default="real")
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)

model = create_model(opt)

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)

test_files = glob(osp.join(args.input_dir, "*"))
for inx, path in tqdm(enumerate(test_files)):
    name = path.split("/")[-1].split(".")[0]

    img = cv2.imread(path)[:, :, [2, 1, 0]]
    h, w, c = img.shape
    img = img.transpose(2, 0, 1)[None] / 255
    img_t = torch.as_tensor(np.ascontiguousarray(img)).float()

    model.feed_data(img_t)
    model.test()

    sr = model.fake_SR.detach().float().cpu()[0]
    sr_im = util.tensor2img(sr)

    save_path = osp.join(args.output_dir, "{}_x{}.png".format(name, opt["scale"]))
    cv2.imwrite(save_path, sr_im)
