from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import random

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

import sys
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import pycocotools.mask as mask_utils

if len(sys.argv) < 5:
    print('Please pass the folder image_folder Annotation_file cpu/gpu batch_size')
    exit(0)

image_path = sys.argv[1]
ann_path = sys.argv[2]
if(sys.argv[3] == "cpu"):
    _rali_cpu = True
else:
    _rali_cpu = False
bs = int(sys.argv[4])
nt = 1
di = 0
random_seed = random.SystemRandom().randint(0, 2**32 - 1)

pipe = Pipeline(batch_size=bs, num_threads=nt,device_id=di, seed=random_seed, rocal_cpu=_rali_cpu)

with pipe:
    jpegs, bboxes, labels = fn.readers.coco(
        file_root=image_path, annotations_file=ann_path, random_shuffle=True, seed=di, masks=True)
    images_decoded = fn.decoders.image(jpegs, output_type=types.RGB, file_root=image_path, annotations_file=ann_path, shard_id=di, num_shards=nt, random_shuffle=True, seed=di)
    coin_flip = fn.random.coin_flip(probability=0.5)
    rmn_images = fn.resize_mirror_normalize(images_decoded,
                                        device="gpu",
                                        output_dtype=types.FLOAT16,
                                        output_layout=types.NCHW,
                                        resize_min = 1344,
                                        resize_max = 1344,
                                        mirror=coin_flip,
                                        mean= [102.9801, 115.9465, 122.7717],
                                        std = [1. , 1., 1.])
    pipe.set_outputs(rmn_images)

pipe.build()
#output = pipe.run()