# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import torch
import random
import numpy as np
# seed = 76
# random.seed(seed) # python seed
#     #os.environ['PYTHONHASHSEED'] = str(seed) # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
# np.random.seed(seed) # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
# torch.manual_seed(seed) # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
# torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
#     #torch.cuda.manual_seed_all(seed) # 使用多块GPU时，均设置随机种子
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark =  False # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
#     #torch.backends.cudnn.enabled = True # pytorch使用CUDANN加速，即使用GPU加速

from afgn_core.utils.env import setup_environment  # noqa F401 isort:skip
from afgn_core.utils.dist_env import init_dist
from afgn_core.utils.distributed import ompi_rank, ompi_local_rank

import argparse
import time
import os


import numpy as np
from afgn_core.config import cfg
from afgn_core.data import make_data_loader
from afgn_core.solver import make_lr_scheduler
from afgn_core.solver import make_optimizer
from afgn_core.engine.inference import inference
from afgn_core.engine.trainer import do_train
from afgn_core.modeling.detector import build_detection_model
from afgn_core.utils.checkpoint import DetectronCheckpointer
from afgn_core.utils.collect_env import collect_env_info
from afgn_core.utils.comm import synchronize, get_rank
from afgn_core.utils.imports import import_file
from afgn_core.utils.logger import setup_logger
from afgn_core.utils.miscellaneous import mkdir, save_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, ignore=cfg.MODEL.VID.IGNORE)
   

    if not cfg.MODEL.VID.IGNORE:
        arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
    )

    return model


def run_test(cfg, model, distributed, motion_specific=False):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            motion_specific=motion_specific,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

# pytorch 设置随机种子（我试下来是有用的）(万一没用了，参考https://blog.csdn.net/weixin_40400177/article/details/105625873)
def seed_torch(seed=42):
    random.seed(seed) # python seed
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
    np.random.seed(seed) # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed) # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =  False # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
    torch.backends.cudnn.enabled = True # pytorch使用CUDANN加速，即使用GPU加速

def main():
    # seed_torch(seed=78)
    torch.cuda.set_device(2)
    parser = argparse.ArgumentParser(description="PyTorch Video Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        help="path to config file",
        type=float,
    )
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument("--master_port", "-mp", type=str, default='29999')
    parser.add_argument("--save_name", default="", help="Where to store the log", type=str)
    parser.add_argument(
        "--motion-specific",
        "-ms",
        action="store_true",
        help="if True, evaluate motion-specific iou"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        init_dist(args.launcher, args=args)
        synchronize()

    # this is similar to the behavior of detectron2, which I think is a nice option.
    BASE_CONFIG = "configs/BASE_RCNN_{}gpu.yaml".format(num_gpus)
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    cfg.SOLVER.BASE_LR = args.lr

    if output_dir:
        mkdir(output_dir)
    cfg.freeze()

    logger = setup_logger("afgn_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    if args.launcher == "mpi":
        args.local_rank = ompi_local_rank()
    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed, args.motion_specific)

if __name__ == "__main__":
    main()