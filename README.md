# Requirements

torch = 1.7.1+cu110
python = 3.8.13

Some basic python packages such as 
ninja
yacs
cython
matplotlib
tqdm
OpenCV
scipy
......

# Usage

Train the model

`python -m torch.distributed.launch \
--nproc_per_node=1 \
tools/train_net.py \
--master_port=$((RANDOM + 10000)) \
--config-file configs/AFGN/vid_R_101_C4_AFGN_1x.yaml \
OUTPUT_DIR training_dir/AFGN`

Get evaluate results images

Firstly, move the evaluate result masks to the eval folder in data folder.

Then run 
`python -m torch.distributed.launch
--nproc_per_node=1
tools/test_net.py 
--config-file configs/AFGN/vid_R_101_C4_AFGN_1x.yaml
MODEL.WEIGHT AFGN_R_101.pth \`

# Acknowledgement
* The AFGN is adapted from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [MEGA](https://github.com/Scalsol/mega.pytorch) for medical video object detection.
