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

#Inference Speed
Backbone: Resnet-101
| No. of GPUs      | mAP | FPS     |
| :---        |    :----:   |          ---: |
| 1      | 82.45%       | 3.51   |
| 4   | 82.45%        | 9.21      |

Backbone: Resnet-50
| No. of GPUs      | mAP | FPS     |
| :---        |    :----:   |          ---: |
| 1      | 81.28%       | 4.46   |
| 4   | 81.28%        | 14.10      |

# Usage

Train the model

`python -m torch.distributed.launch 
--nproc_per_node=1 
tools/train_net.py 
--master_port=$((RANDOM + 10000)) 
--config-file configs/AFGN/vid_Res_101_C4_AFGN_1x.yaml 
OUTPUT_DIR training_dir/AFGN`

Test the model

Then run 
`python -m torch.distributed.launch
--nproc_per_node=1
tools/test_net.py 
--config-file configs/AFGN/vid_Res_101_C4_AFGN_1x.yaml
MODEL.WEIGHT AFGN_R_101.pth `

# Acknowledgement
* The AFGN is based [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [MEGA](https://github.com/Scalsol/mega.pytorch) for medical video object detection.
