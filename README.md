# SOD-RepCIN

Implementation of paper - [Article Title]{SOD-RepCIN: A Re-parameterized Multi-scale Compact Feature Extraction Model for Small Object Detection](-)

[![arxiv.org](-)](-)
[![Hugging Face Spaces](-)](-)
[![Hugging Face Spaces](-)](-)


## Performance 

MS VisDrone

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>50-95</sub><sup>val</sup> | Param. | FLOPs |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**SOD-RepCIN-T**]() | 640 | **46.3%** | **33.3%** | **34.5%** | **4.2M** | **21.5G** |
| [**SOD-RepCIN-S**]() | 640 | **52.6%** | **39.5%** | **41.3%** | **16.0M** | **80.1G** |
| [**SOD-RepCIN-M**]() | 640 | **57.9%** | **43.5%** | **46.1%** | **40.30M** | **179.2G** |
| [**SOD-RepCIN-C**]() | 640 | **59.2%** | **45.7%** | **48.1%** | **65.7M** | **355.1G** |


## Installation
conda create -n SOD-RepCIN python=3.11 <br>

conda activate SOD-RepCIN <br>

pip install -r requirements.txt

</details>


## Evaluation

[`SOD-RepCIN-T.pt`](https://github.com/magic524/SOD-RepCIN/releases/download/download/SOD-RepCINt_vd200.pt) [`SOD-RepCIN-S.pt`](https://github.com/magic524/SOD-RepCIN/releases/download/download/SOD-RepCINs_vd200.pt) [`SOD-RepCIN-M.pt`](https://github.com/magic524/SOD-RepCIN/releases/download/download/SOD-RepCINm_vd200.pt.pt) [`SOD-RepCIN-C.pt`](https://github.com/magic524/SOD-RepCIN/releases/download/download/SOD-RepCINc_vd200.pt) 
``` shell
# evaluate converted yolov9 models
python val_dual.py --data data/VisDrone.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights '.SOD-RepCIN-C.pt' --save-json --name SOD-RepCIN-C_640_val

# evaluate yolov9 models
# python val_dual.py --data data/VisDrone.yaml --img 640 --batch 4 --conf 0.001 --iou 0.7 --device 0 --weights '.SOD-RepCIN-C.pt' --save-json --name SOD-RepCIN-C_640_val


## Training

Data preparation
https://docs.ultralytics.com/datasets/detect/visdrone/#what-are-the-main-subsets-of-the-visdrone-dataset-and-their-applications


Single GPU training

``` shell
# train SDO-RepCIN models
python train_dual.py --workers 8 --device 0 --batch 4 --data data/VisDrone.yaml --img 640 --cfg models/detect/SOD-RepCINt.yaml --weights '' --name SOD-RepCINt --hyp hyp.scratch-high.yaml --min-items 0 --epochs 200 --close-mosaic 15

# train gelan models
# python train.py --workers 8 --device 0 --batch 32 --data data/coco.yaml --img 640 --cfg models/detect/gelan-c.yaml --weights '' --name gelan-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15
```

Multiple GPU training

``` shell
# train yolov9 models
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

# train gelan models
# python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/gelan-c.yaml --weights '' --name gelan-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15
```


## Re-parameterization

See [reparameterization.ipynb](https://github.com/WongKinYiu/yolov9/blob/main/tools/reparameterization.ipynb).


## Inference

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="49%"/>
    </a>
</div>

``` shell
# inference converted yolov9 models
python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-c-converted.pt' --name yolov9_c_c_640_detect

# inference yolov9 models
# python detect_dual.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-c.pt' --name yolov9_c_640_detect

# inference gelan models
# python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './gelan-c.pt' --name gelan_c_c_640_detect
```


## Citation

```
@article{wang2024yolov9,
  title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
  booktitle={arXiv preprint arXiv:2402.13616},
  year={2024}
}
```

```
@article{chang2023yolor,
  title={{YOLOR}-Based Multi-Task Learning},
  author={Chang, Hung-Shuo and Wang, Chien-Yao and Wang, Richard Robert and Chou, Gene and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2309.16921},
  year={2023}
}
```


## Teaser

Parts of code of [YOLOR-Based Multi-Task Learning](https://arxiv.org/abs/2309.16921) are released in the repository.

<div align="center">
    <a href="./">
        <img src="./figure/multitask.png" width="99%"/>
    </a>
</div>

#### Object Detection

[`gelan-c-det.pt`](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-det.pt)

`object detection`

``` shell
# coco/labels/{split}/*.txt
# bbox or polygon (1 instance 1 line)
python train.py --workers 8 --device 0 --batch 32 --data data/coco.yaml --img 640 --cfg models/detect/gelan-c.yaml --weights '' --name gelan-c-det --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 10
```

| Model | Test Size | Param. | FLOPs | AP<sup>box</sup> |
| :-- | :-: | :-: | :-: | :-: |
| [**GELAN-C-DET**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-det.pt) | 640 | 25.3M | 102.1G |**52.3%** |
| [**YOLOv9-C-DET**]() | 640 | 25.3M | 102.1G | **53.0%** |

#### Instance Segmentation

[`gelan-c-seg.pt`](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-seg.pt)

`object detection` `instance segmentation`

``` shell
# coco/labels/{split}/*.txt
# polygon (1 instance 1 line)
python segment/train.py --workers 8 --device 0 --batch 32  --data coco.yaml --img 640 --cfg models/segment/gelan-c-seg.yaml --weights '' --name gelan-c-seg --hyp hyp.scratch-high.yaml --no-overlap --epochs 300 --close-mosaic 10
```

| Model | Test Size | Param. | FLOPs | AP<sup>box</sup> | AP<sup>mask</sup>  |
| :-- | :-: | :-: | :-: | :-: | :-: |
| [**GELAN-C-SEG**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-seg.pt) | 640 | 27.4M | 144.6G | **52.3%** | **42.4%** |
| [**YOLOv9-C-SEG**]() | 640 | 27.4M | 145.5G | **53.3%** | **43.5%** |

#### Panoptic Segmentation

[`gelan-c-pan.pt`](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-pan.pt)

`object detection` `instance segmentation` `semantic segmentation` `stuff segmentation` `panoptic segmentation`

``` shell
# coco/labels/{split}/*.txt
# polygon (1 instance 1 line)
# coco/stuff/{split}/*.txt
# polygon (1 semantic 1 line)
python panoptic/train.py --workers 8 --device 0 --batch 32  --data coco.yaml --img 640 --cfg models/panoptic/gelan-c-pan.yaml --weights '' --name gelan-c-pan --hyp hyp.scratch-high.yaml --no-overlap --epochs 300 --close-mosaic 10
```

| Model | Test Size | Param. | FLOPs | AP<sup>box</sup> | AP<sup>mask</sup>  | mIoU<sub>164k/10k</sub><sup>semantic</sup> | mIoU<sup>stuff</sup> | PQ<sup>panoptic</sup> |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [**GELAN-C-PAN**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-pan.pt) | 640 | 27.6M | 146.7G | **52.6%** | **42.5%** | **39.0%/48.3%** | **52.7%** | **39.4%** |
| [**YOLOv9-C-PAN**]() | 640 | 28.8M | 187.0G | **52.7%** | **43.0%** | **39.8%/-** | **52.2%** | **40.5%** |

#### Image Captioning (not yet released)

<!--[`gelan-c-cap.pt`]()-->

`object detection` `instance segmentation` `semantic segmentation` `stuff segmentation` `panoptic segmentation` `image captioning`

``` shell
# coco/labels/{split}/*.txt
# polygon (1 instance 1 line)
# coco/stuff/{split}/*.txt
# polygon (1 semantic 1 line)
# coco/annotations/*.json
# json (1 split 1 file)
python caption/train.py --workers 8 --device 0 --batch 32  --data coco.yaml --img 640 --cfg models/caption/gelan-c-cap.yaml --weights '' --name gelan-c-cap --hyp hyp.scratch-high.yaml --no-overlap --epochs 300 --close-mosaic 10
```

| Model | Test Size | Param. | FLOPs |  AP<sup>box</sup> | AP<sup>mask</sup>  | mIoU<sub>164k/10k</sub><sup>semantic</sup>  | mIoU<sup>stuff</sup> | PQ<sup>panoptic</sup> | BLEU@4<sup>caption</sup> | CIDEr<sup>caption</sup> |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [**GELAN-C-CAP**]() | 640 | 47.5M | - | **51.9%** | **42.6%** | **42.5%/-** | **56.5%** | **41.7%** | **38.8** | **122.3** |
<!--| [**YOLOv9-C-CAP**]() | 640 | 47.5M | - | **52.1%** | **42.6%** | **43.0%/-** | **56.4%** | **42.1%** | **39.1** | **122.0** |-->
<!--| [**YOLOR-MT**]() | 640 | 79.3M | - | **51.0%** | **41.7%** | **-/49.6%** | **55.9%** | **40.5%** | **35.7** | **112.7** |-->


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)

</details>
