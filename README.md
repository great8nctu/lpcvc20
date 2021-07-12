# FoxPandaNet

The 3rd place winner of the 2020 On-device Visual Intelligence Competition ([OVIC](https://lpcv.ai/2020CVPR/ovic-track)) of Low-Power Computer Vision Challenge ([LPCVC](https://lpcv.ai/)), subtrack of Real-time Image Classification on LG G8.

## Model

Tested with tflite-runtime 2.1.0 on Raspberry Pi 4 (single core)

|Model|Input Image Size|Accuracy (UINT8)|Latency|Download|SHA256 Checksum|
| - | - | - | - | - | - |
|fpnet_pixel4 (uint8)|192x192|71.93%|60.1ms|[Download Link](https://drive.google.com/file/d/1zToDUmViDMmAAziz4ozAna-1uF7ZJ4-y/view?usp=sharing)|6e927d7af8da1eb9297017ebe92a67632ce73f612ff32cbfa3917f88d761a5f9|
|fpnet_dsp (uint8)|224x224|74.18%|94.9ms|[Download Link](https://drive.google.com/file/d/1HebGFcB60mm0VM8P2KwH9FlVerIf0nda/view?usp=sharing)|b77445326ef3f64fc8d3236b213e121aba5004dee4449deceb13f246477add4e|
|fpnet_fpga (uint8)|160x160|70.28%|48ms|[Download Link](https://drive.google.com/file/d/1WNbI244hUU3vSmXlMAiXK0Do_PU3wLva/view?usp=sharing)|7b15050f0f2f723b13cfc001026a36133f78103049c94ae8fe023807e355fc20|


## Compared with MobileNetV3/MobileNetV2 on Raspberry Pi 4 CPU (single core)

![](https://raw.githubusercontent.com/great8nctu/lpcvc20/master/figures/rpi4_cpu_compare_20210201.png)

## Training and Evaluation

```
$ python3 train.py --model_name fpnet_pixel4 \
  --batch_size 1024 \
  --epochs 250 \
  --warmup_epochs 5 \
  --base_lr 0.4 \
  --init_lr 0.1 \
  --image_size 192 \
  --use_cache \
  --imagenet_path $IMAGENET_PATH \
  --checkpoint_path $CHECKPOINT_PATH
$ python3 convert_quant.py --keras_model_file $KERAS_FILE \
  --output_file $TFLITE_FILE \
  --image_size 192 \
  --imagenet_path $IMAGENET_PATH
$ python3 val_quant.py --tflite_model_file $TFLITE_FILE \
  --image_size 192 \
  --imagenet_path $IMAGENET_PATH
```

## Methodology

+ Neural architecture search with multivariate regression
+ Once-for-All supernet
+ MobileNet V3 backbone
+ Replacing Hard-swish with ReLU6 for better quantization performace

**Problem statement**                                                             
Given a targeted latency on the specific hardware, we aim to find an optimal neural network, based on neural architecture search (NAS) techniques, with highest accuracy while meeting the constraint.
 
**Neural architecture search based on multivariate regression**  

![](https://raw.githubusercontent.com/great8nctu/lpcvc20/master/figures/FoxPanda_flowchart.png)
