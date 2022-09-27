# Overview
This is the Pytorch implementation of "Conv2NeXt: Reconsidering ConvNeXt Network Design for Image Recognition"
# Requirements
To use this project, you need to ensure the following requirements are installed.
- Python >= 3.8
- torch==1.8.0
- torchvision==0.9.0
- timm==0.3.2
# Dataset
The program will download CIFAR10/100 dataset automatically.
Download the Tiny-ImageNet classification dataset and structure the data as follows:
```
/path/to/tiny-imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```
# Train
```
python main.py \
--model conv2next_base --drop_path 0.1 \
--data_set CIFAR10 --input_size 32 \
--batch_size 64 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/cifar10 \
--output_dir /path/to/save_results \
--log_dir /path/to/logs
```
# checkpoints
The checkpoints will be available in GoogleDrive soon.
