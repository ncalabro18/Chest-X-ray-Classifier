# Chest X-ray Multi-Classifier

### Project Description

This project focuses on building a deep learning model for multi-label classification of chest X-ray images using the NIH dataset.

The task is challenging because each image can contain multiple diseases, and the dataset is imbalanced.

To handle this, we chose different approaches, including preprocessing techniques, transformer-based architectures, and advanced training strategies, in order to improve performance and understand what actually works best. Key Approaches:

Image preprocessing techniques such as CLAHE
Transformer-based architectures (Swin Transformer and SwinV2)
Training strategies such as asymmetric loss, unfreeze schedules, and self-supervised pretraining (SimMIM)
Our goal was to improve AUC performance, focusing on our most advanced model, Model 3, which uses a SimMIM pretrained SwinV2 backbone and to push its performance above 81 percent.


### Local Installation


#### Training

To train, clone the repository.

Recommended Python 3.11 for training

```python --version```

```pip install -r requirements.txt```


Beyond Python there are two other dependencies:

  [NIH Chest X-ray dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) (42 GB)
  
  To run train.py, all *_PATH variables must resolve. This includes the image root, metadata file, and checkpoint file.

  [Microsoft SimMIM Swin small checkpoint](https://huggingface.co/zdaxie/SimMIM/blob/main/simmim_swinv2_pretrain_models/swinv2_small_1k_500k.pth) (.2 GB)


### Run

Use the 3.11 interpreter to run train.py. No arguments are needed.

```python train.py```

This will output a best checkpoint and log file.


### Results
[Model 1](old_architecture/swin.ipynb)  ->  ~83%

[Model 2](old_architecture/swin_clahe.ipynb) -> 84%

The following resulting AUCs discard Hernia representation in the latest model because it is difficult to measure with minimal error.
In addition, future experiments use a patient level split so value data doesn't contain x-rays of patients that have used in the train data.
The figures above are closer to 81% and 82% for comparison purposes because of the calculation error, but the following ones are more accurate fix the error:

Model 3 -> 81.5%

| Model 4 Version | Validation AUC |
|---|---:|
| ImageNet baseline | 74.43% |
| 30 epochs on NIH | 80.80% |
| 100 epochs on NIH | 81.9866% |
| 100 epochs + thresholding | 82.01% |


### Dataset
The NIH Chest X-ray Dataset contains 112,120 images of various resolutions
It includes 14 disease catagories with a large imbalance:
```
No Finding           | ██████████████████████████████████████████████ 60361
Infiltration         | ████████████████ 19894
Effusion             | ████████████ 13317
Atelectasis          | ███████████ 11559
Nodule               | ██████ 6331
Mass                 | █████ 5782
Pneumothorax         | █████ 5302
Consolidation        | ████ 4667
Pleural Thickening   | ███ 3385
Cardiomegaly         | ██ 2776
Emphysema            | ██ 2516
Edema                | ██ 2303
Fibrosis             | █ 1686
Pneumonia            | █ 1431
Hernia               | ▏ 227
```
Mean: ```5798.3```

Standard Deviation: ```5429.6```


## Models

### Authors:
Nicholas Calabro
