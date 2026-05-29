# Chest X-ray Multi-Classifier

### Description

This project focuses on building a model for multi-label classification of chest X-ray images using the NIH dataset.

The task is challenging because each image can contain multiple diseases, and the dataset is imbalanced.

To interface with the model, I design a secure web application which runs a selected checkpoint file.

I use FastAPI to handle a DMZ webserver. SlowAPI ratelimits requests for python services while [nginx](frontend/nginx.conf) rate-limits frontend requests. More details are in the [docker-compose file](docker-compose.yml). Cloudflared handles the direct ingress, which sends API requests to caddy. [Caddy](caddyfile) proxies the requests to the appropriate container address. https terminates at the cloudfared tunnel, caddy only accepts http.

Prometheus and grafana are utilized to display realtime metrics extracted from nginx, the classifier, and the web api service.


The final goal of the model is optimizing for clinical relevance, using sensitivity threshold optimization: the model is more sensitive to anomalies, resulting in higher false positives but fewer false negatives. False negatives result in the patient being sent home with a disease; thus, the decision was made to optimize thresholds to prevent this.


## Architecture

[About model webpage](https://classifier.ncalabro.net/about)

### Data Preprocessing / Augmentation

The training pipeline applies a sequence of image augmentations to improve
model robustness and generalization. Tuning parameters are kept conservative as X-ray anatomical structures can be easily distorted.

#### Geometric Augmentations
- Resize input images to a fixed resolution
- Random horizontal flipping
- Random rotation
- Elastic deformation
- Grid distortion

#### Contrast & Intensity Augmentations
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Brightness and contrast jitter
- Random gamma adjustment

#### Noise & Regularization
- Gaussian noise injection
- Coarse dropout / random region masking


### Backbone

Pretrained on the NIH Chest X-ray dataset, SimMIM is utilized to learn features by masking 60% of the image.

The main modification is that stage outputs 0-2 are tapped before their respective patch-merging layers, giving the head access to higher resolution but lower channel feature maps for multi-scale fusion. Stage 3 has no merging and feeds directly into the backbone's LayerNorm, after which its tokens are passed to the head alongside the tapped early features.

### Head

The head receives token streams from all four backbone stages. The three early stages (0-2) each go through a stage_proj - a bottleneck Conv2d pair that projects each stage's native channel dimension down to C//4 then back up to C, after spatially pooling to a common 7×7 grid. Each projected map is then gated by a learned scalar (1 + tanh(gate_i)), letting the model suppress or amplify each scale's contribution. The gated stage tokens are concatenated with the final-stage tokens (scaled by a softplus temperature), then passed through a shared class_norm LayerNorm.

Cross-attention then operates with num_classes learned query vectors against this full multi-scale token set, producing one feature vector per class. The conditioned class features go through a three-layer MLP head (C -> 256 -> 1) to produce the final per-class logits.


### Thresholding

After probabilities are calculated, final thresholds are applied to aid in diagnostic capabilities. These are selected to minimize false negatives, known as sensitivity optimization.

### Model Output

The model gives final probabilities which can be tuned to optimize for different goals. For the final model, sensitivity optimization is utilized rather than f1 or Jouden's.

Attention maps and saliency maps are generated to display the decision making areas weighted the highest for each positive finding.

## Local Execution

### Classifier Web Server

```git clone https://github.com/ncalabro18/Chest-X-ray-Classifier```

```cd Chest-X-ray-Classifier```

Create passwords for your environment:


```vim .env```

```.env``` Should match variable names to:

```
CLASSIFIER_API_KEY=...
CLOUDFLARE_TUNNEL_TOKEN=...
CF_API_TOKEN=...
GRAFANA_ADMIN_PASSWORD=...
INTERNAL_SECRET=...
```

Add under caddy service
```
    ports:
      "127.0.0.1:80:80"
```

Then,


```docker-compose up```


Is all that is needed. Navigate to http://localhost
Local testing happens over http, not https.



To run vulnerability scanning before a run:

```make cve_scan```

To deploy, edit the address in caddyfile and create a cloudflare tunnel that points to http://caddy:80.

### Training

To train, clone the repository.

Recommended Python 3.11 for training

```python --version```

```pip install -r requirements.txt```


Beyond Python and pip dependencies there are 2 more downloads:

  [NIH Chest X-ray dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) (42 GB)
  
  To run train.py, all *_PATH variables must resolve. This includes the image root, metadata file (both in the link above), and checkpoint file (following link).

  __Note:__ Dataset is for training / validation purposes only. See [Web Server](#classifier-web-server)

  [Microsoft SimMIM Swin small checkpoint](https://huggingface.co/zdaxie/SimMIM/blob/main/simmim_swinv2_pretrain_models/swinv2_small_1k_500k.pth) (.2 GB)



Use the 3.11 interpreter to run train.py. No arguments are needed.
The script run_model.sh is useful to control GPU selection and python interpreter; edit the path to your local setup.


```./run_model.sh train.py```

This will output a best checkpoint, log file, and 2 CSV files


## Results

Average AUC of diseases of the saved model is just under 81%.
With threshold optimization less data is used in training, which is my main suspect as to why it is about a percentage below standards on the dataset given the noise of the labels and classification challenge.

Future efforts will focus on multi-image analysis, increasing generalization with extra datsets, and improving maintainability.


## Dataset
The NIH Chest X-ray Dataset contains 112,120 images of various resolutions
It includes 14 disease categories with a large imbalance:
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



## Authors:

Nicholas Calabro

Instructed by Professor Wenjin Zhou

Thank you for contributions to the documentation by:

Anthony Klimas

Luke MacVicar

Hilary Jaen Rodriguez


Extended Final Project for a Computer Science Special Topics Elective: _Computing for Health and Medicine_

[Professor Wenjin's Course Page](https://www.cs.uml.edu/~wzhou/comp5300.html)


## References

### NIH Chest X-ray Dataset
Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017).  
*ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.*  
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).  
https://doi.org/10.1109/CVPR.2017.369

### Swin Transformer v2
Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., Wei, F., & Guo, B. (2022).  
*Swin Transformer V2: Scaling Up Capacity and Resolution.*  
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).  
https://doi.org/10.1109/CVPR52688.2022.01167