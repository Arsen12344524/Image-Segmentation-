# Semantic Segmentation with U-Net (Pascal VOC 2012)

This project implements a semantic segmentation pipeline using U-Net with a ResNet-34 encoder on the Pascal VOC 2012 dataset.

The notebook covers the full workflow: data loading, preprocessing, training, validation, metric calculation, visualization of predictions, and a small fine-tuning experiment.

---

## Overview

The model performs multi-class pixel-wise classification for 21 classes (20 objects + background).

Key points:
- U-Net architecture
- ResNet-34 encoder (ImageNet pretrained)
- CrossEntropy loss with `ignore_index=255`
- Pixel Accuracy and mIoU for evaluation

---

## Dataset

Pascal VOC 2012

Mask specifics:
- Each pixel stores a class index
- Value `255` is ignored during training and metric computation

Images and masks are resized to **256×256**.

---

## Tech Stack

- PyTorch
- segmentation-models-pytorch
- TorchMetrics
- torchvision
- NumPy
- Matplotlib

---

## Model

U-Net with:
- pretrained ResNet-34 encoder
- skip connections between encoder and decoder

Transfer learning significantly speeds up convergence and improves segmentation quality.

---

## Training

Optimizer:
Adam (lr = 1e-4)

Batch size:
4

Loss:
CrossEntropyLoss(ignore_index=255)

During training the following are tracked:
- training loss
- validation loss
- pixel accuracy
- mean IoU

Model weights are saved after training:
unet_voc.pth


---

## Evaluation Metrics

### Pixel Accuracy
Measures overall percentage of correctly classified pixels.

### Mean IoU
Main segmentation metric.  
Computed while ignoring undefined pixels (255).

---

## Visualization

The notebook shows:
- input image
- ground truth mask
- predicted mask

This helps to qualitatively evaluate model behavior.

---

## Fine-tuning Experiment

After the main training, the model is additionally fine-tuned on a smaller subset:
- 1000 training images
- 200 validation images
- 5 extra epochs

This demonstrates how the pretrained model adapts to limited data.

---


### Install dependencies

```bash
pip install segmentation-models-pytorch==0.3.3 torchmetrics==1.4.0
```

## Limitations

- Trained for a small number of epochs
- No heavy data augmentation
- No learning rate scheduling

## Conclusion

This project demonstrates a complete semantic segmentation training pipeline using U-Net with a pretrained encoder.

Because of limited compute resources, the model was trained only for a small number of epochs, so the results do not reflect its full potential. With longer training, a learning rate scheduler, and stronger augmentations, the performance is expected to improve significantly.

Despite that, the model learns meaningful object boundaries and produces reasonable predictions, which confirms that the training setup, loss handling, and metric implementation are correct.
