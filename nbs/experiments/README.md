# Experiments Overview :

## Single Fold Experiment

|  Encoder | Optimizer | Scheduler | Loss_func |Mix-Method | Wandb     | OOF Accuracy| Notebook     |
|:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:-----------:|:------------:|
|resnext50_32x4d| Adam, weight_decay=0, eps=1e-08, betas=(0.9, 0.999)) | MultiStepLR | LabelSmoothingCrossEntropy| SnapMix |[link](https://wandb.ai/ayushman/kaggle-leaf-disease-v2/runs/3t4oxo19)| ... |resnext50-32x4d-snapmix.ipynb|
|	|	|	|	|	|	|	|	|
|	|	|	|	|	|	|	|	|
|	|	|	|	|	|	|	|	|



## 5 Fold Experminet
|  Encoder | Optimizer | Scheduler | Loss_func | CV Accuracy | LB Accuracy| TTA  | LB  |
|:--------:|:---------:|:---------:|:---------:|:-----------:|:----------:|:----:|:---:|
|	       |	       |	       |	       |	         |	          |	     |     |