# Sliced Wasserstein Autoencoder with Class-Fair Representation

## Environment Installation

This repository is designed to operate with `Python 3.10.12`. To set up the required environment, execute the following commands:

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Training Script

To initiate the training process, execute the following command:

```bash
bash train_run.sh
```

## Evaluation Script

Once the training is completed, the evaluation of all methods is performed 10 times, with the average value calculated for each metric. To conduct the evaluation, utilize the following script:

```bash
bash evaluate.sh
```