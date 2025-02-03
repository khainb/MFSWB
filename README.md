# MFSWB
Official PyTorch implementation for paper:  Towards Marginal Fairness Sliced Wasserstein Barycenter


Details of the model architecture and experimental results can be found in our papers.

```
@inproceedings{nguyen2025towards,
	title={Towards Marginal Fairness Sliced Wasserstein Barycenter},
	author={Anonymous},
	booktitle={The Thirteenth International Conference on Learning Representations},
	year={2025},
	url={https://openreview.net/forum?id=NQqJPPCesd}
}
```
Please CITE our paper whenever this repository is used to help produce published results or incorporated into other software.

This implementation is made by [Khai Nguyen](https://khainb.github.io) (Gaussian Averaging, Color Harmonization, Point Cloud Averaging) and [Hai Nguyen](https://scholar.google.com/citations?user=zIXsuREAAAAJ&hl=vi) (Sliced Wasserstein Autoencoder).

## Requirements
To install the required python packages, run
```
pip install -r requirements.txt
```

## What is included?
* Gaussian Averaging
* Point Cloud Averaging
* Color Harmonization
* Sliced Wasserstein Autoencoder

## Gaussian Averaging

```
cd GaussianAveraging;
python gaussian_example.py;
```

## Point Cloud Averaging

```
cd PointCloudAveraging;
mkdir saved;
python main_point.py
```

## Color Harmonization

```
cd ColorHarmonization;
python main.py;
```

## Sliced Wasserstein Autoencoder

See README.md in SWAE folder