# GANs
Repository containing implementation of DCGAN for generating images trained on a subset of million-AID dataset.

## DCGAN architecture
<img src="https://github.com/user-attachments/assets/aa0642f2-33a7-47d2-bb95-c7119852b677" alt="Alt Text" width="500" height="300">

## Model training
1. Run `pip install -r requirements.txt`
2. `DCGAN_config.yaml` have default configuration for training the DCGAN on `Million-AID` dataset. Can be further modified according to the requirements.
3. `cd DCGAN`
4. Run `python train.py` to train DCGAN

## For Generating Image
1. `cd DCGAN`
2. Run `python generate_images.py`
3. This generates a single image by randomly sampling a latent vector and also a collage of 64 images again by sampling a batch of random latent vectors. The images are stored in Figures directory.

## Model generated images
Trained DCGAN generate 64x64 pixel image.

<img src="https://github.com/user-attachments/assets/0a3c65cf-fb4c-4807-8fc0-8ad9f161c9db" alt="Alt Text">

## Reference
Radford, Alec. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint [arXiv:1511.06434](https://arxiv.org/abs/1511.06434) (2015).
