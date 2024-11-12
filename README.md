# GANs
Repository containing implementation of GANs for generating images trained on a subset of million-AID dataset.

## Model training
1. Run `pip install -r requirements.txt`
2. `DCGAN_config.yaml` have default configuration for training the DCGAN on `Million-AID` dataset. Can be further modified according to the requirements.
3. `cd DCGAN`
4. Run `python train.py` to train DCGAN

## For Generating Image
1. `cd DCGAN`
2. Run `python generate_images.py`
