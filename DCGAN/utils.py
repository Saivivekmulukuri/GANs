from torch.utils.data import Dataset
import argparse
import yaml

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label_3'] # or label_2 or label_3 as needed

        if self.transform:
            image = self.transform(image)

        return image, label


def load_config(config_path="DCGAN_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_args():
    config = load_config()

    parser = argparse.ArgumentParser(description="Model Training Parameters")

    # Set default values from the YAML config
    parser.add_argument("--checkpoint_dir", type=str, default=config.get("checkpoint_dir"),
                        help="Directory to save model checkpoints")
    parser.add_argument("--fig_dir", type=str, default=config.get("fig_dir"),
                        help="Directory to save generated figures")
    parser.add_argument("--manualSeed", type=int, default=config.get("manualSeed"),
                        help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size"),
                        help="Batch size during training")
    parser.add_argument("--image_size", type=int, default=config.get("image_size"),
                        help="Spatial size of training images")
    parser.add_argument("--nc", type=int, default=config.get("nc"),
                        help="Number of channels in training images")
    parser.add_argument("--nz", type=int, default=config.get("nz"),
                        help="Size of z latent vector (input to generator)")
    parser.add_argument("--ngf", type=int, default=config.get("ngf"),
                        help="Size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=config.get("ndf"),
                        help="Size of feature maps in discriminator")
    parser.add_argument("--ngpu", type=int, default=config.get("ngpu"),
                        help="Number of GPUs available. Use 0 for CPU mode")
    parser.add_argument("--dataset_name", type=str, default=config.get("dataset_name"),
                        help="Hugging Face dataset name")
    parser.add_argument("--data_source", type=str, default=config.get("data_source"),
                        choices=['huggingface', 'imagedir'], 
                        help="Data source: 'huggingface' or 'imagedir'")
    parser.add_argument("--image_dir", type=str, default=config.get("image_dir"),
                        help="Path to the image directory (used if data_source is 'imagedir')")
    parser.add_argument("--reproducible", type=bool, default=config.get("reproducible"), 
                        help="Set to True for reproducible results (set fixed random seed)")
    parser.add_argument("--workers", type=int, default=config.get("workers", 2),
                        help="Number of workers for dataloader")
    parser.add_argument("--num_epochs", type=int, default=config.get("num_epochs", 300),
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.get("lr", 0.0002),
                        help="Learning rate for optimizers")
    parser.add_argument("--beta1", type=float, default=config.get("beta1", 0.5),
                        help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--genckpt", type=str, default = config.get("genckpt"),
                        help="Path to the generator checkpoint file for generating new images")
    
    args = parser.parse_args()

    return args