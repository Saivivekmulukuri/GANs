from model import Generator
# import random
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from utils import get_args
from torchvision.models import inception_v3
from torchvision import transforms
import torch.nn.functional as F

def inception_score(images, batch_size=32, splits=10):
    """
    Compute Inception Score for a batch of images.

    Args:
        images (torch.Tensor): Tensor of generated images (BxCxHxW), range [0, 1].
        batch_size (int): Batch size for feeding images to Inception model.
        splits (int): Number of splits for calculating statistics.

    Returns:
        float: Inception Score.
    """
    # Load Inception model
    model = inception_v3(pretrained=True, transform_input=False).eval().cuda()
    preprocess = transforms.Resize((299, 299))
    
    # Preprocess images
    images = preprocess(images)
    images = images.cuda()
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            logits = model(batch)
            preds.append(F.softmax(logits, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Compute IS
    scores = []
    split_size = preds.shape[0] // splits
    for i in range(splits):
        part = preds[i * split_size: (i + 1) * split_size]
        p_y = np.mean(part, axis=0)
        scores.append(np.exp(np.sum(part * np.log(part / p_y), axis=1)))
    
    return np.mean(scores), np.std(scores)

if __name__=="__main__":
    args = get_args()

    # Decide which device we want to run on
    device = torch.device("cuda:7" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(args.ngpu, args.nz, args.ngf, args.nc).to(device)
    # Load the saved state dictionaries
    netG.load_state_dict(torch.load(f"{args.checkpoint_dir}/{args.genckpt}"))
    netG.to(device)
    # Set models to evaluation mode if needed
    netG.eval()

    noise_batch = torch.randn(64, args.nz, 1, 1, device=device)
    fake_images = netG(noise_batch).detach().cpu()
    is_mean, is_std = inception_score(fake_images)
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")