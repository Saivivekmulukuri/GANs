from model import Generator
# import random
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from utils import get_args

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

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    noise_batch = torch.randn(64, args.nz, 1, 1, device=device)
    fake = netG(noise_batch).detach().cpu()
    gen_imgs = vutils.make_grid(fake, padding=2, normalize=True)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(gen_imgs,(1,2,0)))
    plt.savefig(f"{args.fig_dir}/Generated_images_batch.png")
    plt.close()

    # Generate a single random noise vector
    noise_single = torch.randn(1, args.nz, 1, 1, device=device)  # Adjust nz to your latent vector size
    fake_single = netG(noise_single).detach().cpu()
    gen_img_single = vutils.make_grid(fake_single, padding=2, normalize=True)

    # Plot and save the single generated image
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Image")
    plt.imshow(np.transpose(gen_img_single, (1, 2, 0)))
    plt.savefig(f"{args.fig_dir}/Generated_image_single.png")
    plt.close()