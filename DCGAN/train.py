from datasets import load_dataset
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation #run the following command on terminal if you face "version `GLIBCXX_3.4.29' not found" issue with matplotlib: export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
from tqdm import tqdm
from model import Generator, Discriminator
from utils import get_args, CustomDataset

if __name__=="__main__":
    args = get_args()

    if args.reproducible:
        # Set random seed for reproducibility
        manualSeed = args.manualSeed
    else:
        print("Using random seed for non-reproducible random results.")
        manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results

    # Decide which device we want to run on
    device = torch.device("cuda:7" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    if args.data_source == "huggingface":
        # Define the transformations
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ds = load_dataset(args.dataset_name)
        train_dataset = ds['train']
        # Create an instance of the custom dataset
        dataset = CustomDataset(train_dataset, transform=transform)
        # Now you can use 'dataset' in your DataLoader as before
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers)
    elif args.data_source == "imagedir":
        # Create the dataset from a directory.
        dataset = dset.ImageFolder(root=args.image_dir,
                                transform=transforms.Compose([
                                    transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers)
    else:
        raise ValueError("Invalid data_source. Choose 'huggingface' or 'imagedir'.")
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(f"{args.fig_dir}/Training_images.png")
    plt.close()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Create the generator
    netG = Generator(args.ngpu, args.nz, args.ngf, args.nc).to(device)
    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(args.ngpu, args.nc, args.ndf).to(device)
    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))
    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Commented out IPython magic to ensure Python compatibility.
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in tqdm(range(args.num_epochs)):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            # print(output.shape, label.shape)
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # # Output training stats
            # if i % 50 == 0:
            #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader),
            #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        if epoch > 0 and epoch%10 == 0:
            # Save models at the end of each epoch
            torch.save(netG.state_dict(), f"{args.checkpoint_dir}/generator_epoch{epoch}.pth")
            torch.save(netD.state_dict(), f"{args.checkpoint_dir}/discriminator_epoch{epoch}.pth")
    # Save final models after training
    torch.save(netG.state_dict(), f"{args.checkpoint_dir}/generator_final.pth")
    torch.save(netD.state_dict(), f"{args.checkpoint_dir}/discriminator_final.pth")

    # **Loss versus training iteration**
    # Below is a plot of D & G’s losses versus training iterations.
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{args.fig_dir}/Generator_Discriminator_Loss.png")  # Save the plot
    plt.close()  # Close to free memory

    # **Visualization of G’s progression**
    # Remember how we saved the generator’s output on the fixed_noise batch
    # after every epoch of training. Now, we can visualize the training
    # progression of G with an animation.
    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    # Save the animation as a GIF or MP4
    ani.save(f"{args.fig_dir}/generator_progression.gif", writer="imagemagick", fps=1)  # For GIF format
    # ani.save("generator_progression.mp4", writer="ffmpeg", fps=1)  # For MP4 format
    plt.close(fig)

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    # Save the plot
    plt.savefig(f"{args.fig_dir}/Real_vs_Generated_Images.png")
    plt.close()