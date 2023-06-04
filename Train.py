import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import src
import src.Model as m
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr =  2e-4 
batch_size = 128
image_size = 64
img_channels = 1
features_g = 64
features_d = 64
epochs = 5
z_dim = 100

#Transform
tranforms = transforms.Compose([
    transforms.Resize(image_size),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5 for _ in range(img_channels)],
        [0.5 for _ in range(img_channels)]
        )
]     
)


#DATASET 
dataset = datasets.MNIST(root = "dataset/", download = True, transform = tranforms)
#load dataset
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

#define disc and gen objects
disc = m.Discriminator(img_channels,features_d).to(device)
gen = m.Generator(z_dim,img_channels,features_g).to(device)

#weight initialization
m.initialize_weights(disc)
m.initialize_weights(gen)

#optimizers
disc_opti = optim.Adam(disc.parameters(),lr= lr,betas=(0.5,0.999))
gen_opti = optim.Adam(gen.parameters(),lr= lr,betas=(0.5,0.999))

#loss
criterion = nn.BCELoss()

#fixed noise
fixed_noise = torch.randn(32,z_dim,1,1).to(device)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

disc.train()
gen.train()

for epoch in range(epochs):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((batch_size,z_dim,1,1)).to(device)
        fake = gen(noise)  #G(Z)

        #loss of discriminator max log D(x) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)  # D(x)

        loss_disc_real = criterion(disc_real,torch.ones_like(disc_real)) # max log D(x)
        disc_fake = disc(fake).reshape(-1)  # D(G(z))
        loss_disc_fake = criterion(disc_fake,torch.zeros_like(disc_fake)) # max log(1 - D(G(z)))

        loss_disc = (loss_disc_fake + loss_disc_real) /2 
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        disc_opti.step()

        #Generator min log(1 - D(G(z)))  or max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        gen_opti.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "FAKE", img_grid_fake, global_step = step
                )

                writer_real.add_image(
                    "REAL", img_grid_real, global_step = step
                )

                step += 1
