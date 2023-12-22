import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from dataset import MapDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, save_some_examples
from generator_model import Generator
from discriminator_model import Discriminator


def train_fn(gen, disc, train_loader, opt_disc, opt_gen, BCELoss, L1_Loss, g_scaler, d_scaler):

    for idx, (x, y) in enumerate(tqdm(train_loader, leave=True)):

        x, y = x.to(config.device), y.to(config.device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            disc_fake = disc(x, y_fake)
            disc_real = disc(x, y)

            lossD_fake = BCELoss(disc_fake, torch.zeros_like(disc_fake))
            lossD_real = BCELoss(disc_real, torch.ones_like(disc_real))

            D_loss = 0.5 * (lossD_real + lossD_fake)

        disc.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            disc_fake = disc(x, y_fake)
            loss_gen = BCELoss(disc_fake, torch.ones_like(disc_fake))
            l1 = L1_Loss(y_fake, y) * config.L1_lambda
            lossG = loss_gen + l1

        gen.zero_grad()
        g_scaler.scale(lossG).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()



def main():
    disc = Discriminator(in_channels=3).to(config.device)
    gen = Generator(in_channels=3).to(config.device)
    opt_disc = optim.Adam(disc.parameters(), lr=config.learning_rate, betas=[0.5, 0.999])
    opt_gen = optim.Adam(gen.parameters(), lr=config.learning_rate, betas=[0.5, 0.999])

    BCELoss = nn.BCEWithLogitsLoss()
    L1_Loss = nn.L1Loss()

    if config.load_model:
        load_checkpoint(config.checkpoint_gen, gen, opt_gen, config.learning_rate)
        load_checkpoint(config.checkpoint_disc, disc, opt_disc, config.learning_rate)

    train_dataset = MapDataset(root_dir = 'cityscapes/cityscapes/train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.n_workers)

    val_dataset = MapDataset(root_dir='cityscapes/cityscapes/val')
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=True, num_workers=config.n_workers)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.n_epochs):
        train_fn(gen, disc, train_loader, opt_disc, opt_gen, BCELoss, L1_Loss, g_scaler, d_scaler)

        if config.save_model and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, config.checkpoint_gen)
            save_checkpoint(disc, opt_disc, config.checkpoint_disc)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")

if __name__ == "__main__":
    main()