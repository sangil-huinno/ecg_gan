import torchvision
from sklearn import preprocessing
from torch import nn, autograd
import torch

import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchsummary import summary
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=250, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
# cuda = False
#
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.init_size = opt.img_size // 4
        # print( 128 * self.init_size ** 2)
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim,64 * 500))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 128, 9, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 32, 9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 8, 9, stride=1, padding=4),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(8, opt.channels, 9, stride=1, padding=4),
            nn.BatchNorm1d(1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, 125)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv1d(in_filters, out_filters, 9, 2, 4), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters))
            return block

        # self.model = nn.Sequential(
        #     *discriminator_block(opt.channels, 64, bn=False),
        #     *discriminator_block(64, 128),
        #     *discriminator_block(128, 256),
        #     *discriminator_block(256, 512),
        # )

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 8, bn=False),
            *discriminator_block(8, 32, bn=False),
            *discriminator_block(32, 128, bn=False),
            *discriminator_block(128, 256, bn=False),
        )


        # The height and width of downsampled image
        # ds_size = opt.img_size // 2 ** 4
        ds_size = 2000 // 2** 4
        self.adv_layer = nn.Sequential(nn.Linear(125*256 , 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class Generator16(nn.Module):
    def __init__(self):
        super(Generator16, self).__init__()

        # self.init_size = opt.img_size // 4
        # print( 128 * self.init_size ** 2)
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim,16 * 256))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 128, 9, stride=1, padding=4),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 9, stride=1, padding=4),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, 9, stride=1, padding=4),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 16, 9, stride=1, padding=4),
            # nn.BatchNorm1d(16),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, 16)
        img = self.conv_blocks(out)
        return img


class Discriminator16(nn.Module):
    def __init__(self):
        super(Discriminator16, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv1d(in_filters, out_filters, 5, 2, 2), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters))
            return block

        # self.model = nn.Sequential(
        #     *discriminator_block(opt.channels, 64, bn=False),
        #     *discriminator_block(64, 128),
        #     *discriminator_block(128, 256),
        #     *discriminator_block(256, 512),
        # )

        self.model = nn.Sequential(
            *discriminator_block(16, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )


        # The height and width of downsampled image
        # ds_size = opt.img_size // 2 ** 4
        ds_size = 256 // 2** 4
        self.adv_layer = nn.Sequential(nn.Linear(16*256 , 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        self.lastout = out
        return validity

    def getinermediate(self):
        return self.lastout

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # print(real_samples.shape)
    # print(fake_samples.shape)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
# Loss function

adversarial_loss = torch.nn.BCELoss()

if __name__ == "__main__":
    # Initialize generator and discriminator
    generator = Generator()
    # summary(generator, tuple([256]))
    discriminator = Discriminator()
    # summary(discriminator, (1,28,28))
    # exit(0)
    cuda = False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    os.makedirs("data/mnist", exist_ok=True)

    import gzip

    num_images = 1000
    image_size = 28

    data = np.empty(shape=(num_images,1, image_size, image_size), dtype=np.float32)

    f = gzip.open('datasets/train-images-idx3-ubyte.gz', 'r')
    g = gzip.open('datasets/train-labels-idx1-ubyte.gz', 'r')
    g.read(8)
    f.read(16)

    for i in range(num_images):

        buf_target = g.read(1)
        buf = f.read(image_size * image_size)
        buf_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # print(buf_data.shape)
        buf_data = buf_data.reshape( image_size, image_size)
        # plt.imshow(buf_data)
        # plt.show()
        # if int.from_bytes(buf_target,'big') != 0:
        #     i-=1
        #     continue
        data[i] = buf_data.reshape( 1,image_size, image_size) / 256
        # data[i] = preprocessing.minmax_scale(data[i] /256)
        # print(data.shape)
        # print(buf_target)
    print(data.shape)
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    cuda = False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    iterator_train = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, pin_memory=True,
                                                 num_workers=1)
    # ----------
    #  Training
    # ----------
    lambda_gp = 10


    batches_done = 0

    for epoch in range(opt.n_epochs):
        # for i, (imgs) in enumerate(data):
        for i, (imgs) in enumerate(iterator_train):
            # imgs = torch.reshape(imgs[0],(1,28,28))
            imgs.cuda
            # Configure input
            # print(imgs.shape)
            real_imgs = Variable(Tensor(imgs))
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            # Generate a batch of images
            fake_imgs = generator(z)
            # print(real_imgs.shape)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)

            print(real_validity,fake_validity)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            # d_loss = (-real_validity + -fake_validity) / 2
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            # print(d_loss.shape)

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)
                print(g_loss.shape)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, 1, d_loss.item(), g_loss.item())
                )

                # if batches_done % opt.sample_interval == 0:
                #     save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += opt.n_critic