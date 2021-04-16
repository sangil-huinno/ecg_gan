import argparse
import pickle

import cv2
import torch
from sklearn import preprocessing
from torch.autograd import Variable
from torchsummary import summary

from ecg_wgan_denoiser import Generator
from ecg_wgan_denoiser import Discriminator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=250, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")

noise_type =0
OUT_PATH = 'Generator'+ str(noise_type)
opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False
print(cuda)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    # print(real_samples.shape)
    # print(fake_samples.shape)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # print(interpolates.shape)
    # print(real_samples.shape)
    # print(fake_samples.shape)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    # print(gradients.shape   )
    print(gradients.norm(2, dim=1))
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def correlate(A,B):
    A_nom = normalized(A)
    B_nom = normalized(B)
    dim = len(A.shape)

    if dim == 2 :
        n = A.shape[0]
        corel = 0
        for i in range(n):
            corel += np.max(np.correlate(A_nom[i], B_nom[i], 'full'))
        return corel
    elif dim == 3:
        pass
    else:
        print('[correlate_max] : dimiension is not 2d or 3d')
   #normalize first


lead_1_pkl = '/data/data/gunguk/konkuk_v2_7_1_lead1.pkl'
load_pickle = open(lead_1_pkl, 'rb')
load_data = pickle.load(load_pickle)
load_pickle.close()
load_data = load_data[load_data.label1 == '1']
X_real = np.array(load_data.iloc[:, 11:], dtype=np.float32)

df= pd.read_csv('../patch_noise_integrated_20201216_noiselabed.csv')
df = df [df['nlabel'] == noise_type]
X = df.iloc[:,12:2012].to_numpy(dtype=np.float32)

ratio_konkuk = 2000/5000
X = cv2.resize(X_real, None, fx=ratio_konkuk, fy=1, interpolation=cv2.INTER_AREA)
OUT_PATH = 'Generator_normal'
X = preprocessing.minmax_scale(X, axis=1) * 2 - 1


# X_l = cv2.resize(X_l, None, fx=ratio_konkuk, fy=1, interpolation=cv2.INTER_AREA)
# counter = 0
# X = preprocessing.minmax_scale(X_l,axis=1)* 2 - 1

print(np.argwhere(np.isnan(X)))
assert not np.any(np.isnan(X))
# X = np.array(X,dtype=np.float)
# print(X)
generator = Generator()
generator.to('cuda:0')
summary(generator, tuple([250]))

discriminator = Discriminator()
discriminator.to('cuda:0')

summary(discriminator,(1,2000))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
iterator_train = torch.utils.data.DataLoader(X, batch_size=opt.batch_size, shuffle=True, pin_memory=True,
                                             num_workers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adversarial_loss = torch.nn.MSELoss()

if cuda:
    generator.cuda()
    discriminator.cuda()
    # adversarial_loss.cuda()
batches_done = 0
lambda_gp = 1
counter = 0
for epoch in range(opt.n_epochs):
    # for i, (imgs) in enumerate(data):
    for i, (imgs) in enumerate(iterator_train):
        if imgs.shape[0] != opt.batch_size:
            continue
        counter += 1

        arr3D = imgs.reshape(opt.batch_size,1,2000)

        # arr3D = np.repeat (imgs.reshape(1,1,2000), 1,axis=1)
        #
        # scaler = 1.000
        # for i in range(64):
        #     # arr3D[0,i,:] = arr3D[0,i,:] * scaler
        #     arr3D[0,i,:] = torch.from_numpy(np.roll(arr3D[0,i,:],i) * scaler)
        #     scaler -= 0.001
        # print(arr3D.shape)

        img = arr3D.to(device)
        real_imgs = Variable(Tensor(img))
        # real_imgs = real_imgs.reshape(1,64,64)
        # priint(real_imgs)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # print(real_imgs.)
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # Generate a batch of images
        fake_imgs = generator(z)
        # print(real_imgs.shape)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        # print(real_validity.shape)
        # print(fake_imgs.shape)

        fake_validity = discriminator(fake_imgs)
        # print(fake_validity)
        imgshow = np.array(real_imgs.cpu(), dtype=float)[0][0]
        # print(imgshow)
        if counter % 1000 == 0:
            plt.plot(imgshow)
            # plt.imshow(imgshow)
            plt.title('real image')
            plt.figure(figsize=(20, 4))
            plt.show()
            # print(fake_imgs.shape)
            npshow = fake_imgs[0][0].cpu().detach().numpy()
            # npshow = np.array((npshow+1)*128, dtype=int)

            # npshow = npshow.reshape(64,64,1)
            # print(npshow.shape)
            # plt.imshow(npshow)
            plt.figure(figsize=(20, 4))
            plt.plot(npshow)
            plt.title('fake image')
            # plt.imshow(np.array(fake_imgs.cpu(), dtype=float))
            plt.show()

        # print(real_validity, fake_validity)

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
        if counter % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            subimg = (fake_imgs) - real_imgs
            l1loss = subimg.norm(1, dim=1).mean()
            # print(l1loss)
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            lsloss = adversarial_loss(fake_validity, valid)
            lambda_l1  = 1

            summer = 0
            for i in range(opt.batch_size):
                fake_img_cpu = fake_imgs[i][0].cpu().detach().numpy()
                real_img_cpu = real_imgs[i][0].cpu().detach().numpy()
                corel = correlate(fake_imgs[i].cpu().detach().numpy(),real_imgs[i][0].cpu().detach().numpy())
                summer += corel

            summer = summer / opt.batch_size
            print('corelsummer : ',summer)
            with open('correler.txt', 'a') as the_file:
                the_file.write(str(summer))

            g_loss = lsloss +l1loss * lambda_l1
            g_loss = -torch.mean(fake_validity) + l1loss
            # print(g_loss)
            # print(g_loss.shape)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, 1, d_loss.item(), g_loss.item())
            )

            # if batches_done % opt.sample_interval == 0:
                # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic
#for lst img end
torch.save(generator.state_dict(), OUT_PATH )