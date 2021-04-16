import argparse
import pickle
import time

import cv2
import torch
from sklearn import preprocessing
from torch.autograd import Variable
from torchsummary import summary
from torchvision.utils import save_image

from ecg_wgan_denoiser import Generator, Generator16, Discriminator16
from ecg_wgan_denoiser import Discriminator
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.interpolate import splrep, splev


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=250, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")

ecg_type= 'normalcor2'
# ecg_type= 'testtest'
noise_type =0
OUT_PATH = 'Generator'+ ecg_type + str(noise_type)
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
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
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
    # print(gradients.norm(2, dim=1))
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
def seg_signal(data,rpeaks):
    #return 250, 1d signal with rpeaks
    # print(data.shape)
    l = len(rpeaks)
    seged = np.empty(shape=(l,256),dtype=float)
    segedList =[]
    for i in range(l):
        if rpeaks[i] -56 <0 :
            l -=1
            continue
        if rpeaks[i] +200 > data.shape[0]:
            l -=1
            break
        segedList.append( data[rpeaks[i]-56:rpeaks[i]+200].cpu().detach().numpy())
    # print(segedList)
    ss = np.array(segedList,dtype=float)
    return ss,ss.shape[0]

# seg signal and interporlation
def seg_signal2(data,rpeaks):
    #return 250, 1d signal with rpeaks
    # print(data.shape)
    l = len(rpeaks) - 1
    segedList =[]

    for i in range(l):
        if rpeaks[i] - 56 < 0:
            #l -= 1
            continue
        seg1 = data[rpeaks[i]:rpeaks[i+1]].cpu().detach().numpy()
        r_range = rpeaks[i + 1] - rpeaks[i] -1
        if r_range< 50:
            continue
        x0 = np.linspace(0,r_range,r_range+1)
        # print(len(x0),seg1.shape)
        spl = splrep(x0, seg1)
        x1 = np.linspace(0, r_range, 256)
        y1 = splev(x1, spl)
        # print ('[seg_signal2] y1.shape:', len(y1))
        segedList.append(y1)

    ss = np.array(segedList,dtype=float)
    return ss,ss.shape[0]


def fill_channel16(data,l_original):
    # print('filling',data.shape,l_original)
    if l_original >= 16:
        return data[0:16]
    else:
        newdata = data
        for i in range (16-l_original):
            newdata = np.vstack((newdata,data[i%l_original]))
        return newdata


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
        corel = corel / n
        return corel
    elif dim == 3:
        pass
    else:
        print('[correlate_max] : dimiension is not 2d or 3d')

def dot(A,B):
    A_nom = normalized(A)
    B_nom = normalized(B)
    dim = len(A.shape)

    if dim == 2:
        n = A.shape[0]
        dot = 0
        for i in range(n):
            dot += np.dot(A_nom[i], B_nom[i])
        dot = dot / n
        return dot
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
X = preprocessing.minmax_scale(X, axis=1) * 2 - 1

# X_l = cv2.resize(X_l, None, fx=ratio_konkuk, fy=1, interpolation=cv2.INTER_AREA)
# counter = 0
# X = preprocessing.minmax_scale(X_l,axis=1)* 2 - 1

print(np.argwhere(np.isnan(X)))
assert not np.any(np.isnan(X))
# X = np.array(X,dtype=np.float)
# print(X)
generator = Generator16()
generator.to('cuda:0')
summary(generator, tuple([250]))

discriminator = Discriminator16()
discriminator.to('cuda:0')

summary(discriminator,(16,256))
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

        from biosppy import storage
        from biosppy.signals import ecg

        arr3D_segmented = []
        for i in range(opt.batch_size):

            signal = arr3D[i][0]
            # process it and plot
            rpeaks = ecg.ecg(signal=signal, sampling_rate=100., show=False)[2]

            segmentedsignal,num_seged = seg_signal2(signal,rpeaks)
            # print(rpeaks,num_seged)
            if num_seged == 0:
                print(num_seged)
                break
            segmentedsignal16 = fill_channel16(segmentedsignal,num_seged)
            # segmentedsignal16.reshape(16,256)
            arr3D_segmented.append(segmentedsignal16)
        # print(i,opt.batch_size)
        if num_seged == 0:
            continue
        arr3D_segmented = np.array(arr3D_segmented)
        if len(arr3D_segmented.shape) < 2:
            continue
        # print(arr3D_segmented.shape[1])
        # if np.isnan(np.sum(arr3D_segmented)):
        #     print('nan!!!' )
        #     continue
        arr3D = Tensor(arr3D_segmented)
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
        fake_validity = discriminator(fake_imgs)
        real_validity = discriminator(real_imgs)

        if counter % 1000 == 0:
            # print('img show start')
            imgshow = np.array(real_imgs.cpu(), dtype=float)[0][0]
            plt.figure(figsize=(20, 4))
            plt.plot(imgshow)
            # plt.imshow(imgshow)
            plt.title('real image')
            plt.show()
            plt.close()
            imgshow = np.array(real_imgs.cpu(), dtype=float)[0].flatten()
            plt.figure(figsize=(20, 4))
            plt.plot(imgshow)
            # plt.imshow(imgshow)
            plt.title('real image')
            plt.show()
            plt.close()

            # print(fake_imgs.shape)
            npshow = fake_imgs[0][0].cpu().detach().numpy()
            plt.figure(figsize=(20, 4))
            plt.plot(npshow)
            plt.title('fake image')
            # plt.imshow(np.array(fake_imgs.cpu(), dtype=float))
            plt.show()
            plt.close()

            npshow_list = fake_imgs[0].cpu().detach().numpy()
            npshow= npshow_list.flatten()#reshape((1,-1))
            plt.figure(figsize=(20, 4))
            plt.plot(npshow)
            plt.title('fake image')
            # plt.imshow(np.array(fake_imgs.cpu(), dtype=float))
            plt.show()
            plt.close()

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
            real_extracted = discriminator.getinermediate()
            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            subimg = (fake_imgs) - real_imgs

            l1loss = subimg.norm(1, dim=1).mean()
            l2loss = subimg.norm(1, dim=2).mean()
            # print(l1loss)
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            lsloss = adversarial_loss(fake_validity, valid)

            fake_extracted = discriminator.getinermediate()

            # print('extracted :', real_extracted)
            # print('extracted :', fake_extracted)

            #g_loss = -torch.mean(fake_validity) + l1loss
            #g_loss = -torch.mean(fake_validity) + l2loss

            summer = 0
            for i in range(opt.batch_size):
                corel = correlate(fake_imgs[i].cpu().detach().numpy(), real_imgs[i].cpu().detach().numpy())
                summer += corel
            summer = summer / opt.batch_size


            summer_dot = 0
            for i in range(opt.batch_size):
                dotter = dot(fake_imgs[i].cpu().detach().numpy(), real_imgs[i].cpu().detach().numpy())
                summer_dot += dotter
            summer_dot = summer_dot / opt.batch_size

            regulizer = fake_imgs.norm(1, dim=2).mean()
            # print(regulizer)
            # print(real_extracted.shape)
            sub_ext=real_extracted.cpu().detach().numpy()-fake_extracted.cpu().detach().numpy()
            # print(sub_ext.norm(2, dim=1))
            ext_loss = np.linalg.norm(sub_ext,2,1).mean()
            # print(ext_loss)

            # g_loss =  Variable(Tensor(1).fill_(ext_loss), requires_grad=True) +  0.01*l2loss
            g_loss = -torch.mean(fake_validity) + l2loss * 0.01 + (1-summer) * 80
            # print(g_loss)
            print('correlsummer : ', summer)
            with open(OUT_PATH + '_correler.txt', 'a') as the_file:
                the_file.write(str(summer) + '\n')

            print('dotsummer : ', summer_dot)
            with open(OUT_PATH+ '_dotter.txt', 'a') as the_file:
                the_file.write(str(summer_dot) + '\n')

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, 1, d_loss.item(), g_loss.item())
            )

            with open(OUT_PATH + '_dloss.txt', 'a') as the_file:
                the_file.write(str( float(d_loss.cpu().detach()))  + '\n')

            with open(OUT_PATH + '_gloss.txt', 'a') as the_file:
                the_file.write(str( float(g_loss.cpu().detach())) + '\n')

            # if batches_done % opt.sample_interval == 0:
                # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic
#for lst img end
torch.save(generator.state_dict(), OUT_PATH )