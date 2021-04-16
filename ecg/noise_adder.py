import argparse
import pickle
import time

import cv2
import torch
from sklearn import preprocessing
from torch.autograd import Variable
from torchsummary import summary
from torchvision.utils import save_image

import noise_extractor
from ecg_wgan_denoiser import Generator
from ecg_wgan_denoiser import Discriminator
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.signal as sig


generator = Generator()
noise_type = '0'
OUT_PATH = 'Generator'+ str(noise_type)
generator.to('cuda:1')
model.load_state_dict(torch.load(PATH))
model.eval()

N = 1000
list = []
for i in range(N):
     # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    # Generate a batch of images
    fake_imgs = generator(z)
    ar = [0,0,0,0,0,0,0,0,0,0]
    ar.extend(fake_imgs)
    list.append(fake_imgs)

df = pd.DataFrame(list)
df.to_pickle("noise_generated.pkl")
