from itertools import chain

import torch
from torch.autograd import Variable

from ecg_wgan_denoiser import Generator16
import numpy as np
import os
import matplotlib.pyplot as plt


def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

cuda = True if torch.cuda.is_available() else False
generator = Generator16()
noise_type = '3'
PATH = 'Generatornormalcor20'#+ str(noise_type)
imgPATH = PATH +'_img'
generator.to('cuda:0')
generator.load_state_dict(torch.load(PATH))
generator.eval()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
latent_dim = 256
os.makedirs(imgPATH)

N = 20
list = []
for i in range(N):
     # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (1, 250))))
    # Generate a batch of images
    fake_imgs = generator(z)
    # ar = [0,0,0,0,0,6,0,0,0,0,0]
    # ar.extend(flatten(flatten(fake_imgs.tolist())))
    # list.append(ar)
    # print(ar)
    npshow_list = fake_imgs[0].cpu().detach().numpy()
    npshow = npshow_list.flatten()  # reshape((1,-1))
    plt.figure(figsize=(20, 4))
    plt.plot(npshow)
    plt.title('fake image'+ str(i))

    plt.savefig(imgPATH+'/fake'+str(i) + '.png')
    # plt.imshow(np.array(fake_imgs.cpu(), dtype=float))
    plt.show()
    plt.close()

# df = pd.DataFrame(list,columns= ['filename',    'patient_id', 'hospital_name',        'method',
#                 'lead',        'label1',        'label2',        'label3',
#               'label4',        'label5'])
# df = pd.DataFrame(list)
# df.to_pickle('noise_generated'+noise_type+'.pkl')
