import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
class Digits(Dataset):
    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)
        self.transforms = transforms
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class ScoreMatching(nn.Module):
    def __init__(self, snet, alpha, sigma, eta, D, T):
        super(ScoreMatching, self).__init__()
        print('Score Matching by JT.')
        self.snet = snet
        self.D = D
        self.sigma = sigma
        self.T = T
        self.alpha = alpha
        self.eta = eta

    def sample_base(self, x_1):
        return 2. * torch.rand_like(x_1) - 1.

    def langevine_dynamics(self, x):
        for t in range(self.T):
            x = x + self.alpha * self.snet(x) + self.eta * torch.randn_like(x)
        return x

    def forward(self, x, reduction='mean'):
        epsilon = torch.randn_like(x)
        tilde_x = x + self.sigma * epsilon
        s = self.snet(tilde_x)
        SM_loss = (1. / (2. * self.sigma)) * ((s + epsilon)**2.).sum(-1)
        if reduction == 'sum':
            loss = SM_loss.sum()
        else:
            loss = SM_loss.mean()
        return loss

    def sample(self,  batch_size=64):
        x = self.sample_base(torch.empty(batch_size, self.D))
        x = self.langevine_dynamics(x)
        x = torch.tanh(x)
        return x

def evaluation(test_loader, name=None, model_best=None, epoch=None):
    if model_best is None:
        model_best = torch.load(name + '.model')
    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        loss_t = model_best.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss


def samples_real(name, test_loader):
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()
    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')
    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()

def samples_generated(name, data_loader, extra_name='', T=None):
    model_best = torch.load(name + '.model')
    model_best.eval()
    if T is not None:
        model_best.T = T
    num_x = 4
    num_y = 4
    x = model_best.sample(batch_size=num_x * num_y)
    x = x.detach().numpy()
    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')
    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()

def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('score matching loss')
    plt.savefig(name + '_sm_val_curve.pdf', bbox_inches='tight')
    plt.close()

def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0
    for e in range(num_epochs):
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)
        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_val
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val
transforms = tt.Lambda(lambda x: 2. * (x / 17.) - 1.)
train_data = Digits(mode='train', transforms=transforms)
val_data = Digits(mode='val', transforms=transforms)
test_data = Digits(mode='test', transforms=transforms)
training_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
D = 64
M = 512
alpha = 0.1
sigma = 0.1
eta = 0.05
T = 100
lr = 1e-4
num_epochs = 1000
max_patience = 20
name = 'sm' + '_' + str(T)
result_dir = name + '/'
if not (os.path.exists(result_dir)):
    os.mkdir(result_dir)
snet = nn.Sequential(nn.Linear(D, M), nn.SELU(),
                     nn.Linear(M, M), nn.SELU(),
                     nn.Linear(M, M), nn.SELU(),
                     nn.Linear(M, D), nn.Hardtanh(min_val=-4., max_val=4.))
model = ScoreMatching(snet=snet, alpha=alpha, sigma=sigma, eta=eta, T=T, D=D)
optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model, optimizer=optimizer,
                       training_loader=training_loader, val_loader=val_loader)
test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
f = open(result_dir + name + '_test_loss.txt', "w")
f.write(str(test_loss))
f.close()
samples_real(result_dir + name, test_loader)
samples_generated(result_dir + name, test_loader, extra_name='FINAL')
plot_curve(result_dir + name, nll_val)
