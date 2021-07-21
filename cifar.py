'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from local_models import *
#from utils import progress_bar
from torchvision.utils import save_image, make_grid

from autoenc.ae_model import tmpViT, tmpTransGan, facebook_vit

from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume_enc', '-re', action='store_true',
                    help='resume encoder from checkpoint')
parser.add_argument('--resume_dec', '-rd', action='store_true',
                    help='resume decoder from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
print("DEVICE:", device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

allSampleSet = trainset + testset

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

train_inliers = [np.where(np.array(trainset.targets) == class_idx)[0]
              for class_idx in trainset.class_to_idx.values()]
train_outliers = [np.where(np.array(trainset.targets) != class_idx)[0]
              for class_idx in trainset.class_to_idx.values()]
test_inliers = [np.where(np.array(testset.targets) == class_idx)[0]
              for class_idx in testset.class_to_idx.values()]
test_outliers = [np.where(np.array(testset.targets) != class_idx)[0]
              for class_idx in testset.class_to_idx.values()]

for i in range(len(classes)):
    test_inliers[i] += len(trainset)
    test_outliers[i] += len(trainset)

    # Drop elements
    train_outliers[i] = np.random.choice(train_outliers[i], size=500, replace=False)
    test_outliers[i] = np.random.choice(test_outliers[i], size=500, replace=False)

inliers_zip = zip(train_inliers, test_inliers)
inliers = [ np.concatenate((i,j), dtype=np.int64) for i, j in inliers_zip]

outliers_zip = zip(train_outliers, test_outliers)
outliers = [ np.concatenate((i,j), dtype=np.int64) for i, j in outliers_zip]

for i in outliers:
    print("Outlier size: ", len(i))

trainloader = [
    DataLoader(
        dataset=Subset(allSampleSet, inds),
        batch_size=128,
        shuffle=False,
        num_workers=2
    ) for inds in inliers]

testloader = [
    DataLoader(
        dataset=Subset(allSampleSet, inds),
        batch_size=128,
        shuffle=True,
        num_workers=2
    ) for inds in outliers]

unified_loaders = list(zip(trainloader, testloader))
#loaders = list(map(iter, unified_loader))

# Training
def train(epoch, trainloader, loader_idx):
    print('\nEpoch: %d' % epoch)
    # net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        encodings = encoder(inputs)
        recons = decoder(encodings)
        loss = criterion(recons, inputs)
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()

        if (batch_idx % 500 == 0):
            print("writing image")
            cpu_inp = inputs.cpu()
            cpu_recons = recons.cpu()
            save_image(make_grid(cpu_inp, nrows=10),
                       "./cifar_imgs/train_input_Inlier:" + classes[loader_idx] + "_epoch_" + str(epoch) + "_" + str(batch_idx) + ".jpg")
            save_image(make_grid(cpu_recons, nrows=10),
                       "./cifar_imgs/train_recon_Inlier:" + classes[loader_idx] + "_epoch_" + str(epoch) + "_" + str(batch_idx) + ".jpg")
        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        print("Epoch No. ", epoch, "Batch Index.", batch_idx, "Loss: ", (train_loss / (batch_idx + 1)))


def test(epoch, testloader, loader_idx):
    print("TESTING")
    global best_acc
    #net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            encodings = encoder(inputs)
            recons = decoder(encodings)
            loss = criterion(recons, inputs)

            test_loss += loss.item()

            print("TEST: Epoch No. ", epoch, "Batch Index.", batch_idx, "Loss: ", (test_loss/(batch_idx+1)))

            if (batch_idx % 500 == 0):
                cpu_inp = inputs.cpu()
                cpu_recons = recons.cpu()
                save_image(make_grid(cpu_inp, nrows=10), "./cifar_imgs/test_input_Inlier: " + classes[loader_idx] + "_epoch_" + str(epoch) + "_" + str(batch_idx) + ".jpg")
                save_image(make_grid(cpu_recons, nrows=10), "./cifar_imgs/test_recon_Inlier:" + classes[loader_idx] + "_epoch_" + str(epoch) + "_" + str(batch_idx) + ".jpg")
    '''
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    '''

# TODO: implement
def save_model(x):
    return x

for idx, loaders in enumerate(unified_loaders):

    # Model
    print('==> Building models..')
    encoder = facebook_vit(image_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=512,
                           depth=12, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                           drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm).get_ViT()

    decoder = tmpTransGan(depth1=5, depth2=2, depth3=2, initial_size=8, dim=512,
                          heads=8, mlp_ratio=4, drop_rate=0.5).get_TransGan()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    if device == 'cuda':
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)
        cudnn.benchmark = True

    criterion = nn.MSELoss()
    enc_optimizer = optim.Adam(encoder.parameters(), lr=3e-5)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=0.0001, eps=1e-08)

    if args.resume_enc:
        # Load checkpoint.
        print('==> Resuming encoder from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_encoder = torch.load('./checkpoint/ckpt_enc.pth')
        encoder.load_state_dict(checkpoint_encoder['encoder'])
        decoder.load_state_dict(checkpoint_encoder['decoder'])
        start_epoch = checkpoint_encoder['epoch']

    if args.resume_dec:
        # Load checkpoint.
        print('==> Resuming decoder from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_decoder = torch.load('./checkpoint/ckpt_dec.pth')
        encoder.load_state_dict(checkpoint_encoder['encoder'])
        decoder.load_state_dict(checkpoint_encoder['decoder'])
        start_epoch = checkpoint_decoder['epoch']
    print("Training inlier class ", classes[idx])
    for epoch in range(start_epoch, start_epoch+200):

        train(epoch, loaders[0], idx)
        test(epoch, loaders[1], idx)

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    print("saving model...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': enc_optimizer.state_dict(),
    }, "./checkpoint/ckpt_enc_" + classes[idx] + ".pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': dec_optimizer.state_dict(),
    }, "./checkpoint/ckpt_dec_" + classes[idx] + ".pth")
    print("Save complete.")
