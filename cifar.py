'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from local_models import *
#from utils import progress_bar
from torchvision.utils import save_image, make_grid

from autoenc.ae_model import tmpViT, tmpTransGan

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
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building models..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
encoder = tmpViT(image_size=32, patch_size=4, num_classes=10, dim=512, depth=6, heads=16,
                 mlp_dim=1024, dropout=0.1, emb_dropout=0.1, keep_head=False).get_ViT()

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
#scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

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

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    #net.train()
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

        if (batch_idx % 50 == 0):
            print("writing image")
            cpu_inp = inputs.cpu()
            cpu_recons = recons.cpu()
            save_image(make_grid(cpu_inp, nrows=10),
                       "./cifar_imgs/train_input_" + str(epoch) + "_" + str(batch_idx) + ".jpg")
            save_image(make_grid(cpu_recons, nrows=10),
                       "./cifar_imgs/train_recon_" + str(epoch) + "_" + str(batch_idx) + ".jpg")
        train_loss += loss.item()
        #_, predicted = outputs.max(1)
        #total += targets.size(0)
        #correct += predicted.eq(targets).sum().item()

        print("Epoch No. ", epoch, "Batch Index.", batch_idx, "Loss: ", (train_loss/(batch_idx+1)))

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f'
        #             % (train_loss/(batch_idx+1)))


def test(epoch):
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

            print("Epoch No. ", epoch, "Batch Index.", batch_idx, "Loss: ", (test_loss/(batch_idx+1)))

            if (batch_idx % 50 == 0):
                cpu_inp = inputs.cpu()
                cpu_recons = recons.cpu()
                save_image(make_grid(cpu_inp, nrows=10), "./cifar_imgs/test_input_" + str(epoch) + "_" + str(batch_idx) + ".jpg")
                save_image(make_grid(cpu_recons, nrows=10), "./cifar_imgs/test_recon_" + str(epoch) + "_" + str(batch_idx) + ".jpg")

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
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

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    print("saving model...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': enc_optimizer.state_dict(),
    }, "./checkpoint/ckpt_enc.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': dec_optimizer.state_dict(),
    }, "./checkpoint/ckpt_dec.pth")
    print("Save complete.")
    #scheduler.step()
