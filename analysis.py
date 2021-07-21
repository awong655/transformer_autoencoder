from os import listdir
from os.path import isfile, join

import PIL.Image
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
from collections import OrderedDict

from autoenc.ae_model import tmpViT, tmpTransGan, facebook_vit
from sklearn.decomposition import PCA
from visualize_attn import visualize_attn
from attention_rollout import VITAttentionRollout
from grad_rollout import VITAttentionGradRollout
import cv2
from PIL import Image
from torchvision import transforms

"""
Evaluating the performance of Transformer Encoder
"""

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def load_models(checkpoint_directory):
    ckpt_enc = [f for f in listdir(checkpoint_directory) if isfile(join(checkpoint_directory, f)) and "enc" in f]
    ckpt_dec = [f for f in listdir(checkpoint_directory) if isfile(join(checkpoint_directory, f)) and "dec" in f]
    ckpt_enc.sort()
    ckpt_dec.sort()
    return zip(ckpt_enc, ckpt_dec)

def get_data(data_directory):
    # Data
    print('==> Preparing data..')

    trainset = torchvision.datasets.CIFAR10(
        root='data_directory', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='data_directory', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
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
    inliers = [np.concatenate((i, j), dtype=np.int64) for i, j in inliers_zip]

    outliers_zip = zip(train_outliers, test_outliers)
    outliers = [np.concatenate((i, j), dtype=np.int64) for i, j in outliers_zip]

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
            shuffle=False,
            num_workers=2
        ) for inds in outliers]
    unified_loaders = list(zip(trainloader, testloader))
    return unified_loaders

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    print("heatmap shape", len(heatmap[0]))
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
    #return heatmap

def test(device, encoder, decoder, criterion, testloader, loader_idx):
    global best_acc
    #net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            print("Input shape", inputs.shape)
            is_gpu = next(encoder.parameters()).is_cuda
            # Get activation
            #encoder.transformer.layers[5][0].fn.attend.register_forward_hook(get_activation('transformer.layers.5.0.fn.attend'))
            #hooks = {}
            #for name, module in encoder.named_modules():
            #    hooks[name] = module.register_forward_hook(get_activation(name))

            encodings = encoder(inputs)
            recons = decoder(encodings)
            loss = criterion(recons, inputs)

            errors = inputs - recons
            #errors = torch.abs(errors)
            #errors = errors
            #attentions = encoder.get_last_selfattention(inputs)
            #rint("Attentions Shape", attentions.shape)
            #visualize_attn(attentions=attentions, img_height=32, img_width=32, patch_size=4)

            model = torch.hub.load('facebookresearch/deit:main',
                                   'deit_tiny_patch16_224', pretrained=True)
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            img = Image.open('plane4.jpg')
            img = img.resize((32, 32), resample=PIL.Image.BILINEAR)
            print("NUMPY IMAGE SIZE", img.size)
            input_tensor = transform(img).unsqueeze(0)

            for i in range(1):
                #fusions = ['min', 'mean', 'max']
                fusions = ['mean']
                print("INPUTS", inputs.shape)
                current_inp = inputs[1]
                mask = VITAttentionRollout(encoder, head_fusion=fusions[i], discard_ratio=0.9)(torch.stack([inputs[0], inputs[1]]))
                print("MASK OUTPUT SHAPE", mask.shape)
                mask = mask[1]
                #attn_grad_rollout = VITAttentionGradRollout(encoder)(inputs)
                print(mask.shape)

                name = "attention_rollout_{:.3f}_{}.png".format(0.1, fusions[i])

                inv_normalize = transforms.Normalize(
                    mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
                    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
                )
                inv_tensor = inv_normalize(current_inp)

                np_img = torchvision.transforms.ToPILImage()(inv_tensor)
                np_img = np_img.resize((32, 32))

                np_img = np.array(np_img)[:, :, ::-1]

                mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
                print("MASK SHAPE:", mask.shape)
                mask = show_mask_on_image(np_img, mask)
                #save_image(encodings[0].unsqueeze[0], "cifar_imgs/input.jpg")
                cv2.imwrite("cifar_imgs/input.png", np_img)
                cv2.imwrite("cifar_imgs/"+name, mask)
            '''
            activations = activation['to_patch_embedding']
            print(activations.shape)
            activations = activations.to('cpu')
            activations = torch.reshape(activations, (activations.shape[0], activations.shape[1]*activations.shape[2] ))
            print("RESHAPED", activations.shape)
            activation_tensors = np.array(28)
            #for i in range(2, 29):
            pca = PCA(n_components=2)
            pca.fit(activations)
            X = pca.transform(activations)
            X = torch.from_numpy(X)
            print("Explained Variance: W", pca.explained_variance_ratio_)
            #activation_tensors[i-2] = X
            #print(activation_tensors)
            '''


            #test_loss += loss.item()

            print("TEST: Inlier: ", classes[batch_idx], "Loss: ", (test_loss/(batch_idx+1)))

            if (batch_idx % 500 == 0):

                cpu_inp = inputs.cpu()
                cpu_recons = recons.cpu()
                save_image(make_grid(cpu_inp, nrows=10), "./cifar_imgs/test_input_Inlier: " + classes[loader_idx] + "_" + str(batch_idx) + ".jpg")
                save_image(make_grid(cpu_recons, nrows=10), "./cifar_imgs/test_recon_Inlier:" + classes[loader_idx] + "_" + str(batch_idx) + ".jpg")
                save_image(make_grid(errors, nrows=10, normalize=False),
                           "./cifar_imgs/test_err_Inlier:" + classes[loader_idx] + "_" + str(
                               batch_idx) + ".jpg")

                save_image(attn_mask, "./cifar_imgs/test_act_Inlier:" + classes[loader_idx] + "_" + str(batch_idx) + ".jpg")

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    print("DEVICE:", device)

    checkpoints = list(load_models(args.checkpoint_directory))
    #unified_loader = get_data(args.data_directory)

    print("ENCODER ARCHITECTURE")
    encoder = facebook_vit(image_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=512,
                           depth=12, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                           drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm).get_ViT()
    print(encoder)

    print("DECODER ARCHITECTURE")
    decoder = tmpTransGan(depth1=5, depth2=2, depth3=2, initial_size=8, dim=512,
                          heads=8, mlp_ratio=4, drop_rate=0.5).get_TransGan()
    print(decoder)

    print("Grabbing Data")

    unified_loader = get_data(args.data_directory)

    for checkpoint in checkpoints:
        print("Loading Checkpoint")
        criterion = nn.MSELoss()

        # Load models
        enc = facebook_vit(image_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=512,
                           depth=12, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                           drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm).get_ViT()
        tmp_dict = torch.load(args.checkpoint_directory + "/" + checkpoint[0])['model_state_dict']
        enc_state_dict = OrderedDict()
        for k, v in tmp_dict.items():
            name = k[7:]  # remove `module.`
            enc_state_dict[name] = v
        enc.load_state_dict(enc_state_dict)
        enc.eval()
        enc.to(device)

        dec = tmpTransGan(depth1=5, depth2=2, depth3=2, initial_size=8, dim=512,
                              heads=8, mlp_ratio=4, drop_rate=0.5).get_TransGan()
        tmp_dict = torch.load(args.checkpoint_directory + "/" + checkpoint[1])['model_state_dict']
        dec_state_dict = OrderedDict()
        for k, v in tmp_dict.items():
            name = k[7:]  # remove `module.`
            dec_state_dict[name] = v
        dec.load_state_dict(dec_state_dict)
        dec.eval()
        dec.to(device)

        for idx, loader in enumerate(unified_loader):
            trainloader = loader[0]
            testloader = loader[1]
            test(device, enc, dec, criterion, trainloader, idx)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_directory',
                        help='Directory where checkpoints are stored',
                        type=str, default='./checkpoint')
    parser.add_argument('--data_directory',
                        help='Directory where data is  stored',
                        type=str, default='./data')

    args = parser.parse_args()
    main(args)





