import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    print("heatmap shape", len(heatmap[0]))
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def mask_input_tensor(input_tensor, mask, image_shape, unnormalize=True):
    # un normalize image for visualization
    invert_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    if unnormalize:
        input_tensor = invert_normalize(input_tensor)
    np_img = transforms.ToPILImage()(input_tensor) # convert to numpy image
    np_img = np_img.resize((np_img.shape[0], 32, 32)) # ensure size is 32x32
    np_img = np.array(np_img)[:, :, :, ::-1]
    mask = cv2.resize(mask, (mask.shape[0], np_img.shape[1], np_img.shape[2]))
    mask = show_mask_on_image(np_img, mask)

def rollout(attentions, discard_ratio, head_fusion): # Attentions: (depth, B, H, W)
    result = torch.eye(attentions[0].size(-1))
    result = result.expand(attentions[0].shape[0],-1,-1)
    print("result shape", result.shape)
    with torch.no_grad():
        for attention in attentions: #
            if head_fusion == "mean":
                print("ATTENTION SHAPE", attention.shape)
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                print(attention)
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            print("FUSED SHAPE", attention_heads_fused.shape)

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            print("flat shape", flat.shape)
            _, indices = flat.topk(k=int(flat.size(-1) * discard_ratio), dim=-1, largest=False)
            print("HOW MANY TOPK", int(flat.size(-1) * discard_ratio))
            print("TOPK", indices, indices.shape)
            print("NONZERO", indices.nonzero(), len(indices.nonzero()))
            for i in range(flat.shape[0]):
                ind = indices[i][indices[i] != 0]
                flat[i][ind] = 0
            #indices = indices[indices.nonzero()]

            #flat[indices.nonzero()] = 0

            flat = torch.reshape(flat, attention_heads_fused.shape)
            print("flat shape:", flat.shape)


            I = torch.eye(attention_heads_fused.size(-1))
            print("I shape", I.shape)
            a = (flat + 1.0 * I) / 2
            print("A sum shape", a.sum(dim=-1).shape)
            asum = a.sum(dim=-1)
            asum = torch.reshape(asum, (asum.shape[0], 1, asum.shape[1]))
            a = a / asum # (b, tdim, tdim)
            print("A shape", a.shape)

            #result = torch.matmul(a, result)
            result = torch.bmm(a, result)
            print("MIDDLE RESULT SHAPE", result.shape)

    # Look at the total attention between the class token,
    # and the image patches
    # result shape (b, tdim, tdim)

    print("RESULT SHAPE", result.shape)
    mask = result[:, 0, 1:] # tdim - 1
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(mask.shape[0], width, width).numpy()
    print("mask shape", mask.shape)
    maxes = np.max(mask, axis=(1,2))
    print("Maxes Shape 1", maxes.shape)
    print("old max shape", np.max(mask).shape)
    print("Old max", np.max(mask[0]))
    print("New max 1", maxes[0])
    #maxes = np.reshape(maxes, (maxes.shape[0], 1, 1))
    maxes = maxes[:, np.newaxis, np.newaxis]
    mask = mask / maxes

    return mask


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        print(len(output))
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        print("ATTENTIONS", len(self.attentions))
        for attention in self.attentions:
            print(len(attention))

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
