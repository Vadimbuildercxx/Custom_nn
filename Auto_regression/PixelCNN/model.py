import numpy as np

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MaskedConv2dPartial(torch.nn.Conv2d):
    def __init__(self, conv_type, centered: bool, *args, **kwargs):
        super(MaskedConv2dPartial, self).__init__(*args, **kwargs)

        assert conv_type in ["V", "H"]

        out_channels, in_channels, height, width = self.weight.size()
        yc, xc = height // 2, width // 2

        mask = torch.zeros(self.weight.size(), dtype=torch.float32)
        if conv_type == "V":
            if centered:
                mask[:, :, :yc + 1, :] = 1
            else:
                mask[:, :, :yc, :] = 1
        else:
            if centered:
                mask[:, :, yc, :xc + 1] = 1
            else:
                mask[:, :, yc, :xc] = 1


        def cmask(out_c, in_c):
            a = (np.arange(out_channels) % 1 == out_c)[:, None]
            b = (np.arange(in_channels) % 1 == in_c)[None, :]
            return a * b

        for o in range(1):
            for i in range(o + 1, 1):
                mask[cmask(o, i), yc, xc] = 0

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(MaskedConv2dPartial, self).forward(x)
        return x

class GatedMaskedConv(torch.nn.Module):

    def __init__(self, c_in, c_out, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = MaskedConv2dPartial("V", True, c_in, 2*c_out, **kwargs)
        self.conv_horiz = MaskedConv2dPartial("H", True, c_in, 2*c_out, **kwargs)
        self.conv_vert_to_horiz = torch.nn.Conv2d(2*c_out, 2*c_out, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = torch.nn.Conv2d(c_out, c_out, kernel_size=1, padding=0)
        self.conv_horiz_skip = torch.nn.Conv2d(c_in, c_out, kernel_size=self.conv_horiz.kernel_size, padding=self.conv_horiz.padding)

        # Weight init
        torch.nn.init.kaiming_normal_(self.conv_vert.weight)
        torch.nn.init.kaiming_normal_(self.conv_horiz.weight)

        torch.nn.init.kaiming_normal_(self.conv_vert_to_horiz.weight)
        torch.nn.init.kaiming_normal_(self.conv_horiz_1x1.weight)
        torch.nn.init.kaiming_normal_(self.conv_horiz_skip.weight)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)

        # Fully connected layers
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)

        # Skip connection
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


class PixelCNN(torch.nn.Module):

    def __init__(self, c_in, inner_dim = 64):
        super().__init__()

        self.digit_embs = torch.nn.Embedding(10, inner_dim, padding_idx=0)

        # Initial convolutions skipping the center pixel
        self.conv_vstack = MaskedConv2dPartial("V", False, c_in, inner_dim, kernel_size = (3, 3), padding=(1, 1))
        self.conv_hstack = MaskedConv2dPartial("H", False, c_in, inner_dim, kernel_size = (3, 3), padding=(1, 1))
        # Convolution block of PixelCNN.
        self.conv_layers = torch.nn.ModuleList([
            GatedMaskedConv(inner_dim, inner_dim, kernel_size = (7, 7), padding = (3, 3)),
            GatedMaskedConv(inner_dim, inner_dim, kernel_size = (3, 3), padding = (2, 2), dilation = 2),
            GatedMaskedConv(inner_dim, inner_dim, kernel_size = (3, 3), padding = (2, 2), dilation = 2),
            GatedMaskedConv(inner_dim, inner_dim, kernel_size = (3, 3), padding = (2, 2), dilation = 2),
            GatedMaskedConv(inner_dim, inner_dim, kernel_size = (3, 3), padding = (2, 2), dilation = 2),
            GatedMaskedConv(inner_dim, inner_dim, kernel_size = (3, 3), padding = (2, 2), dilation = 2),
            GatedMaskedConv(inner_dim, inner_dim, kernel_size = (3, 3), padding = (2, 2), dilation = 2),
            GatedMaskedConv(inner_dim, inner_dim, kernel_size = (3, 3), padding = (1, 1)),
        ])
        self.batch_norm2d = torch.nn.BatchNorm2d(inner_dim)
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(inner_dim, c_in * 256, kernel_size=1, padding=0)

    def forward(self, x, cls_label):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to 255 back to -1 to 1
        x = torchvision.transforms.Normalize((0.5, ), (0.5, ))(x.float())

        # Initial convolutions
        v_stack = self.conv_vstack(x) + self.digit_embs(cls_label).unsqueeze(2).unsqueeze(2)
        h_stack = self.conv_hstack(x) + self.digit_embs(cls_label).unsqueeze(2).unsqueeze(2)

        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)

        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(out.shape[0], 256, out.shape[1]//256, out.shape[2], out.shape[3])

        return out

    @torch.no_grad()
    def sample(self, img_shape, img=None, cls_labels = None):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        if img is None:
            img = torch.zeros(img_shape, dtype=torch.long).to(device) - 1
        if cls_labels is None:
            cls_labels = torch.randint(0, 10, (img_shape[0], )).to(device)
            print(f"Predict classes: {cls_labels}")

        # Generation loop
        for h in tqdm(range(img_shape[2]), leave=False):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # Skip if not to be filled (-1)
                    if (img[:,c,h,w] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    pred = self.forward(img[:,:,:,:], cls_labels) #img[:,:,:h+1,:]
                    probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                    img[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        return img, cls_labels