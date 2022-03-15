import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.util import imresize
from scipy.io import loadmat
from torch.autograd import Variable


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], "Scale [{}] is not supported".format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi

        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], "reflect")

    gaussian_filter = (
        torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    )
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k]  # PCA matrix


def random_batch_kernel(
    batch,
    l=21,
    sig_min=0.2,
    sig_max=4.0,
    rate_iso=1.0,
    tensor=True,
    random_disturb=False,
):

    if rate_iso == 1:

        sigma = np.random.uniform(sig_min, sig_max, (batch, 1, 1))
        ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xx = xx[None].repeat(batch, 0)
        yy = yy[None].repeat(batch, 0)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)
        return torch.FloatTensor(kernel) if tensor else kernel

    else:

        sigma_x = np.random.uniform(sig_min, sig_max, (batch, 1, 1))
        sigma_y = np.random.uniform(sig_min, sig_max, (batch, 1, 1))

        D = np.zeros((batch, 2, 2))
        D[:, 0, 0] = sigma_x.squeeze() ** 2
        D[:, 1, 1] = sigma_y.squeeze() ** 2

        radians = np.random.uniform(-np.pi, np.pi, (batch))
        mask_iso = np.random.uniform(0, 1, (batch)) < rate_iso
        radians[mask_iso] = 0
        sigma_y[mask_iso] = sigma_x[mask_iso]

        U = np.zeros((batch, 2, 2))
        U[:, 0, 0] = np.cos(radians)
        U[:, 0, 1] = -np.sin(radians)
        U[:, 1, 0] = np.sin(radians)
        U[:, 1, 1] = np.cos(radians)
        sigma = np.matmul(U, np.matmul(D, U.transpose(0, 2, 1)))
        ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
        xy = xy[None].repeat(batch, 0)
        inverse_sigma = np.linalg.inv(sigma)[:, None, None]
        kernel = np.exp(
            -0.5
            * np.matmul(
                np.matmul(xy[:, :, :, None], inverse_sigma), xy[:, :, :, :, None]
            )
        )
        kernel = kernel.reshape(batch, l, l)
        if random_disturb:
            kernel = kernel + np.random.uniform(0, 0.25, (batch, l, l)) * kernel
        kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)

        return torch.FloatTensor(kernel) if tensor else kernel


def stable_batch_kernel(batch, l=21, sig=2.6, tensor=True):
    sigma = sig
    ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xx = xx[None].repeat(batch, 0)
    yy = yy[None].repeat(batch, 0)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)
    return torch.FloatTensor(kernel) if tensor else kernel


def b_Bicubic(variable, scale):
    B, C, H, W = variable.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = variable.view((B, C, H, W))
    re_tensor = imresize(tensor_v, 1 / scale)
    return re_tensor


def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(
        torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)),
        sigma.view(sigma.size() + (1, 1)),
    ).to(tensor.device)
    return torch.clamp(noise + tensor, min=min, max=max)

def b_GaussianNoising(tensor, noise_high, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.FloatTensor(
        np.random.normal(loc=mean, scale=noise_high, size=size)
        ).to(tensor.device)
    return torch.clamp(noise + tensor, min=min, max=max)


class BatchSRKernel(object):
    def __init__(
        self,
        l=21,
        sig=2.6,
        sig_min=0.2,
        sig_max=4.0,
        rate_iso=1.0,
        random_disturb=False,
    ):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.random_disturb = random_disturb

    def __call__(self, random, batch, tensor=False):
        if random == True:  # random kernel
            return random_batch_kernel(
                batch,
                l=self.l,
                sig_min=self.sig_min,
                sig_max=self.sig_max,
                rate_iso=self.rate,
                tensor=tensor,
                random_disturb=self.random_disturb,
            )
        else:  # stable kernel
            return stable_batch_kernel(batch, l=self.l, sig=self.sig, tensor=tensor)


class BatchBlurKernel(object):
    def __init__(self, kernels_path):
        kernels = loadmat(kernels_path)["kernels"]
        self.num_kernels = kernels.shape[0]
        self.kernels = kernels

    def __call__(self, random, batch, tensor=False):
        index = np.random.randint(0, self.num_kernels, batch)
        kernels = self.kernels[index]
        return torch.FloatTensor(kernels).contiguous() if tensor else kernels


class PCAEncoder(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)
        self.size = self.weight.size()

    def forward(self, batch_kernel):
        B, H, W = batch_kernel.size()  # [B, l, l]
        return torch.bmm(
            batch_kernel.view((B, 1, H * W)), self.weight.expand((B,) + self.size)
        ).view((B, -1))


class BatchBlur(object):
    def __init__(self, l=15):
        self.l = l
        if l % 2 == 1:
            self.pad =(l // 2, l // 2, l // 2, l // 2)
        else:
            self.pad = (l // 2, l // 2 - 1, l // 2, l // 2 - 1)
        # self.pad = nn.ZeroPad2d(l // 2)

    def __call__(self, input, kernel):
        B, C, H, W = input.size()
        pad = F.pad(input, self.pad, mode='reflect')
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = (
                kernel.contiguous()
                .view((B, 1, self.l, self.l))
                .repeat(1, C, 1, 1)
                .view((B * C, 1, self.l, self.l))
            )
            return F.conv2d(input_CBHW, kernel_var, groups=B * C).view((B, C, H, W))


class SRMDPreprocessing(object):
    def __init__(
        self, scale, pca_matrix,
        ksize=21, code_length=10,
        random_kernel = True, noise=False, cuda=False, random_disturb=False,
        sig=0, sig_min=0, sig_max=0, rate_iso=1.0, rate_cln=1., noise_high=0,
        stored_kernel=False, pre_kernel_path = None
    ):
        self.encoder = PCAEncoder(pca_matrix).cuda() if cuda else PCAEncoder(pca)

        self.kernel_gen = (
            BatchSRKernel(
                l=ksize,
                sig=sig, sig_min=sig_min, sig_max=sig_max,
                rate_iso=rate_iso, random_disturb=random_disturb,
            ) if not stored_kernel else
            BatchBlurKernel(pre_kernel_path)
        )

        self.blur = BatchBlur(l=ksize)
        self.para_in = code_length
        self.l = ksize
        self.noise = noise
        self.scale = scale
        self.cuda = cuda
        self.rate_cln = rate_cln
        self.noise_high = noise_high
        self.random = random_kernel
        self.rate_iso = rate_iso

    def __call__(self, hr_tensor, kernel=False, return_blur=False):
        # hr_tensor is tensor, not cuda tensor

        hr_var = Variable(hr_tensor).cuda() if self.cuda else Variable(hr_tensor)
        device = hr_var.device
        B, C, H, W = hr_var.size()

        b_kernels = Variable(self.kernel_gen(self.random, B, tensor=True)).to(device)
        hr_blured_var = self.blur(hr_var, b_kernels)

        # B x self.para_input
        kernel_code = self.encoder(b_kernels)

        # Down sample
        if self.scale != 1:
            lr_blured_t = b_Bicubic(hr_blured_var, self.scale)
            lr_t = b_Bicubic(hr_var, self.scale)

        else:
            lr_blured_t = hr_blured_var
            lr_t = hr_var


        # Noisy
        if self.noise:
            Noise_level = torch.FloatTensor(
                random_batch_noise(B, self.noise_high, self.rate_cln)
            )
            lr_noised_t = b_GaussianNoising(lr_blured_t, self.noise_high)
        else:
            Noise_level = torch.zeros((B, 1))
            lr_noised_t = lr_blured_t

        Noise_level = Variable(Noise_level).cuda()
        re_code = (
            torch.cat([kernel_code, Noise_level * 10], dim=1)
            if self.noise
            else kernel_code
        )
        lr_re = Variable(lr_noised_t).to(device)

        if return_blur:
            return (lr_re, re_code, b_kernels, lr_blured_t, lr_t)

        return (lr_re, re_code, b_kernels) if kernel else (lr_re, re_code)
