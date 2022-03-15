import os
import torch
import torchvision.utils as tvutils


# ------------------------------------------------------
# -----------Reformulate degradation kernel-------------
def normkernel_to_downkernel(rescaled_blur_hr, rescaled_hr, ksize, eps=1e-10):
    blur_img = torch.rfft(rescaled_blur_hr, 3, onesided=False)
    img = torch.rfft(rescaled_hr, 3, onesided=False)

    denominator = img[:, :, :, :, 0] * img[:, :, :, :, 0] \
                  + img[:, :, :, :, 1] * img[:, :, :, :, 1] + eps

    # denominator[denominator==0] = eps

    inv_denominator = torch.zeros_like(img)
    inv_denominator[:, :, :, :, 0] = img[:, :, :, :, 0] / denominator
    inv_denominator[:, :, :, :, 1] = -img[:, :, :, :, 1] / denominator

    kernel = torch.zeros_like(blur_img).cuda()
    kernel[:, :, :, :, 0] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 0] \
                            - inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 1]
    kernel[:, :, :, :, 1] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 1] \
                            + inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 0]

    ker = convert_otf2psf(kernel, ksize)

    return ker


# ------------------------------------------------------
# -----------Constraint Least Square Filter-------------
def get_uperleft_denominator(img, kernel, grad_kernel):
    ker_f = convert_psf2otf(kernel, img.size()) # discrete fourier transform of kernel
    ker_p = convert_psf2otf(grad_kernel, img.size()) # discrete fourier transform of kernel

    denominator = inv_fft_kernel_est(ker_f, ker_p)
    numerator = torch.rfft(img, 3, onesided=False)
    deblur = deconv(denominator, numerator)
    return deblur


# --------------------------------
# --------------------------------
def inv_fft_kernel_est(ker_f, ker_p):
    inv_denominator = ker_f[:, :, :, :, 0] * ker_f[:, :, :, :, 0] \
                      + ker_f[:, :, :, :, 1] * ker_f[:, :, :, :, 1] \
                      + ker_p[:, :, :, :, 0] * ker_p[:, :, :, :, 0] \
                      + ker_p[:, :, :, :, 1] * ker_p[:, :, :, :, 1]
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.zeros_like(ker_f)
    inv_ker_f[:, :, :, :, 0] = ker_f[:, :, :, :, 0] / inv_denominator
    inv_ker_f[:, :, :, :, 1] = -ker_f[:, :, :, :, 1] / inv_denominator
    return inv_ker_f

# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    deblur_f = torch.zeros_like(inv_ker_f).cuda()
    deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
                            - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
    deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
                            + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    deblur = torch.irfft(deblur_f, 3, onesided=False)
    return deblur

# --------------------------------
# --------------------------------
def convert_psf2otf(ker, size):
    psf = torch.zeros(size).cuda()
    # circularly shift
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    otf = torch.rfft(psf, 3, onesided=False)
    return otf

# --------------------------------
# --------------------------------
def convert_otf2psf(otf, size):
    ker = torch.zeros(size).cuda()
    psf = torch.irfft(otf, 3, onesided=False)

    # circularly shift
    ksize = size[-1]
    centre = ksize//2 + 1

    ker[:, :, (centre-1):, (centre-1):] = psf[:, :, :centre, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, (centre-1):, :(centre-1)] = psf[:, :, :centre, -(centre-1):]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), (centre-1):] = psf[:, :, -(centre-1):, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), :(centre-1)] = psf[:, :, -(centre-1):, -(centre-1):]#.mean(dim=1, keepdim=True)

    return ker


def zeroize_negligible_val(k, n=40):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    pc = k.shape[-1]//2 + 1
    k_sorted, indices = torch.sort(k.flatten(start_dim=1))
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[:, -n - 1]
    # Clip values lower than the minimum value
    filtered_k = torch.clamp(k - k_n_min.view(-1, 1, 1, 1), min=0, max=1.0)
    filtered_k[:, :, pc, pc] += 1e-20
    # Normalize to sum to 1
    norm_k = filtered_k / torch.sum(filtered_k, dim=(2, 3), keepdim=True)
    return norm_k

def postprocess(*images, rgb_range):
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]