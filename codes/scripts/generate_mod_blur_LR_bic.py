import os
import sys
import cv2
import numpy as np
import torch

try:
    sys.path.append('..')
    from data.util import imresize
    import utils as util
except ImportError:
    pass

def generate_mod_LR_bic():
    # set parameters
    up_scale = 4
    mod_scale = 4
    # set data dir
    sourcedir = "/data/dataset/sr_benchmark/val_benchmark/HR/Set5/x4/"
    savedir = "/data/dataset/research/setting1/Set5/"

    # load PCA matrix of enough kernel
    print("load PCA matrix")
    pca_matrix = torch.load(
        "../../pca_matrix/DANv2/pca_matrix.pth", map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    degradation_setting = {
        "random_kernel": False,
        "code_length": 10,
        "ksize": 21,
        "pca_matrix": pca_matrix,
        "scale": up_scale,
        "cuda": True,
        "rate_iso": 1.0
    }

    # set random seed
    util.set_random_seed(0)

    saveHRpath = os.path.join(savedir, "HR", "x" + str(mod_scale))
    saveLRpath = os.path.join(savedir, "LR", "x" + str(up_scale))
    saveBicpath = os.path.join(savedir, "Bic", "x" + str(up_scale))
    saveLRblurpath = os.path.join(savedir, "LRblur", "x" + str(up_scale))
    saveKernelblurpath = os.path.join(savedir, "kernel", "x" + str(up_scale))

    if not os.path.isdir(sourcedir):
        print("Error: No source data found")
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, "HR")):
        os.mkdir(os.path.join(savedir, "HR"))
    if not os.path.isdir(os.path.join(savedir, "LR")):
        os.mkdir(os.path.join(savedir, "LR"))
    if not os.path.isdir(os.path.join(savedir, "Bic")):
        os.mkdir(os.path.join(savedir, "Bic"))
    if not os.path.isdir(os.path.join(savedir, "LRblur")):
        os.mkdir(os.path.join(savedir, "LRblur"))
    if not os.path.isdir(os.path.join(savedir, "kernel")):
        os.mkdir(os.path.join(savedir, "kernel"))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print("It will cover " + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print("It will cover " + str(saveLRpath))

    if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
    else:
        print("It will cover " + str(saveBicpath))

    if not os.path.isdir(saveLRblurpath):
        os.mkdir(saveLRblurpath)
    else:
        print("It will cover " + str(saveLRblurpath))

    if not os.path.isdir(saveKernelblurpath):
        os.mkdir(saveKernelblurpath)
    else:
        print("It will cover " + str(saveKernelblurpath))

    filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith(".png")])
    print(filepaths)
    num_files = len(filepaths)

    # kernel_map_tensor = torch.zeros((num_files, 1, 10)) # each kernel map: 1*10

    # prepare data with augementation

    for i in range(num_files):
        filename = filepaths[i]
        print("No.{} -- Processing {}".format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0 : mod_scale * height, 0 : mod_scale * width, :]
        else:
            image_HR = image[0 : mod_scale * height, 0 : mod_scale * width]
        # LR_blur, by random gaussian kernel
        img_HR = util.img2tensor(image_HR)
        C, H, W = img_HR.size()

        for sig in np.linspace(1.8, 3.2, 8):

            prepro = util.SRMDPreprocessing(sig=sig, **degradation_setting)

            LR_img, ker_map, kernel = prepro(img_HR.view(1, C, H, W), kernel=True)
            image_LR_blur = util.tensor2img(LR_img)
            image_kernel = util.tensor2img(kernel, min_max=(kernel.min().item(), kernel.max().item()))

            cv2.imwrite(os.path.join(saveLRblurpath, 'sig{}_{}'.format(sig,filename)), image_LR_blur)
            cv2.imwrite(os.path.join(saveHRpath, 'sig{}_{}'.format(sig,filename)), image_HR)
            cv2.imwrite(os.path.join(saveKernelblurpath, 'sig{}_{}'.format(sig,filename)), image_kernel)
        # LR
        image_LR = imresize(image_HR, 1 / up_scale, True)
        # bic
        image_Bic = imresize(image_LR, up_scale, True)

        # cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)

        # kernel_map_tensor[i] = ker_map
    # save dataset corresponding kernel maps
    # torch.save(kernel_map_tensor, './Set5_sig2.6_kermap.pth')
    print("Image Blurring & Down smaple Done: X" + str(up_scale))


if __name__ == "__main__":
    generate_mod_LR_bic()