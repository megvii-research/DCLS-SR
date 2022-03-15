import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import util

ker_wid_x4 = np.random.choice([0.2, 4.0], 1, replace=False)
blur_ker = util.isogkern(21, ker_wid_x4[0])

kernel = util.random_isotropic_gaussian_kernel(
    sig_min=0.2, sig_max=4.0, l=21, tensor=False
)

plt.imshow(kernel)
plt.colorbar()
plt.show()
