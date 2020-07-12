##import packages
import numpy as np
from scipy.stats import poisson
from scipy.io import loadmat

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale


import matplotlib.pyplot as plt

## Load image (jpg 파일은 실행불가, png파일만 됨)
img = plt.imread("lenna.png")


## gray image generation
# img = np.mean(img, axis=2, keepdims=True)
sz = img.shape

cmap = 'gray' if sz[2] == 1 else None

plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)  #vmin&vmax = 픽셀의 value, 원래 0~255이나 0~1로 Normalize
plt.title("ground Truth")
plt.show()

## 1-1. Inpainting: Uniform sampling
ds_y =2
ds_x =4

msk = np.zeros(sz)
msk[::ds_y, ::ds_x, :] = 1

dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Groud Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Uniform sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")
## 1-2. Inpainting: random sampling
# rnd = np.random.rand(sz[0], sz[1], sz[2])
# prob = 0.5
# msk = (rnd > prob).astype(np.float)

#채널 방향으로는 동일한 샘플링 적용하는 방법
rnd = np.random.rand(sz[0], sz[1], 1)
prob = 0.5
msk = (rnd > prob).astype(np.float)

msk = np.tile(msk, (1, 1, sz[2]))


dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Random mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling Image")

plt.show()  #이게 있어야 화면에 나온다


## 1-3. inpainting Gaussian sampling / 가우시안 분포를 따르며 중앙에서 가장 많이 샘플링 된다.
ly = np.linspace(-1, 1, sz[0])
lx = np.linspace(-1, 1, sz[1])

x, y = np.meshgrid(lx, ly)

x0 = 0
y0 = 0
sgmx = 1
sgmy = 1

a = 1

# gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
# gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))

# plt.imshow(gaus)
# plt.show()

# rnd = np.random.rand(sz[0], sz[1], sz[2])
# msk = (rnd < gaus).astype(np.float)

# 채널 방향으로 같은 샘플링 적용
gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
gaus = np.tile(gaus[:, :, np.newaxis], (1,1,1))

rnd = np.random.rand(sz[0], sz[1], 1)
msk = (rnd <gaus).astype((np.float))
msk = np.tile(msk, (1, 1, sz[2]))


dst = img*msk

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")

plt.show()


## 2-1. Denoising: Random noise
sgm = 30.0

noise = sgm/255.0 * np.random.randn(sz[0], sz[1], sz[2])

dst = img * noise

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")

plt.show()

## 2-2. Denoising: poisson noise (image_domain) , 포아송 노이즈는 주로 CT 이미지에 많이 사용
dst = poisson.rvs(255.0 * img) / 255.0
noise = dst - img

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")

plt.show()

## 2-3. Denoising: poisson noise (CT-domain)


## 3. Super-resolution (downsampling image -> upsampling image) 다운/업샘플링엔 다양한 기법들이 존재
'''
-------------------
odder options
-------------------
0: Nearest-neighbor
1: bi-linear (default)
2: Bi-quadratic
3: Bi-cubic
4: Bi-quartic
5: Bi-quintic
'''

dw = 1/5.0  #down-sampling ratio
order = 0

dst_dw = rescale(img, scale=(dw, dw, 1), order=order)
dst_up = rescale(dst_dw, scale=(1/dw, 1/dw, 1), order=order)

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(msk), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")

plt.show()

# 실제로 보면 downsclaed와 upscaled 이미지가 비슷해보이지만 실제 확대하면 다르다