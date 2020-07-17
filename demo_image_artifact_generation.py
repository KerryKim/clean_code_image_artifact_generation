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
# 채널 방향으로 평균을 내주면 gray scale image를 생성할 수 있습니다.
# img = np.mean(img, axis=2, keepdims=True)
sz = img.shape

cmap = 'gray' if sz[2] == 1 else None

'''
np.squeeze는 차원이 1인 원소를 삭제한다.
Remove single-dimensional entries from the shape of an array.
https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
차원이 없어진다는 건 꺾쇠가 하나 벗겨진다는 의미이다.
'''

plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)  #vmin&vmax = 픽셀의 value, 원래 0~255이나 0~1로 Normalize
plt.title("ground Truth")
plt.show()

# Sampling mask를 생성해 봅시다. (uniform mask, uniform random mask, gaussian random mask)
# Undersampling 기법을 활용합니다.

## 1-1. Inpainting: Uniform sampling
# 먼저 x방향과 y방향의 sampling ratio를 설정합니다.
ds_y =2
ds_x =4

msk = np.zeros(sz)
msk[::ds_y, ::ds_x, :] = 1  # ds_y와 ds_x 간격으로 sampling하는 마스크를 만들어 줍니다.

dst = img*msk


# vmin=0, vmax=1은 image value=0~255 값을 0과 1로 Normalize해주는 방법이다.
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
# 이미지 사이즈와 동일한 random variable 설정. 즉 세로, 가로, 채널의 shape
rnd = np.random.rand(sz[0], sz[1], sz[2])
prob = 0.5
msk = (rnd > prob).astype(np.float)

'''
#채널 방향으로는 동일한 샘플링 적용하는 방법 (위 방법은 채널도 랜덤 샘플링이 적용됨)
rnd = np.random.rand(sz[0], sz[1], 1)
prob = 0.5
msk = (rnd > prob).astype(np.float)

msk = np.tile(msk, (1, 1, sz[2]))
'''

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

plt.show()  # 이게 있어야 화면에 나온다


## 1-3. inpainting Gaussian sampling
# 가우시안 분포를 따르며 중앙에서 가장 많이 샘플링 된다.

'''
np.linspace()와 np.arage()는 등차수열을 생성한다.
둘 다 거의 유사하지만 np.linspace()는 시작과 끝을 주어진 개수로 나누고 
np.arange()는 시작부터 주어진 간격을 공차로 하는 등차수열을 만들어 끝점보다 커저기 직전까지 생성한다.
https://blog.naver.com/boubleman/221666484068
'''
ly = np.linspace(-1, 1, sz[0])
lx = np.linspace(-1, 1, sz[1])


# 동영상 참고
x, y = np.meshgrid(lx, ly)

x0 = 0
y0 = 0
sgmx = 1
sgmy = 1

a = 1

# 2D-Gaussian Distribution sampling mask
# np.tile 동일한 배열을 반복하여 연결하는 메소드.
# 아래에서는 gaus[:, :, np.newaxis]로 layer한장을 만든 다음 (1, 1, sz[2])로 3채널을 하나로 만드는 것으로 보임.
# gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
# gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))

# plt.imshow(gaus)
# plt.show()

# rnd = np.random.rand(sz[0], sz[1], sz[2])
# msk = (rnd < gaus).astype(np.float)

# Random sampling 예제처럼 채널 방향으로 같은 샘플링 적용
# rnd = np.random.rand(sz[0], sz[1], sz[2])에서  sz[2]만 1로 바뀐다.
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
# BM3D 논문을 검색하면 sigma/PNSR에 따라서 해당영상의 노이즈 정도를 판단한다.
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
order options (downsampling을 하는 방법들)
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

# 실제로 보면 downsclaed와 upscaled 이미지가 비슷해보이지만
# 실제로 Upscaled image를 확대하면 5by5 pixel이 하나의 value 즉 같은 색으로 칠해져 있는 걸 알 수 있다.