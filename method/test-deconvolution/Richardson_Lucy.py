import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.stats import norm  # おまけの項で使います
from scipy.stats import multivariate_normal
from scipy.signal import convolve
from skimage import restoration


"""シミュレーション（1次元バージョン）"""
# 真の画像
true_px = 128
true_img = np.zeros(true_px)
true_img[52] = 0.5
true_img[76] = 0.5

# PSF
psf_px = 127
mean = psf_px // 2
psf_sigma = 13
psf = norm.pdf(np.arange(psf_px), loc=mean, scale=psf_sigma)

print(len(true_img))
print(len(psf))

# 真の画像をPSFで畳み込む（撮像画像）
obtained_img = convolve(true_img, psf, "same")

# RL法
iterations = 50
rl_img = restoration.richardson_lucy(obtained_img, psf, iterations)

"""シミュレーションの結果を図に表示"""
fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(18, 4), ncols=4)

# 真の画像
ax1.set_title("The true image ({0}×{0} pixel)\n".format(true_px))
ax1.plot(true_img)
ax1.set_xlabel("pixel")
ax1.set_ylabel("value")
ax1.set_ylim(0, 1)

# PSF
ax2.set_title("PSF σ={0} ({1}×{1} pixel)\n".format(psf_sigma, psf_px))
ax2.plot(np.arange(-63, 64, 1), psf)
ax2.set_xlabel("pixel")
ax2.set_ylabel("probability")
ax2.set_ylim(bottom=0)

# 撮像画像
ax3.set_title("The obtained image ({0}×{0} pixel)\n".format(true_px))
ax3.plot(obtained_img)
ax3.set_xlabel("pixel")
ax3.set_ylabel("value")
ax3.set_ylim(bottom=0)

# RL法による生成画像
ax4.set_title("Richardson-Lucy deconvolution\n image iterations={0}".format(iterations))
ax4.plot(rl_img)
ax4.set_xlabel("pixel")
ax4.set_ylabel("value")
ax4.set_ylim(bottom=0)

fig.tight_layout()
plt.show()

