import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import deconvolve


def gaussian(x, mean, std_dev, random_noise):
    gaussian_list = (1.0 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)
    random_noise_list = np.random.uniform(-random_noise, random_noise, size=len(x))
    
    #gaussian_list[(x < mean - 3 * std_dev) | (x > mean + 3 * std_dev)] = 0
    gaussian_list[(x >= mean - 3 * std_dev) & (x <= mean + 3 * std_dev)] += random_noise + random_noise_list[(x >= mean - 3 * std_dev) & (x <= mean + 3 * std_dev)]
    return gaussian_list


def fft_deconvolution(signal, kernel):
    signal_fft = np.fft.fft(signal)
    kernel_fft = np.fft.fft(kernel)
    recover_fft = signal_fft / kernel_fft
    recover = np.fft.fftshift(np.fft.ifft(recover_fft))
    return np.real(recover)

def power_wiener_deconvolution(signal, kernel, noise_variance):
    # ノイズの分散を計算
    noise_power = noise_variance * np.ones_like(input_signal)

    signal_fft = fft(signal)
    kernel_fft = fft(kernel)

    # Wiener deconvolutionをFFT領域で実行
    recover_fft = ( np.conj(kernel_fft) * signal_fft ) / ( np.conj(kernel_fft) * kernel_fft * signal_fft + noise_power)

    # 逆FFTを適用して復元
    recover = np.fft.fftshift(np.fft.ifft(recover_fft))
    return np.real(recover)
    
def snr_wiener_deconvolution(signal, kernel, snr):
    signal_fft = np.fft.fft(signal)
    kernel_fft = np.fft.fft(kernel)
    
    # スペクトルの推定
    estimated_spectrum = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + 1 / snr)
    
    # Wiener deconvolution を適用
    recover_fft = signal_fft * estimated_spectrum
    # 逆FFT を使用して結果を時系列領域に変換
    recover = np.fft.fftshift(np.fft.ifft(recover_fft))
    return np.real(recover)


# histogram parameter
bin0 = 100
bin_width = 1
eps = 1e-10
min_value = - (bin0 * bin_width + round(bin_width/2, 2))
max_value = + (bin0 * bin_width + round(bin_width/2, 2)) + eps
bin_edges   = np.arange(min_value, max_value, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


mean = 0
std_dev = 10
random_noise = 0.001
kernel  = gaussian(bin_centers, mean, std_dev, random_noise)
signal  = gaussian(bin_centers, mean, std_dev, random_noise)


fft_deconv = fft_deconvolution(signal, kernel)


snr = 1000
wiener_deconv = snr_wiener_deconvolution(signal, kernel, snr)



fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(12, 3))
titles = ['Kernel', 'Signal', 'fft Deconve', 'wiener Deconv']
for i, ax in enumerate(axes.flat):
    ax.hist(bin_edges[:-1], bin_edges, weights=[kernel, signal, fft_deconv, wiener_deconv][i])
    ax.set_title(titles[i])
plt.tight_layout()
plt.show()
