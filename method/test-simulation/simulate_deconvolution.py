import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration

def main():

    def hist_parameters(bin_zero, bin_width):
        eps = 1e-10
        min_value = - (bin_zero * bin_width + round(bin_width/2, 2))
        max_value = + (bin_zero * bin_width + round(bin_width/2, 2)) + eps
        bin_edges   = np.arange(min_value, max_value, bin_width)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        nbins = len(bin_centers)
        
        return bin_centers, bin_edges


    def get_real_charge_data():
        file_path = "./test_data/Test_Charge_HistData_On.npy"
        hist_on = np.load(file_path)
        norm_hist_on = hist_on / np.sum(hist_on)

        file_path = "./test_data/Test_Charge_HistData_Off.npy"
        hist_off = np.load(file_path)
        norm_hist_off = hist_off / np.sum(hist_off)
        
        bin_zero = 1500
        bin_width = 0.16
        bin_centers, bin_edges = hist_parameters(bin_zero, bin_width)
        
        return norm_hist_on, norm_hist_off, bin_centers, bin_edges


    def make_ideal_simu_data():
        # ヒストグラムのパラメータ設定
        bin_zero = 50
        bin_width = 1
        bin_centers, bin_edges = hist_parameters(bin_zero, bin_width)
        
        # 平均(mu)と標準偏差(sigma)を指定
        signal_mu = 0
        signal_sigma = 5
        noise_mu = 10
        noise_sigma = 10

        # ガウス分布に従う10000のデータ点を生成
        data_points = 10000
        signal_data = np.random.normal(signal_mu, signal_sigma, data_points)
        noise_data = np.random.normal(noise_mu, noise_sigma, data_points)

        hist_signal, _ = np.histogram(signal_data, bins=bin_edges, density=True)
        hist_noise, _ = np.histogram(noise_data, bins=bin_edges, density=True)
        hist_NOISY_signal = np.convolve(hist_signal, hist_noise, mode='same')

        print(np.sum(hist_signal))
        print(np.sum(hist_noise))
        print(np.sum(hist_NOISY_signal))

        return hist_signal, hist_noise, hist_NOISY_signal, bin_centers, bin_edges


    def make_statistical_fluctuation_simu_data():
        # ヒストグラムのパラメータ設定
        bin_zero = 50
        bin_width = 1
        bin_centers, bin_edges = hist_parameters(bin_zero, bin_width)

    
    hist_signal, hist_noise, hist_NOISY_signal, bin_centers, bin_edges = make_ideal_simu_data()
    
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(16, 3))
    titles = ['Signal', 'Noise', 'NOISY_Signal']
    hist_data = [hist_signal, hist_noise, hist_NOISY_signal]
    for i, ax in enumerate(axes.flat):
        ax.hist(bin_centers, bins=bin_edges, weights=hist_data[i])
        ax.set_title(titles[i])
    plt.tight_layout()
    


    wiener_deconvolved = restoration.wiener(hist_NOISY_signal, hist_noise, balance=1)
    rl_deconvolved = restoration.richardson_lucy(hist_NOISY_signal, hist_noise)
    
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(16, 3))
    titles = ['Signal', 'Wiener', 'RL']
    hist_data = [hist_signal, wiener_deconvolved, rl_deconvolved]
    for i, ax in enumerate(axes.flat):
        ax.hist(bin_centers, bins=bin_edges, weights=hist_noise)
        ax.hist(bin_centers, bins=bin_edges, weights=hist_data[i])
        ax.set_title(titles[i])
    plt.tight_layout()
    



    ############################################################################
    
    
    

    hist_on, hist_off, bin_centers, bin_edges = get_real_charge_data()

    wiener_deconvolved = restoration.wiener(hist_on, hist_off, balance=1)
    rl_deconvolved = restoration.richardson_lucy(hist_on, hist_off)
    
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(16, 3))
    titles = ['On', 'Wiener', 'RL']
    hist_data = [hist_on, wiener_deconvolved, rl_deconvolved]
    for i, ax in enumerate(axes.flat):
        ax.hist(bin_centers, bins=bin_edges, weights=hist_off)
        ax.hist(bin_centers, bins=bin_edges, weights=hist_data[i])
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()


    #fft_deconv = fft_deconvolution(hist_NOISY_signal, hist_noise)
    #snr = 1000
    #wiener_deconv = snr_wiener_deconvolution(hist_NOISY_signal, hist_noise, snr)



'''
    bin_zero  = 100
    bin_width = 1
    eps = 1e-10
    min_value = - (bin_zero * bin_width + round(bin_width/2, 2))
    max_value = + (bin_zero * bin_width + round(bin_width/2, 2)) + eps
    bin_edges   = np.arange(min_value, max_value, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    nbins = len(bin_centers)
    
    のヒストグラムのパラメータのもとで、
    
    def make_ideal_simu_data(bin_edges):
        # 平均(mu)と標準偏差(sigma)を指定
        signal_mu = 0
        signal_sigma = 5
        noise_mu = 10
        noise_sigma = 10

        # ガウス分布に従う100のデータ点を生成
        data_points = 100
        signal_data = np.random.normal(signal_mu, signal_sigma, data_points)
        noise_data = np.random.normal(noise_mu, noise_sigma, data_points)

        hist_signal, _ = np.histogram(signal_data, bins=bin_edges, density=True)
        hist_noise, _ = np.histogram(noise_data, bins=bin_edges, density=True)
        hist_NOISY_signal = np.convolve(hist_signal, hist_noise, mode='same')
        
    で生成されたhist_signal, hist_noise, hist_NOISY_signalをnp.sum()で計算すると、１になるが、 bin_widthの値を1から0.1等に変えると、hist分布の合計の値（np.sum()で計算した値）が1にならないのはなぜ？
'''
