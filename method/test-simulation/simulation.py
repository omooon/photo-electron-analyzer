import numpy as np
import matplotlib.pyplot as plt
import math

nentries = 40000

bin_width = 0.16
num_positive_bins = 3000
num_negative_bins = 1000
positive_bin_edges = np.arange(bin_width, bin_width * (num_positive_bins + 2), bin_width) - bin_width/2
negative_bin_edges = np.arange(-bin_width, -bin_width * (num_negative_bins + 2), -bin_width) + bin_width/2
bin_edges = np.concatenate((negative_bin_edges[::-1], positive_bin_edges))
bin_range = (bin_edges[0], bin_edges[-1])
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2





## sumilation_noise_histの作成 ##
simu_noise_charge = np.random.normal(0, 10, nentries)
simu_noise_hist, _ = np.histogram(simu_noise_charge, bins=bin_edges)



## simulation_signal_histの作成 ##
simu_1PE_charge = np.random.normal(20, 10, nentries)
simu_1PE_hist, _ = np.histogram(simu_1PE_charge, bins=bin_edges)

simu_2PE_charge = np.random.normal(40, 14, nentries)
simu_2PE_hist, _ = np.histogram(simu_2PE_charge, bins=bin_edges)

simu_3PE_charge = np.random.normal(60, 17, nentries)
simu_3PE_hist, _ = np.histogram(simu_3PE_charge, bins=bin_edges)

simu_4PE_charge = np.random.normal(80, 20, nentries)
simu_4PE_hist, _ = np.histogram(simu_4PE_charge, bins=bin_edges)

simu_5PE_charge = np.random.normal(100, 22, nentries)
simu_5PE_hist, _ = np.histogram(simu_5PE_charge, bins=bin_edges)



simu_0PE_signal_hist = simu_noise_hist * (0.3**0 * math.exp(-0.3) / 1)
simu_1PE_hist = simu_1PE_hist * (0.3**1 * math.exp(-0.3) / 1)
simu_2PE_hist = simu_2PE_hist * (0.3**2 * math.exp(-0.3) / 2)
simu_3PE_hist = simu_3PE_hist * (0.3**3 * math.exp(-0.3) / 6)
simu_4PE_hist = simu_4PE_hist * (0.3**4 * math.exp(-0.3) / 24)
simu_5PE_hist = simu_5PE_hist * (0.3**5 * math.exp(-0.3) / 120)



simu_1PE_signal_hist = np.convolve(simu_1PE_hist, simu_0PE_signal_hist) / (np.sum(simu_0PE_signal_hist))
simu_1PE_signal_hist = simu_1PE_signal_hist[1000:len(bin_centers)+1000]

simu_2PE_signal_hist = np.convolve(simu_2PE_hist, simu_0PE_signal_hist) / (np.sum(simu_0PE_signal_hist))
simu_2PE_signal_hist = simu_2PE_signal_hist[1000:len(bin_centers)+1000]

simu_3PE_signal_hist = np.convolve(simu_3PE_hist, simu_0PE_signal_hist) / (np.sum(simu_0PE_signal_hist))
simu_3PE_signal_hist = simu_3PE_signal_hist[1000:len(bin_centers)+1000]

simu_4PE_signal_hist = np.convolve(simu_4PE_hist, simu_0PE_signal_hist) / (np.sum(simu_0PE_signal_hist))
simu_4PE_signal_hist = simu_4PE_signal_hist[1000:len(bin_centers)+1000]

simu_5PE_signal_hist = np.convolve(simu_5PE_hist, simu_0PE_signal_hist) / (np.sum(simu_0PE_signal_hist))
simu_5PE_signal_hist = simu_5PE_signal_hist[1000:len(bin_centers)+1000]


simu_signal_hist = [n + a + b + c + d + e for n, a, b, c, d, e in zip(simu_0PE_signal_hist, simu_1PE_signal_hist, simu_2PE_signal_hist, simu_3PE_signal_hist, simu_4PE_signal_hist, simu_5PE_signal_hist)]



## 他のプログラムから呼び出された場合は表示しない ##
if __name__ == "__main__":

    ## サブプロットの作成 ##
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), facecolor='lightblue')
    fig.suptitle(f'simulation histograms', fontsize=16)

    axes[0, 0].bar(bin_centers, simu_noise_hist, width=bin_width, align='center', label='simu_PEND', alpha=0.5, color='black', ec='black')
    axes[0, 0].set_xlabel("Charge (mV * ns)", fontsize=12)
    axes[0, 0].set_ylabel("Entry", fontsize=12)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(0.8,500)
    axes[0, 0].set_xlim(-50,150)
    axes[0, 0].grid(True, linestyle='dotted')
    axes[0, 0].legend()

    axes[0, 1].bar(bin_centers, simu_signal_hist, width=bin_width, align='center', label='simu_PESD', alpha=0.5, color='orange', ec='orange')
    axes[0, 1].plot(bin_centers, simu_1PE_hist, linestyle='-', color='blue', label='simu_real_1PESD')
    axes[0, 1].set_xlabel("Charge (mV * ns)", fontsize=12)
    axes[0, 1].set_ylabel("Entry", fontsize=12)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylim(0.8,500)
    axes[0, 1].set_xlim(-50,150)
    axes[0, 1].grid(True, linestyle='dotted')
    axes[0, 1].legend()

    plt.tight_layout()
    plt.show()
