import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import QTable, Column

from HistgramPlotter import Hist1D
from PhotoElectronGenerator import PhotoElectron

class SimulationCharge(Hist1D):
    def __init__(
        self,
        maxpe=3,
        niterations=5,
        nentries=10000,
        bin_width=1,
        bin0=150,
    ):
        super().__init__(
            bin_width=bin_width,
            bin0=bin0,
        )
        self.maxpe = maxpe
        self.niterations = niterations
        self.nentries = nentries

    def _poisson_probability(self,k, λ=0.3):
        import math
        if k == self.maxpe:
            p = 0
            for i in range(0, self.maxpe):
                p += (np.float_power(λ, i) * np.exp(-λ) / math.factorial(i))
            probability = 1 - p
        else:
            probability = (np.float_power(λ, k) * np.exp(-λ) / math.factorial(k))
        return probability

    def _delta_hist(self, pos=None):
        if pos is None:
            pos = self.bin0
        else:
            pos = self.bin0 + (pos*self.bin_width)
        simu_hist = np.zeros(self.nbins)
        simu_hist[pos] += self.nentries
        return simu_hist
        
    def _gaussian_hist(self, mu, sigma, density=False):
        simu_data = np.random.normal(mu, sigma, self.nentries)
        simu_hist, _ = np.histogram(simu_data, bins=self.bin_edges, density=density)
        return simu_hist

    def show_simu_hist(self):
        #TODO: プロットは任意にしたい
        print('')
        print('Display *th iteration result (iteration number:', " ".join(output), ')')
        self.true_simu_hist_result.pprint(max_lines=10, max_width=-1)

    def get_simu_hist_dict(self):
        return self._simu_hist_result

    def IdealGaussian(self):
        ###########
        # True分布 #
        ###########
        #真off分布
        self.true_hist_off = self._delta_hist(pos=0)

        #真npes分布
        self.true_hist_0pe = self._delta_hist(pos=0)\
                                    * self._poisson_probability(k=0)
        self.true_hist_1pe = self._delta_hist(pos=20)\
                                    * self._poisson_probability(k=1)
        entries_0pe = np.sum(self.true_hist_0pe)
        kernel = self.true_hist_1pe
        for i in range(1, self.maxpe):
        # WARNING: note that the distribution will not scale well unless ipe=1
            for j in range(i):
                if j == 0:
                    input = self.true_hist_1pe
                else:
                    input = convolved_hist_npe
                convolved_hist_npe = np.convolve(input, kernel, mode='same')
            true_hist_npe = convolved_hist_npe / (np.math.factorial(i+1) * (entries_0pe ** i) )
            setattr(self, f'true_hist_{i+1}pe', true_hist_npe)

        #真on分布
        self.true_hist_on = np.zeros(self.nbins)
        for i in range(0, self.maxpe+1):
            self.true_hist_on += getattr(self, f'true_hist_{i}pe')
        
        ##############
        # Blurred分布 #
        ##############
        noise_hist = self._gaussian_hist(mu=0, sigma=2)
        entries_noise = np.sum(noise_hist)

        #観測off分布
        kernel = noise_hist
        input = self.true_hist_off
        self.obs_hist_off =np.convolve(input, kernel, 'same') / (entries_noise)

        #観測npes分布
        kernel = noise_hist
        for i in range(0, self.maxpe+1):
            input = getattr(self, f'true_hist_{i}pe')
            hist_blurred_npe = np.convolve(input, kernel, 'same') / (entries_noise)
            setattr(self, f'obs_hist_{i}pe', hist_blurred_npe)

        #観測on分布
        kernel = noise_hist
        input = self.true_hist_on
        self.obs_hist_on = np.convolve(input, kernel, 'same') / (entries_noise)

    def Gaussian(self): #ポアソン分布に従ったカウント数でガウシアン分布
        ###########
        # True分布 #
        ###########
        #真off分布
        self.true_hist_off = self._delta_hist(pos=0)
        
        #真npes分布
        self.true_hist_0pe = self._delta_hist(pos=0)\
                                    * self._poisson_probability(k=0)
        self.true_hist_1pe = self._gaussian_hist(mu=20, sigma=8)\
                                    * self._poisson_probability(k=1)
        
        entries_0pe = np.sum(self.true_hist_0pe)
        kernel = self.true_hist_1pe
        for i in range(1, self.maxpe):
        # WARNING: note that the distribution will not scale well unless ipe=1
            for j in range(i):
                if j == 0:
                    input = self.true_hist_1pe
                else:
                    input = convolved_hist_npe
                convolved_hist_npe = np.convolve(input, kernel, mode='same')
            true_hist_npe = convolved_hist_npe / (np.math.factorial(i+1) * (entries_0pe ** i) )
            setattr(self, f'true_hist_{i+1}pe', true_hist_npe)

        #真on分布
        self.true_hist_on = np.zeros(self.nbins)
        for i in range(0, self.maxpe+1):
            self.true_hist_on += getattr(self, f'true_hist_{i}pe')
        
        ##############
        # Blurred分布 #
        ##############
        noise_hist = self._gaussian_hist(mu=0, sigma=5)
        entries_noise = np.sum(noise_hist)
        
        #観測off分布
        kernel = noise_hist
        input = self.true_hist_off
        self.obs_hist_off =np.convolve(input, kernel, 'same') / (entries_noise)

        #観測npes分布
        kernel = noise_hist
        for i in range(0, self.maxpe+1):
            input = getattr(self, f'true_hist_{i}pe')
            hist_blurred_npe = np.convolve(input, kernel, 'same') / (entries_noise)
            setattr(self, f'obs_hist_{i}pe', hist_blurred_npe)

        #観測on分布
        #self.simu_hist_on = np.zeros(self.nbins)
        #for i in range(0, self.maxpe+1):
        #    self.simu_hist_on += getattr(self, f'simu_hist_{i}pe')
        kernel = noise_hist
        input = self.true_hist_on
        self.obs_hist_on = np.convolve(input, kernel, 'same') / (entries_noise)
        

        #plt.hist(self.bin_centers, bins=self.bin_edges, weights=self.simu_hist_off, label=f'hist_off {np.sum(self.simu_hist_off)}', alpha=0.5)
        #plt.hist(self.bin_centers, bins=self.bin_edges, weights=self.true_simu_hist_on, label=f'hist_on {np.sum(self.true_simu_hist_on)}', alpha=0.5)
        #plt.hist(self.bin_centers, bins=self.bin_edges, weights=self.simu_hist_on, label=f'hist_on {np.sum(self.simu_hist_on)}', alpha=0.5)
        #plt.hist(self.bin_centers, bins=self.bin_edges, weights=self.simu_hist_1pe, label=f'hist_1pe {np.sum(self.simu_hist_1pe)}', alpha=0.5)
        #plt.hist(self.bin_centers, bins=self.bin_edges, weights=self.simu_hist_2pe, label=f'hist_2pe {np.sum(self.simu_hist_2pe)}', alpha=0.5)
        #plt.hist(self.bin_centers, bins=self.bin_edges, weights=self.true_simu_hist_1pe, label='hist_off', alpha=0.5)
        #plt.legend()
        #plt.show()

    def BimodalGaussian(self):
        #ポアソン分布に従ったカウント数で1peが二峰ガウシアン分布
        pass

def main():
    from SimulationChargeGenerator import SimulationCharge
    from PhotoElectronGenerator import OriginalMethod
    from PhotoElectronGenerator import BlurredMethod
    from PhotoElectronGenerator import DeconvolvedMethod
    
    sim = SimulationCharge(maxpe=10)
    sim.Gaussian()
    '''
    org = OriginalMethod(
        maxpe=3,
        niterations=5,
        hist_on=sim.simu_hist_on,
        hist_off=sim.simu_hist_off,
        bin_width=sim.bin_width,
        bin0=sim.bin0,
    )
    org.analysis()
    
    org.hist_charge()
    plt.show()
    
    blu = BlurredMethod(
        maxpe=3,
        niterations=5,
        hist_on=sim.simu_hist_on,
        hist_off=sim.simu_hist_off,
        bin_width=sim.bin_width,
        bin0=sim.bin0,
    )
    blu.analysis()
    blu.hist_charge()
    plt.show()
    
    dec = DeconvolvedMethod(
        maxpe=3,
        niterations=5,
        hist_on=sim.simu_hist_on,
        hist_off=sim.simu_hist_off,
        bin_width=sim.bin_width,
        bin0=sim.bin0,
    )
    dec.analysis()
    dec.hist_charge(iteration=1)
    plt.show()
    '''

    #sim.IdealGaussian()
    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot2grid((1, 3), (0, 0))
    ax2 = plt.subplot2grid((1, 3), (0, 1))
    ax3 = plt.subplot2grid((1, 3), (0, 2))

    a=sim.obs_hist_off/sim.nentries
    from skimage import restoration
    true = sim.true_hist_on/sim.nentries
    obs = sim.obs_hist_on/sim.nentries
    #dec = restoration.wiener(obs, a, balance=0.3)
    dec = restoration.richardson_lucy(obs, a)

    ax1.hist(
        sim.bin_centers,
        bins=sim.bin_edges,
        weights=true,
        alpha=0.75,
        label='true on'
    )
    ax2.hist(
        sim.bin_centers,
        bins=sim.bin_edges,
        weights=obs,
        alpha=0.75,
        label='observed on'
    )
    ax3.hist(
        sim.bin_centers,
        bins=sim.bin_edges,
        weights=true,
        alpha=1,
        label='true on'
    )
    ax3.hist(
        sim.bin_centers,
        bins=sim.bin_edges,
        weights=dec,
        alpha=0.75,
        label='deconvolved on'
    )
    
    '''
    for pe in range(0, self.maxpe+1):
        if pe < 2:
            ax2.hist(
                self.bin_centers,
                bins=self.bin_edges,
                weights=getattr(self, f'_hist_result_{iteration}')[f'hist_{pe}pe'],
                alpha=0.75,
                label=f'{pe}p.e.'
            )
        else:
            ax2.plot(
                self.bin_centers,
                getattr(self, f'_hist_result_{iteration}')[f'hist_{pe}pe'],
                alpha=0.75,
                label=f'{pe}p.e.'
            )
            
    for pe in range(0, self.maxpe+1):
        if pe < 2:
            ax3.hist(
                self.bin_centers,
                bins=self.bin_edges,
                weights=getattr(self, f'_hist_result_{iteration}')[f'hist_{pe}pe'],
                alpha=0.75,
                label=f'{pe}p.e.'
            )
        else:
            ax3.plot(
                self.bin_centers,
                getattr(self, f'_hist_result_{iteration}')[f'hist_{pe}pe'],
                alpha=0.75,
                label=f'{pe}p.e.'
            )
        #ymax = max(ax.get_ylim()[1], np.max(yvalues.value)*2.0)
        #ymin = min(ax.get_ylim()[0], np.min(yvalues.value)*0.5)
        #ymax = max(ax3.get_ylim()[1])
        ymin = 10e-2
        ax3.set_ylim(ymin, )
        ax3.set_yscale('log')
    '''

    ax1.legend()
    ax2.legend()
    ax3.legend()
    #ax3.set_yscale('log')
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
