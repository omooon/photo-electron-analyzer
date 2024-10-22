import numpy as np
from .PhotoElectronGenerator import PhotoElectronGenerator

class DeconvolvedBlurredMethod(PhotoElectronGenerator):
    def __init__(self, integ_on_list, integ_off_list, bin_params):
        super().__init__(integ_on_list, integ_off_list, bin_params)
   
    def run(self, maxpe, niterations):
        print('Deconvolved Method Single Photo-Electron Analysis')

        self.deconvolved_on_to_off()         # new-step
        
        # iterationごとの名前　self.hist_{i}pe_{iteration}を辞書で保持
        results = {}

        self.estimate_0pe_component()
        self.estimate_lumin_component()      # step1

        for iteration in range(1, niterations + 1):
            self.estimate_1pe_component()         # step2
            self.estimate_npes_component(maxpe)   # step3

            #データの保存
            results[f"{iteration}iteration"] = {}
            results[f"{iteration}iteration"]["alpha"] = self._alpha
            results[f"{iteration}iteration"]["alpha_err"] = self._alpha_err
            results[f"{iteration}iteration"]["elambda"] = self._elambda
            results[f"{iteration}iteration"]["elambda_err"] = self._elambda_err
            results[f"{iteration}iteration"]["clambda"] = self._clambda
            results[f"{iteration}iteration"]["clambda_err"] = self._clambda_err
            for i in range(0, maxpe + 1):
                results[f"{iteration}iteration"][f"hist_{i}pe"] = getattr(self, f'_hist_{i}pe')

            self.re_estimate_1pe_component(maxpe) # step4
        return results
  
    def deconvolved_on_to_off(self):
        from skimage import restoration
        norm_hist_on  = self.hist_on / np.sum(self.hist_on)
        norm_hist_off = self.hist_off / np.sum(self.hist_off)
        rl_deconvolved_on = restoration.richardson_lucy(norm_hist_on, norm_hist_off)
        rl_deconvolved_off = restoration.richardson_lucy(norm_hist_off, norm_hist_off)

        self.hist_on  = rl_deconvolved_on * np.sum(self.hist_on)
        self.hist_off = rl_deconvolved_off * np.sum(self.hist_off)