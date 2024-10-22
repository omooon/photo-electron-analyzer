import numpy as np
from .PhotoElectronGenerator import PhotoElectronGenerator

class OriginalMethod(PhotoElectronGenerator):
    def __init__(self, integ_on_list, integ_off_list, bin_params):
        super().__init__(integ_on_list, integ_off_list, bin_params)

    def run(self, maxpe, niterations):
        """
         Estimation of photoelectron signal distribution by takahashi.
         Analysis results are obtained by executing step5.
        
         Reference:
           M. Takahashi et al, A technique for estimating the absolute gain of a photomultiplier tube, Nuclear Inst. and Methods in Physics Research, A 894 (2018) 1–7, 2018.
        
         --- 2.2. Poisson distribution of the number of photoelectrons and 0 PESD ---
           step0:
        
         --- 2.3. Estimation of 1 PESD ---
           step1: Estimation of luminescent components
           step2: Estimation of 1 p.e. components
           step3: Estimation of n(>1) p.e. components
           step4: Re estimation of 1 p.e. components
           step5: iteration step2 to step 4 until convergence
        """
        print('Original Method Single Photo-Electron Analysis')

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

    def estimate_0pe_component(self):
        nume_alpha = np.sum(
            self.hist_on[0 : self._bin_zero - int(1/self._bin_width)]
        )
        deno_alpha = np.sum(
            self.hist_off[0 : self._bin_zero - int(1/self._bin_width)]
        )
        
        self._alpha = nume_alpha / deno_alpha
        self._alpha_err = np.sqrt(np.power(np.sqrt(nume_alpha) / deno_alpha, 2) + np.power((nume_alpha * np.sqrt(deno_alpha) / (deno_alpha * deno_alpha)), 2))
        self._hist_0pe = self.hist_off * self._alpha

    def estimate_lumin_component(self):
        # subtract 0 p.e. component from self.hist_on to estimate luminescence component
        # add the total number of entries from the bin with an entry of (-1 + bin_width) to the bin with an entry of (0 - bin_width) to bin0 and set all entries from the bin with an entry of -∞ to (bin0 - 1) to 0
        self._hist_lumin = self.hist_on - self._hist_0pe
        
        together_entry = 0
        together_bin_number = []
        for ibin in range(0, self._bin_zero):
            together_entry += self._hist_lumin[ibin]
            together_bin_number.append(ibin)
            self._hist_lumin[ibin] = 0
        if together_entry >= 0:
            self._hist_lumin[self._bin_zero] = together_entry

    def estimate_1pe_component(self):
        # estimation of the average number of photoelectrons from entry
        entries_on = np.sum(self.hist_on)                # calculate number of entries for self.hist_on
        entries_0pe = np.sum(self._hist_0pe)              # calculate number of entries for hist_0pe
        self._elambda = np.log(entries_on / entries_0pe)
        self._elambda_err = np.sqrt(entries_on * (1 - entries_0pe/entries_on) * entries_0pe/entries_on) / entries_0pe
        #elambda_err2 = alpha_err / alpha

        # estimation of 1 p.e. distribution
        entries_1pe = entries_0pe * self._elambda         # calculate number of entries for hist_1pe

        if hasattr(self, 'hist_1pe'):
            # hist_1pe 変数が存在する場合の処理
            sum_hist = self._hist_1pe
            self._hist_1pe = np.zeros(self._nbins)
        else:
            # hist_1pe 変数が存在しない場合の処理
            sum_hist = self._hist_lumin
            self._hist_1pe = np.zeros(self._nbins)
        
        total_entry = 0
        for ibin in range(self._bin_zero, self._nbins):
            partial_entry = sum_hist[ibin]
            total_entry += partial_entry
            if total_entry <= entries_1pe:
                self._hist_1pe[ibin] = partial_entry
            else:
                self._hist_1pe[ibin] = entries_1pe - (total_entry - partial_entry)
                break
            
        # estimation of the average number of photoelectrons from charge
        mean_on    = np.average(self.bin_centers, weights=self.hist_on)
        mean_off   = np.average(self.bin_centers, weights=self.hist_off)
        self.mean_1pe   = np.average(self.bin_centers, weights=self._hist_1pe)
        variance_1pe = np.average((self.bin_centers - self.mean_1pe)**2, weights=self._hist_1pe)
        self.mean_1pe_err = np.sqrt(variance_1pe) #標準偏差
        self._clambda = (mean_on - mean_off) / self.mean_1pe
        self._clambda_err = np.sqrt(pow(np.sqrt(mean_on-mean_off)/self.mean_1pe,2) + pow(((mean_on-mean_off)*np.sqrt(self.mean_1pe)/(self.mean_1pe*self.mean_1pe)), 2))
        
    def re_estimate_1pe_component(self, maxpe):
        # create hist_new_lumin by hist_subtraction from hist_org_lumin
        # add up all 2, 3, ... n p.e. distributions to hist_subtraction
        hist_subtraction = np.zeros(self.nbins)
        for i in range(2, maxpe+1):
            hist_subtraction += getattr(self, f'_hist_{i}pe')
        self._hist_1pe = self._hist_lumin - hist_subtraction

    def estimate_npes_component(self, maxpe, faster=False):
        # calculate number of entries for self.hist_on and hist_0pe
        entries_0pe = np.sum(self._hist_0pe)
        # estimating 2, 3, ..., n p.e. using hist_1pe as the kernel function
        # original calculation method
        if not faster:
            kernel = self._hist_1pe
            for i in range(1, maxpe):
            # WARNING: note that the distribution will not scale well unless ipe=1
                for j in range(i):
                    if j == 0:
                        input = self._hist_1pe
                    else:
                        input = convolved_hist_npe
                    convolved_hist_npe = np.convolve(input, kernel, mode='same')
                hist_npe = convolved_hist_npe / (np.math.factorial(i+1) * (entries_0pe ** i) )
            
                setattr(self, f'_hist_{i+1}pe', hist_npe)

        # quick calculation method
        #TODO スケールにバグがありそう
        #解析速度もそんなには変わらない
        if faster:
            kernel = self._hist_1pe
            for i in range(1, maxpe):  # WARNING: Note that the distribution will not scale well unless ipe=1
                if i == 1:
                    input = self._hist_1pe
                else:
                    input = convolved_hist_npe
                convolved_hist_npe = np.convolve(input, kernel, mode='same')
                hist_npe = convolved_hist_npe / ( (i+1) * entries_0pe )
                
                #setattr(self, f'_hist_{i+1}pe_{number}', hist_npe)
                setattr(self, f'_hist_{i+1}pe', hist_npe)