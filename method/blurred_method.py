import numpy as np
from .PhotoElectronGenerator import PhotoElectronGenerator

class BlurredMethod(PhotoElectronGenerator):
    def __init__(self, integ_on_list, integ_off_list, bin_params):
        super().__init__(integ_on_list, integ_off_list, bin_params)
    
    def run(self, maxpe, niterations):
        """
         Estimation of photoelectron signal distribution by ihsida.
         Analysis results are obtained by executing ishida_step5.
        
         Reference:
           http://crportal.isee.nagoya-u.ac.jp:8090/pages/viewpage.action?pageId=25067919
           2020 年度 石田凜 「光電子増倍管の出力電荷分布の解析手法の改良」卒業論文_061700097_石田凛_210329.pdf
        
         takahashi-methodの改良
         ノイズ成分の畳み込みを追加（より正確な分布を作る、しかし、2p.e.以降の分布はノイズがn倍多くのってしまうためノイズが大きいとズレてくると考えられる）
         ishidaさんのを参考に手順を少し変えたやり方、高橋さんのをやった後に、新しく0 p.e.分布を作ってそれを使ってもう一回高橋さんのをやる
        """
        print('Blurred Method Single Photo-Electron Analysis')

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
        # ↑↑↑↑↑↑↑↑↑ ここまでの解析はOriginalMethodと全く同じ ↑↑↑↑↑↑↑↑↑
            

        # ↓↓↓↓↓↓↓↓↓ BlurredMethod独自の新しいステップの追加 ↓↓↓↓↓↓↓↓↓
        #original-method で収束した hist_npe をhist_0peで畳み込んだ後、hist_onから引くことで、新しく0 p.e.分布を推定する
        self.re_estimate_0pe_component(maxpe)  # new-step
        self.estimate_lumin_component()        # step1

        for iteration in range(1, self.niterations+1):
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
    
    
    def re_estimate_0pe_component(self, maxpe):
        entries_0pe = np.sum(self._hist_0pe)
         # カーネル関数の設定（0 p.e.成分の分布）
        kernel = self._hist_0pe
        for i in range(1, maxpe + 1):
            #input = selfs.iteration[number]['hist'][f'{i}pe']
            input = getattr(self, f'_hist_{i}pe')
            hist_blurred_npe = np.convolve(input, kernel, 'same') / (entries_0pe)
            setattr(self, f'_hist_blurred_{i}pe', hist_blurred_npe)

        hist_substraction = np.zeros(self._nbins)
        for i in range(1, maxpe + 1):
            #hist_substraction += self.iteration[number]['hist'][f'blured_{i}pe']
            hist_substraction += getattr(self, f'_hist_blurred_{i}pe')
        hist_non_lumin = self.hist_on - hist_substraction

        #re_zeroPE_histは、スケールとしてはさっきより正しいが、引き算によって変な成分も増えたので、alphaのスケールだけをこれを使って抽出、その後、そのalphaを使って元のzeroPE_histからスケールするという流れにすればいいはず
        nume_new_alpha = sum(hist_non_lumin[0 : self._bin_zero - int(1 / self._bin_width)])
        deno_new_alpha = sum(self.hist_off[0 : self._bin_zero - int(1 / self._bin_width)])
        self._alpha = nume_new_alpha / deno_new_alpha
        self._alpha_err = np.sqrt(pow(np.sqrt(nume_new_alpha)/deno_new_alpha,2) + pow((nume_new_alpha*np.sqrt(deno_new_alpha)/(deno_new_alpha*deno_new_alpha)), 2))
        self._hist_0pe = self.hist_off * self._alpha