import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import QTable, Column


class PhotoElectronGenerator:
    def __init__(self, integ_on_list, integ_off_list, bin_params):
        # bin_width,bin0がNoneの時は、親クラスによってデフォルトの値が設定される
        self._bin_width = bin_params.get("width", 0.16)
        self._bin_zero = bin_params.get("zero_position", 1500)
        eps = 1e-5 * self._bin_width
        min_value = -(self._bin_zero * self._bin_width + round(self._bin_width/2, 2))
        max_value = +(self._bin_zero * self._bin_width + round(self._bin_width/2, 2)) + eps
        self._bin_edges = np.arange(min_value, max_value, self._bin_width)
        self._bin_centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        self._nbins = len(self._bin_centers)

        self._hist_unit = integ_on_list.unit
        self._hist_on, _ = np.histogram(integ_on_list.value, bins=self._bin_edges)
        self._hist_off, _ = np.histogram(integ_off_list.value, bins=self._bin_edges)

    @property
    def bin_width(self):
        return self._bin_width
    @property
    def bin_zero(self):
        return self._bin_zero
    @property
    def bin_edges(self):
        return self._bin_edges
    @property
    def bin_centers(self):
        return self._bin_centers
    @property
    def nbins(self):
        return self._nbins

    @property
    def hist_on(self):
        return self._hist_on
    @property
    def hist_off(self):
        return self._hist_off

    def plot_hist_separate(self, hist_list):    
        if not isinstance(hist_list, list):
            self.logger.error("Input is not a list. Converting to list.")

        # リストの数に応じて適切なプロット数を決定
        n = len(hist_list)  # プロットするヒストグラムの数
        cols = 3  # 1行に何列表示するか
        rows = (n + cols - 1) // cols  # 必要な行数を計算

        fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))  # サブプロットを作成

        # `axs` が2次元配列になる場合に対応
        axs = axs.flatten() if n > 1 else [axs]  # 1つの場合もリストに変換

        # ヒストグラムをループで描画
        for i, histogram in enumerate(hist_list):
            axs[i].hist(
                self._bin_centers,
                bins=self._bin_edges,
                weights=histogram,
                alpha=0.75,
                label=f"Histogram of {histogram}"
            )               
            axs[i].set_title(f"Histogram of {histogram}")
            axs[i].set_xlabel(f"Charge ({self._hist_unit})")
            axs[i].set_ylabel("Entry (/bin)")
            axs[i].legend()
        fig.canvas.draw_idle()

        # 余分なサブプロットを非表示にする
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    def plot_hist_overlay(self, hist_list):
        # 入力がリスト形式でない場合、警告を出してリストに変換
        if not isinstance(hist_list, list):
            self.logger.warning("Input is not a list. Converting to list.")
            hist_list = [hist_list]
        
        fig, ax = plt.subplots(figsize=(8, 6))  # サブプロットを1つ作成

        # ヒストグラムを重ねて描画
        for i, histogram in enumerate(hist_list):
            ax.hist(
                self._bin_centers,
                bins=self._bin_edges,
                weights=histogram,
                alpha=0.5,  # 透明度を調整
                label=f"Histogram {i + 1}"  # ラベルをヒストごとに変更
            )
        
        # 軸ラベルやタイトルを設定
        ax.set_title("Overlaid Histograms")
        ax.set_xlabel(f"Charge ({self._hist_unit})")
        ax.set_ylabel("Entry (/bin)")
        ax.legend()  # 凡例を表示

        plt.tight_layout()
        plt.show()


    def save_results(self, iteration):

        if iteration is None:
        #--------------------
        # input parameterの保存
        #--------------------
        # niterations, maxpe, bin情報
            niterations_dict = Column(data=[self.niterations], name='niterations')
            maxpe_dict       = Column(data=[self.maxpe], name='maxpe')
            bin_width_dict   = Column(data=[self._bin_width], name='bin_width')
            bin0_dict        = Column(data=[self.bin0], name='bin0')

            columns = [
                niterations_dict,
                maxpe_dict,
                bin_width_dict,
                bin0_dict
            ]

            _input_parameter = QTable(columns)
            self._input_parameter = _input_parameter
        
        else:
        #--------------------
        # output parameterの保存
        #--------------------
            alpha_dict        = Column(data=[self.alpha], name='alpha')
            alpha_err_dict    = Column(data=[self.alpha_err], name='alpha_err')
            elambda_dict      = Column(data=[self.elambda], name='elambda')
            elambda_err_dict  = Column(data=[self.elambda_err], name='elambda_err')
            clambda_dict      = Column(data=[self.clambda], name='clambda')
            clambda_err_dict  = Column(data=[self.clambda_err], name='clambda_err')
            mean_1pe_dict     = Column(data=[self.mean_1pe], name='mean_1pe')
            mean_1pe_err_dict = Column(data=[self.mean_1pe_err], name='mean_1pe_err')

            columns = [
                alpha_dict,
                alpha_err_dict,
                elambda_dict,
                elambda_err_dict,
                clambda_dict,
                clambda_err_dict,
                mean_1pe_dict,
                mean_1pe_err_dict
            ]
            
            _output_parameter = QTable(columns)
            setattr(self, f'_output_parameter_{iteration}', _output_parameter)
            
        #--------------------
        # ヒストグラムの結果の保存
        #--------------------
        # Columnオブジェクトを作成
        # hist_pe_dictを作成してColumnオブジェクトに追加
            bin_centers_dict = Column(data=self._bin_centers, name='bin_centers')
            hist_on_dict     = Column(data=self.hist_on, name='hist_on')
            hist_off_dict    = Column(data=self.hist_off, name='hist_off')
            hist_lumin_dict  = Column(data=self.hist_lumin, name='hist_lumin')
            hist_pe_dict = {}
        
            columns = [
                bin_centers_dict,
                hist_on_dict,
                hist_off_dict,
                hist_lumin_dict
            ]
            for i in range(0, self.maxpe + 1):
                column_name = f'hist_{i}pe'
                column_data = getattr(self, f'hist_{i}pe')
                hist_pe_dict[column_name] = Column(data=column_data, name=column_name)
                columns.append(hist_pe_dict[column_name])
            # QTableオブジェクトを作成し、Columnを追加
            _hist_result = QTable(columns)
            setattr(self, f'_hist_result_{iteration}', _hist_result)
            
    def show_input(self):
        print('')
        print('Display input parameter')
        _input_parameter = self._input_parameter
        _input_parameter.pprint()
        
    def show_output(self, iteration=None):
        if iteration is None:
            iteration = self.niterations
            
        #どのiteration結果を表示しているのかを表示
        output = []
        for i in range(1, self.niterations+1):
            if i == iteration:
                output.append(f"\033[91m{i}*\033[0m")  # 一致する値を右側にマーク
            else:
                output.append(str(i))
                
        print('')
        print('Display *th output parameter (iteration number:', " ".join(output), ')')
        _output_parameter = getattr(self, f'_output_parameter_{iteration}')
        _output_parameter.pprint()
        
    def show_hist(self, iteration=None):
        #TODO: プロットは任意にしたい
        if iteration is None:
            iteration = self.niterations
            
        #どのiteration結果を表示しているのかを表示
        output = []
        for i in range(1, self.niterations+1):
            if i == iteration:
                output.append(f"\033[91m{i}*\033[0m")  # 一致する値を右側にマーク
            else:
                output.append(str(i))

        print('')
        print('Display *th iteration result (iteration number:', " ".join(output), ')')
        _hist_result = getattr(self, f'_hist_result_{iteration}')
        _hist_result.pprint(max_lines=10, max_width=-1)

        


"""
class OriginalMethod(PhotoElectronGenerator):

    def __init__(
        self,
        maxpe=3,
        niterations=5,
        integ_on_data=None,
        integ_off_data=None,
        hist_on=None,
        hist_off=None,
        bin_width=None,
        bin0=None,
    ):
        super().__init__(
            integ_on_data=integ_on_data,
            integ_off_data=integ_off_data,
            hist_on=hist_on,
            hist_off=hist_off,
            bin_width=bin_width,
            bin0=bin0,
        )
        
        self.maxpe = maxpe
        self.niterations = niterations

    def analysis(self):
        '''
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
        '''
        print('Original Method Single Photo-Electron Analysis')
        self.save_results(iteration=None)
        self.show_input()
        
        self.estimate_0pe_component()
        self.estimate_lumin_component()      # step1
        # step5
        for iteration in range(1, self.niterations+1):
            self.estimate_1pe_component()    # step2
            self.estimate_npes_component()   # step3
            self.re_estimate_1pe_component() # step4
            # iteration回目の結果を保存
            self.save_results(iteration)
        
        # iterationの最後の結果を表示
        self.show_output()
        self.show_hist()
        
    def simulation(self):
        print('Original Method Single Photo-Electron Analysis')
        self.save_results(iteration=None)
        self.show_input()
        
        self.estimate_0pe_component()
        self.estimate_lumin_component()      # step1
        # step5
        for iteration in range(1, self.niterations+1):
            self.estimate_1pe_component()    # step2
            self.estimate_npes_component()   # step3
            self.re_estimate_1pe_component() # step4
            # iteration回目の結果を保存
            self.save_results(iteration)
        
        # iterationの最後の結果を表示
        self.show_output()
        self.show_hist()

class BlurredMethod(PhotoElectronGenerator):

    def __init__(
        self,
        maxpe=3,
        niterations=5,
        integ_on_data=None,
        integ_off_data=None,
        hist_on=None,
        hist_off=None,
        bin_width=None,
        bin0=None,
    ):
        super().__init__(
            integ_on_data=integ_on_data,
            integ_off_data=integ_off_data,
            hist_on=hist_on,
            hist_off=hist_off,
            bin_width=bin_width,
            bin0=bin0,
        )
        
        self.maxpe = maxpe
        self.niterations = niterations
    
    def analysis(self):
        '''
         Estimation of photoelectron signal distribution by ihsida.
         Analysis results are obtained by executing ishida_step5.
        
         Reference:
           http://crportal.isee.nagoya-u.ac.jp:8090/pages/viewpage.action?pageId=25067919
           2020 年度　石田凜　「光電子増倍管の出力電荷分布の解析手法の改良」卒業論文_061700097_石田凛_210329.pdf
        
         takahashi-methodの改良
         ノイズ成分の畳み込みを追加（より正確な分布を作る、しかし、2p.e.以降の分布はノイズがn倍多くのってしまうためノイズが大きいとズレてくると考えられる）
         ishidaさんのを参考に手順を少し変えたやり方、高橋さんのをやった後に、新しく0 p.e.分布を作ってそれを使ってもう一回高橋さんのをやる
        '''

        self.estimate_0pe_component()
        self.estimate_lumin_component()      # step1
        # step5
        for iteration in range(1, self.niterations+1):
            self.estimate_1pe_component()    # step2
            self.estimate_npes_component()   # step3
            self.re_estimate_1pe_component() # step4
            #↑↑↑ここまでの解析はOriginalMethodと全く同じ
            
        #original-method で収束した hist_npe をhist_0peで畳み込んだ後、hist_onから引くことで、新しく0 p.e.分布を推定する
        self.re_estimate_0pe_component()     # new-step
        self.estimate_lumin_component()      # step1
        # step5
        for iteration in range(1, self.niterations+1):
            self.estimate_1pe_component()    # step2
            self.estimate_npes_component()   # step3
            self.re_estimate_1pe_component() # step4
            # iteration回目の結果を保存, 最後のiterationに関しては結果もprint
            self.save_results(iteration)
    
class DeconvolvedMethod(PhotoElectronGenerator):
    def __init__(
        self,
        maxpe=3,
        niterations=5,
        integ_on_data=None,
        integ_off_data=None,
        hist_on=None,
        hist_off=None,
        bin_width=None,
        bin0=None,
    ):
        super().__init__(
            integ_on_data=integ_on_data,
            integ_off_data=integ_off_data,
            hist_on=hist_on,
            hist_off=hist_off,
            bin_width=bin_width,
            bin0=bin0,
        )
        
        self.maxpe = maxpe
        self.niterations = niterations
   
    def analysis(self):
        self.deconvolved_on_to_off()         # new-step
        
        self.estimate_0pe_component()
        self.estimate_lumin_component()      # step1
        # step5
        for iteration in range(1, self.niterations+1):
            self.estimate_1pe_component()    # step2
            self.estimate_npes_component()   # step3
            self.re_estimate_1pe_component() # step4
            # iteration回目の結果を保存, 最後のiterationに関しては結果もprint
            self.save_results(iteration)

class DeconvolvedBlurredMethod(PhotoElectronGenerator):
    def __init__(
        self,
        maxpe=3,
        niterations=5,
        integ_on_data=None,
        integ_off_data=None,
        hist_on=None,
        hist_off=None,
        bin_width=None,
        bin0=None,
    ):
        super().__init__(
            integ_on_data=integ_on_data,
            integ_off_data=integ_off_data,
            hist_on=hist_on,
            hist_off=hist_off,
            bin_width=bin_width,
            bin0=bin0,
        )
        
        self.maxpe = maxpe
        self.niterations = niterations
   
    def analysis(self):
        self.deconvolved_on_to_off()         # new-step
        
        self.estimate_0pe_component()
        self.estimate_lumin_component()      # step1
        # step5
        for iteration in range(1, self.niterations+1):
            self.estimate_1pe_component()    # step2
            self.estimate_npes_component()   # step3
            self.re_estimate_1pe_component() # step4
            
        #original-method で収束した hist_npe をhist_0peで畳み込んだ後、hist_onから引くことで、新しく0 p.e.分布を推定する
        self.re_estimate_0pe_component()     # new-step
        self.estimate_lumin_component()      # step1
        # step5
        for iteration in range(1, self.niterations+1):
            self.estimate_1pe_component()    # step2
            self.estimate_npes_component()   # step3
            self.re_estimate_1pe_component() # step4
            # iteration回目の結果を保存, 最後のiterationに関しては結果もprint
            self.save_results(iteration)
            
def main():
    import matplotlib.pyplot as plt
    from PhotoElectronGenerator import OriginalMethod
    
    hist_on = np.load('/Users/omooon/lab309/test_data/Test_Charge_HistData_On.npy')
    hist_off = np.load('/Users/omooon/lab309/test_data/Test_Charge_HistData_Off.npy')
    '''
    org = OriginalMethod(
        hist_on=hist_on,
        hist_off=hist_off,
    )
    org.analysis()
    #org.hist_charge()
    #plt.show()
    
    '''
    blu = BlurredMethod(
        hist_on=hist_on,
        hist_off=hist_off,
    )
    blu.analysis()
    #blu.hist_charge()

    dec = DeconvolvedBlurredMethod(
        hist_on=hist_on,
        hist_off=hist_off,
    )
    dec.analysis()
    dec.hist_charge(iteration=1)
    plt.show()
    
    '''
    decblu = DeconvolvedBlurredMethod(
        hist_on=hist_on,
        hist_off=hist_off,
    )
    decblu.analysis()
    
    org.plot_test()
    
    
    plt.hist(org.bin_centers, bins=org.bin_edges, weights=org.hist_on, label='org_hist_on')
    plt.hist(dec.bin_centers, bins=dec.bin_edges, weights=dec.hist_on, label='dec_hist_on', alpha=0.8)
    plt.hist(dec.bin_centers, bins=dec.bin_edges, weights=dec.hist_1pe, label='dec_hist_on', alpha=0.5)
    #plt.plot(blu.bin_centers, blu.hist_1pe, label='blu_hist_1pe')
    #plt.yscale('log')
    plt.legend()
    plt.show()
    '''

if __name__ == '__main__':
    main()
"""