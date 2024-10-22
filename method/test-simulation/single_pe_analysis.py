import click
import yaml

#import glob
#import pandas as pd
#from tabulate import tabulate

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, Button

import astropy.units as u
from scripts.WaveFormGenerator import WaveFormGenerator
from scripts.PhotoElectronGenerator import PhotoElectronGenerator
from scripts.HistgramPlotter import Hist1D
#pyrootも使えるようにしたい

@click.command()
@click.option(
    '--raw_data_file_name', '-f',
    type=str,
    default='./test_data/Test_Led_WaveData_CH1.txt.gz',
    help='Path of the waveform file'
)
@click.option(
    '--parameters_file_name', '-p',
    type=str,
    default='./test_data/Test_Parameters_LED.yaml',
    help='Path of the parameters yaml file'
)


def main(raw_data_file_name, parameters_file_name):

    parameters   = yaml.safe_load(open(parameters_file_name))
    VOLT_UNIT = u.Unit(parameters['unit']['volt'])
    TIME_UNIT = u.Unit(parameters['unit']['time'])
    
    
    def original_method(integ_on_data, integ_off_data, parameters):
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
        niterations = parameters['iteration']['niterations']
    
        spe = PhotoElectronGenerator(integ_on_data, integ_off_data, parameters)
        for i in range(1, niterations+1):
            spe.estimate_0pe(i)
            if i == 1:
                spe.estimate_lumin(i, base=True)
            spe.estimate_1pe(i)
            spe.estimate_npes(i)
            if i != niterations:
                spe.estimate_lumin(i, remake=True)

        result = spe.get_result(niterations)
        hist, bin_info = spe.get_hist(niterations)
        return result, hist, bin_info
    
    
    
    def blurred_method(integ_on_data, integ_off_data, parameters):
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
        niterations = parameters['iteration']['niterations']
    
        spe = PhotoElectronGenerator(integ_on_data, integ_off_data, parameters)
        for k in range(0, 2):
            if k == 0:
                for i in range(1, niterations+1):
                    spe.estimate_0pe(i)
                    if i == 1:
                        spe.estimate_lumin(i, base=True)
                    spe.estimate_1pe(i)
                    spe.estimate_npes(i)
                    if i != niterations:
                        spe.estimate_lumin(i, remake=True)
                # 次に takahashi-method で収束した hist_npe をhist_0peで畳み込んだ後、hist_onから引くことで、新しく0 p.e.分布を推定する
                spe.estimate_npes(niterations, blurred=True)
                    
            else:
                for i in range(niterations+1, 2*niterations+1):
                    
                    if i == niterations+1:
                        spe.estimate_0pe(i, remake=True)
                    if i > niterations+1:
                        spe.estimate_0pe(i, analysis_pass=True)

                    if i == niterations+1:
                        spe.estimate_lumin(i, base=True)
                    spe.estimate_1pe(i)
                    spe.estimate_npes(i)
                    if i != 2*niterations:
                        spe.estimate_lumin(i, remake=True)
                        
        '''
        import matplotlib.pyplot as plt
        plt.hist(spe.bin_centers, bins=spe.bin_edges, weights=spe.hist_on, label='hist_on')
        plt.hist(spe.bin_centers, bins=spe.bin_edges, weights=spe.hist_0pe_5, label='hist_0pe_10')
        plt.hist(spe.bin_centers, bins=spe.bin_edges, weights=spe.hist_1pe_5, label='hist_1pe_10')
        plt.plot(spe.bin_centers, spe.hist_2pe_5, label='hist_2pe_10')
        plt.plot(spe.bin_centers, spe.hist_3pe_5, label='hist_3pe_10')
        plt.plot(spe.bin_centers, spe.hist_4pe_5, label='hist_4pe_10')
        plt.plot(spe.bin_centers, spe.hist_5pe_5, label='hist_5pe_10')
        plt.yscale('log')
        plt.legend()
        #plt.ylim(bottom=1)
        plt.ylim(1, 1e4)
        plt.show()
        '''
        result = spe.get_result(2*niterations)
        hist, bin_info = spe.get_hist(2*niterations)
        return result, hist, bin_info
    
    
        
    def deconvolution_method(integ_on_data, integ_off_data, parameters):
        niterations = parameters['iteration']['niterations']
    
        spe = PhotoElectronGenerator(integ_on_data, integ_off_data, parameters)
        
        spe.deconvolution()
        for i in range(1, niterations+1):
            spe.estimate_0pe(i)
            if i == 1:
                spe.estimate_lumin(i, base=True)
            spe.estimate_1pe(i)
            spe.estimate_npes(i)
            if i != niterations:
                spe.estimate_lumin(i, remake=True)
        
        import matplotlib.pyplot as plt
        plt.hist(spe.bin_centers, bins=spe.bin_edges, weights=spe.hist_on, label='hist_on')
        plt.hist(spe.bin_centers, bins=spe.bin_edges, weights=spe.hist_0pe_5, label='hist_0pe_5')
        plt.hist(spe.bin_centers, bins=spe.bin_edges, weights=spe.hist_1pe_5, label='hist_1pe_5')
        plt.plot(spe.bin_centers, spe.hist_2pe_5, label='hist_2pe_5')
        plt.plot(spe.bin_centers, spe.hist_3pe_5, label='hist_3pe_5')
        plt.plot(spe.bin_centers, spe.hist_4pe_5, label='hist_4pe_5')
        plt.plot(spe.bin_centers, spe.hist_5pe_5, label='hist_5pe_5')
        plt.yscale('log')
        plt.legend()
        #plt.ylim(bottom=1)
        plt.ylim(1, 1e4)
        plt.show()
        
    # WaveFormGeneratorのインスタンスの作成、積分時間に対する波形の計算データ取得
    wave = WaveFormGenerator(raw_data_file_name, parameters)
    str_on  = parameters['integration']['start_on'] * TIME_UNIT
    end_on  = parameters['integration']['end_on'] * TIME_UNIT
    str_off = str_on - parameters['integration']['off_minus'] * TIME_UNIT
    end_off = end_on - parameters['integration']['off_minus'] * TIME_UNIT
    integ_on_data  = wave.integration(str_on, end_on)
    integ_off_data = wave.integration(str_off, end_off)
    
    # single p.e.分布の取得
    org_result, org_hist, bin_info = original_method(integ_on_data, integ_off_data, parameters)
    blu_result, blu_hist, _ = blurred_method(integ_on_data, integ_off_data, parameters)
    #deconvolution_method(integ_on_data, integ_off_data, parameters)





    # Obtaining information for 1-dimensional histograms
    h1d = Hist1D(
        parameters['iteration']['bin_width'],
        parameters['iteration']['bin_zero']
    )
    
    # Plot settings
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Plotting initial data
    iteration_key = 'iteration_1'
    hist_list = []
    for j in range(2):
        pe_key = f'{j}pe_1'
        hist, _, _ = ax.hist(
            h1d.bin_centers,
            bins=h1d.bin_edges,
            weights=org_hist[iteration_key][pe_key],
            alpha=0.75,
            label=f'{pe_key}'
            )
        hist_list.append(hist)
    ax.set_title(f'{iteration_key}')
    ax.set_xlabel('Charge [mV ns]')
    ax.set_ylabel('Entry [/bin]')
    ax.legend()

    # Slider settings
    ax_hist_iteration_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    hist_iteration_slider = Slider(
        ax_hist_iteration_slider,
        'Iteration',
        1, 5,  # 1から5の間でスライダーを動かす
        valinit=1,
        valstep=1
    )
    
    # Reset button settings
    ax_hist_reset_button = plt.axes([0.8, 0.025, 0.1, 0.04])
    hist_reset_button = Button(
        ax_hist_reset_button,
        'Reset',
        color='lightgoldenrodyellow',
        hovercolor='0.975'
    )
 
    # Callback function when the slider is changed
    def show_update_hist(val):
        iteration = int(hist_iteration_slider.val)
        iteration_key = f'iteration_{iteration}'
        ax.clear()
        hist_list = []
        for j in range(2):
            pe_key = f'{j}pe_{iteration}'
            hist, _, _ = ax.hist(
                h1d.bin_centers,
                bins=h1d.bin_edges,
                weights=org_hist[iteration_key][pe_key],
                alpha=0.75,
                label=f'{pe_key}'
                )
            hist_list.append(hist)
        ax.set_title(f'{iteration_key}')
        ax.set_xlabel('Charge [mV ns]')
        ax.set_ylabel('Entry [/bin]')
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        #ymax = max(ax.get_ylim()[1], np.max(yvalues.value)*2.0)
        #ymin = min(ax.get_ylim()[0], np.min(yvalues.value)*0.5)
        #ax.set_ylim(max(1E12, ymin), ymax)
        ax.legend()
        fig.canvas.draw_idle()

    # Callback function when the reset button is pressed
    def reset_hist_plot(event):
        hist_iteration_slider.reset()

    # Set callback functions for sliders and buttons
    hist_iteration_slider.on_changed(show_update_hist)
    hist_reset_button.on_clicked(reset_hist_plot)

    plt.show()

if __name__ == '__main__':
    main()
