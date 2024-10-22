import click
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, Button

import astropy.units as u

from logging import getLogger, StreamHandler
from waveform.WaveFormGenerator import WaveFormGenerator
from method.PhotoElectronGenerator import PhotoElectronGenerator
from method.original_method import OriginalMethod

class PhotoElectronInitialize:
    def __init__(
        self, 
        raw_data = '/Users/omooon/lab309/photo-electron-analyzer/data/Test_Led_WaveData_CH1.txt.gz',
        light_params={"source": "LED",
                      "flash_interval": 500,
                      "detector": "PMT"}, 
        integ_params={"start_on": 420,  # ns 420,390
                      "stop_on": 450,  # ns 450,420
                      "off_minus": 100}, # ns 
        bin_params={"bin_width": 16,
                    "bin_zero": 1500}, 
        loglevel="INFO"
    ):
        self.logger = getLogger(__name__)
        handler = StreamHandler()
        handler.setLevel(loglevel)
        self.logger.addHandler(handler)
        self.logger.setLevel(loglevel)

        self.VOLT_UNIT = u.Unit("mV")
        self.TIME_UNIT = u.Unit("ns")

        if light_params["source"] == "LED":
            light_params["flash_lag"] = 0.0
        elif light_params["source"] == "Laser":           
            light_params["flash_lag"] = 0.07

        # WaveFormGeneratorのインスタンスの作成、積分時間に対する波形の計算データ取得
        self.waveform = WaveFormGenerator(raw_data, light_params, self.VOLT_UNIT, self.TIME_UNIT)
        
        tstart_on = integ_params['start_on'] * self.TIME_UNIT
        tstop_on = integ_params['stop_on'] * self.TIME_UNIT
        tstart_off = (integ_params['start_on'] - integ_params['off_minus']) * self.TIME_UNIT
        tstop_off = (integ_params['stop_on'] - integ_params['off_minus']) * self.TIME_UNIT

        self.integ_on_list  = self.waveform.time_integration(tstart_on, tstop_on)
        self.integ_off_list = self.waveform.time_integration(tstart_off, tstop_off)

        self.bin_params = bin_params

    def run_method(self, method, maxpe=3, niterations=5):
        if method == "original":
            self.org = OriginalMethod(self.integ_on_list, self.integ_off_list, self.bin_params)
            self.results = self.org.run(maxpe, niterations)
        if method == "blurred":
            #self.blu = BlurredMethod(bin_params)
            #self.blu.run()
            pass
        if method == "deconvolved":
            #self.deconv = DeconvolvedMethod(bin_params)
            #self.dec.run()
            pass
