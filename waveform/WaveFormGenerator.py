import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import gzip
import struct
import enum
import numpy as np
from . import isee_sipm


class WaveFormGenerator:
    def __init__(self, raw_data, light_params, vunit, tunit):                
        self._vunit = vunit
        self._tunit = tunit

        self._flash_interval = light_params["flash_interval"] * self._tunit
        if light_params["source"] == "LED":
            self._flash_lag = light_params["flash_lag"] * self._tunit
        elif light_params["source"] == "Laser":
            self._flash_lag = light_params["flash_lag"] * self._tunit

        self._waveform_list = isee_sipm.gzip2waveforms(raw_data)
        self._volt_res = (self._waveform_list[0].get_dv()*u.V).to(self._vunit)
        self._time_res = (self._waveform_list[0].get_dt()*u.s).to(self._tunit).round(1)



    def get_raw_waveform(self):
        wave_npts = self._waveform_list[0].get_n()
        volt_data_list = (self._waveform_list[0].yarray[0 : wave_npts]*u.V).to(self._vunit)  # voltage value list per waveform (V)
        time_data_list = (self._waveform_list[0].xarray[0 : wave_npts]*u.s).to(self._tunit)  # time value list per waveform (s)
        return time_data_list, volt_data_list # x軸、y軸
    
    def plot_raw_waveform(self):
        time_data_list, volt_data_list = self.get_raw_waveform()

        plt.plot(time_data_list, volt_data_list, marker='o')
        #plt.title(f'{self.file_name}')
        plt.xlabel(f'Time ({self._tunit})')
        plt.ylabel(f'Voltage ({self._vunit})')
        plt.show()



    def get_average_waveform(self, norm=True):
        nwaves    = len(self._waveform_list)
        wave_npts = self._waveform_list[0].get_n()
        slice_npts = int(self._flash_interval/self._time_res)
        nslices    = int(wave_npts/slice_npts)

        x_stack = np.arange(slice_npts) * self._time_res
        y_stack = np.zeros(slice_npts) * self._vunit
        
        x_norm = np.arange(slice_npts) * self._time_res
        y_norm = np.array([])
        
        for iwave in range(0, nwaves):
            # Reset to 0 after each analysis of a certain waveform data
            sumtime_correction = 0.   # Accumulation time of self._flash_lag when analyzing each waveform slice
            shift_npt = 0            # Shift the slice start point by shift_npt
            for islice in range(0, nslices):
                sumtime_correction += self._flash_lag
                if sumtime_correction > self._time_res * (shift_npt + 1):
                    shift_npt += 1
                    
                slice_start_pt = islice * slice_npts - shift_npt        # Start point for each slice
                slice_end_pt   = (islice + 1) * slice_npts - shift_npt  # End point for each slice

                baseline_volt = np.mean(self._waveform_list[iwave].yarray[slice_start_pt : slice_start_pt + int(100/self._time_res.value)] * u.V)
                slice_volt    = self._waveform_list[iwave].yarray[slice_start_pt : slice_end_pt] * u.V - baseline_volt
                slice_volt    = slice_volt.to(self._vunit)

                y_stack += slice_volt
        y_norm = y_stack / (nwaves * nslices)

        if norm:
            return x_norm, y_norm
        else:
            return x_stack, y_stack

    def plot_average_waveform(self, norm=True):
        if norm:
            x_norm, y_norm = self.get_average_waveform(norm=True)

            plt.plot(x_norm, y_norm, marker='o')
            #plt.title(f'{self.file_name}')
            plt.xlabel(f'time ({self._tunit})')
            plt.ylabel(f'stacked voltage ({self._vunit})')
            plt.show()
        else:
            x_stack, y_stack = self.get_average_waveform(norm=False)

            plt.plot(x_stack, y_stack, marker='o')
            #plt.title(f'{self.file_name}')
            plt.xlabel(f'time ({self._tunit})')
            plt.ylabel(f'average voltage ({self._vunit})')
            plt.show()



    def get_peak_position(self):
        nwaves    = len(self._waveform_list)
        wave_npts = self._waveform_list[0].get_n()
        slice_npts = int(self._flash_interval/self._time_res)
        nslices    = int(wave_npts/slice_npts)
        
        x_peak = np.arange(slice_npts) * self._time_res
        y_peak = np.zeros(slice_npts)
        
        for iwave in range(0, nwaves):
            # Reset to 0 after each analysis of a certain waveform data
            sumtime_correction = 0.   # Accumulation time of self._flash_lag when analyzing each waveform slice
            shift_npt = 0            # Shift the slice start point by shift_npt
            
            for islice in range(0, nslices):
                sumtime_correction += self._flash_lag
                if sumtime_correction > self._time_res * (shift_npt + 1):
                    shift_npt += 1
                    
                slice_start_pt = islice * slice_npts - shift_npt            # Start point for each slice
                slice_end_pt   = (islice + 1) * slice_npts - shift_npt    # End point for each slice
                
                slice_volt   = self._waveform_list[iwave].yarray[slice_start_pt : slice_end_pt] * u.V
                slice_volt   = slice_volt.to(self._vunit)
                peak_pt = np.argmax(slice_volt)
                y_peak[peak_pt] += 1
        return x_peak, y_peak

    def plot_peak_position(self):
        x_peak, y_peak = self.get_peak_position()

        plt.scatter(x_peak, y_peak)
        #plt.title(f'{self.file_name}')
        plt.xlabel(f'time ({self._tunit})')
        plt.ylabel(f'peak position (/bin)')
        plt.show()



    def time_integration(self, tstart, tstop):
        nwaves    = len(self._waveform_list)
        wave_npts = self._waveform_list[0].get_n()
        slice_npts = int(self._flash_interval/self._time_res)
        nslices    = int(wave_npts/slice_npts)
        
        #この中にif文でpmtとsipmに派生
        tstart_pt = int(tstart/self._time_res)
        tstop_pt   = int(tstop/self._time_res)
        
        integ_value_list = np.array([]) * self._vunit * self._tunit
        for iwave in range(0, nwaves):
            # Reset to 0 after each analysis of a certain waveform data
            sumtime_correction = 0.  # Accumulation time of self._flash_lag when analyzing each waveform slice
            shift_npt = 0            # Shift the slice start point by shift_npt
            
            for islice in range(0, nslices):
                sumtime_correction += self._flash_lag
                if sumtime_correction > self._time_res * (shift_npt + 1):
                    shift_npt += 1
                
                slice_start_pt = islice * slice_npts - shift_npt        # Start point for each slice
                slice_end_pt = (islice + 1) * slice_npts - shift_npt  # End point for each slice
                
                baseline_volt = np.mean(self._waveform_list[iwave].yarray[slice_start_pt + tstart_pt - int(10/self._time_res.value) - (tstop_pt - tstart_pt) : slice_start_pt + tstart_pt - int(10/self._time_res.value)] * u.V)
                slice_volt    = self._waveform_list[iwave].yarray[slice_start_pt + tstart_pt : slice_start_pt + tstop_pt] * u.V - baseline_volt
                slice_volt    = slice_volt.to(self._vunit)
                
                integ_value_list = np.append(integ_value_list, np.sum(slice_volt)*self._time_res)
        return integ_value_list