"""
add some codes for TEKTRONIX new FW3.22 2020.10.7 due to changing header rules.
[old FW1.32 header]2;16;BINARY;RI;MSB;"Ch1, DC coupling, 10.00mV/div, 40.00us/div, 1000000 points, Sample mode";1000000;Y;LINEAR;"s";400.0000E-12;-199.8476E-6;0;"V";1.5625E-6;-13.0560E+3;0.0E+0;TIME;ANALOG;0.0E+0;0.0E+0;0.0E+0;#72000000
[new FW3.22 header]2;16;BINARY;RI;MSB;"Ch1, DC coupling, 10.00mV/div, 40.00us/div, 1000000 points, Sample mode";1000000;Y;       "s";400.0000E-12;-199.8476E-6;0;"V";1.5625E-6;-13.0560E+3;0.0E+0;                                 #72000000
"""

import gzip
import struct
import enum
import numpy as np

def gzip2waveforms(raw_data_file_name, nprocess=1):
    '''
    Convert a gzipped raw data file into SiPMWaveform instances. Here we assume
    individual waveforms are separated by '***'.
    '''
    try:
        f = gzip.open(raw_data_file_name)
        raw_data = f.read()
    except OSError as e:
        if e.args[0].find('Not a gzipped file') == 0:
            f = open(raw_data_file_name)
            raw_data = f.read()
    raw_data_list = raw_data.split(b'***')[:-1] # Assume here that *** is used as data splitter
    if nprocess > 1:
        from concurrent.futures import ProcessPoolExecutor #python2 can't use this module, so call it only when multiprocess are needed
        executor = ProcessPoolExecutor(max_workers = nprocess)
        futures = [executor.submit(SiPMWaveform, rd) for rd in raw_data_list]
        waveform_list = [f.result() for f in futures]
    else:
        waveform_list = [SiPMWaveform(rd) for rd in raw_data_list]
    
    return waveform_list


class InstrumentType(enum.Enum):
    TEKTRONIX = 0
    TEKTRONIX_newFW = 3
    LE_CROY = 1
    TARGET7 = 2

class SiPMWaveform:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.decode_raw_data(raw_data)

    def decode_raw_data(self, raw_data):

        #print("len = " + str(len(raw_data.split(b';'))))
        if raw_data.find(b'\r\n";') >= 0:
            self.instrument = InstrumentType.LE_CROY
            self.decode_lecroy_raw_data(raw_data)

        elif len(raw_data.split(b';')) >= 17 and len(raw_data.split(b';')) <= 20:
            
            self.instrument = InstrumentType.TEKTRONIX
            self.decode_tektronix_raw_data(raw_data)

        elif len(raw_data.split(b';')) >= 23:
            
            self.instrument = InstrumentType.TEKTRONIX_newFW
            self.decode_tektronix_raw_data_newFW(raw_data)

        else:
            print(len(raw_data.split(b';')))
            # Write other functions for TARGET modules and Tektronix
            print('Not implemented yet')

    def decode_lecroy_raw_data(self, raw_data):
        """
        input : lecroy rawdata
                acquired by command "INSPECT? 'WAVEDESC';c1:WAVEFORM? DAT1"
        output : x(np array,y(np array), header_info (dictionary)
        """
        header_raw, waveform_raw = raw_data.split(b'\r\n";')

        # convert header data to dictionary
        self.header = {}
        for readline in header_raw.split(b'\n'):
            if b':' in readline:
                if readline.count(b':') == 1:  # "<key>:<value>"
                    key, value = readline.split(b':')
                    self.header[key.replace(b' ', b'')] = value.replace(b' ', b'').replace(b'\r', b'')
                else:  # ex. "... Time =  4:51:11..."
                    key, value = readline.split(b': D') # add D "Time =  4: 5:10"
                    self.header[key.replace(b' ', b'')] = b'D' + value.replace(b' ', b'')

        Npt = int(self.header[b'WAVE_ARRAY_COUNT'])
        
        Nbyte = Npt * 2  # 16-bit
        wave_byte = waveform_raw[-Nbyte:]  # data segment

        # unpack raw wavedata
        # 16-bit signed integar (h) with lofirst order (>)
        self.yarray = np.array(struct.unpack('>' + 'h' * Npt, wave_byte))

        # scale y and add offset
        vscale = self.get_vscale()
        voffset = self.get_voffset()
        self.yarray = vscale * self.yarray - voffset

        # set x array
        dt = self.get_dt()
        self.xarray = np.linspace(0, dt*(Npt - 1), Npt)

    def decode_tektronix_raw_data_newFW(self, raw_data):
        header = self.header = raw_data.split(b';')[:16 + 6]
        byt_nr =   int(header[ 0])
        bit_nr =   int(header[ 1])
        encdg  =       header[ 2]
        bn_fmt =       header[ 3]
        byt_or =       header[ 4]
        wfid   =       header[ 5]
        nr_pt  =   int(header[ 6])
        pt_fmt =       header[ 7]

        xunit  =       header[ 9]
        xincr  = float(header[ 10])
        xzero  = float(header[11])
        pt_off =  bool(int(header[12]))
        yunit  =       header[13]
        ymult  = float(header[14])
        yoff   = float(header[15])
        yzero  = float(header[16])

##        print(header[13])

##        print('ymult = {0:1.5E}'.format(ymult))
        header_length = 0
        for i in range(16 + 6):
            header_length += len(header[i]) + 1
        raw_data = raw_data[header_length:]
      
        if raw_data[0] == 35:    ##35 == '#' ASCII
            # binary data
            
            ##Length of the part in the string that represents the number of sample points
            ##This length is read as ASCII, subtract ASCII zero character
            number_length = raw_data[1] - ord('0') 
            sample_pts = int(raw_data[2:2 + number_length])
            if sample_pts != byt_nr*nr_pt:
                print('NR_PT (numbr of point) is invalid.')
            if len(raw_data[2 + number_length:-1]) > byt_nr*nr_pt:
                print('Received data length is too long.')
            elif len(raw_data[2 + number_length:-1]) < byt_nr*nr_pt:
                print('Received data length is too short.')
            
            # RI means signed integer
            # RP means positive (unsigned) integer
            # MSB means the MSB is transmitted first
            # LSB means the LSB is transmitted first
            struct_format = ''  ## Has to be an ordinary string. For details refer to Python 'struct' library description
            if byt_or == b'MSB':
                struct_format += '>'
            elif byt_or == b'LSB':
                struct_format += '<'
            else:
                print('Invalid BYT_OR (%s)' % byt_or)


            if bn_fmt == b'RI' and byt_nr == 1:
                # signed char
                struct_format += 'b'*nr_pt
            elif bn_fmt == b'RI' and byt_nr == 2:
                # signed short
                struct_format += 'h'*nr_pt
            elif bn_fmt == b'RP' and byt_nr == 1:
                # unsigned char
                struct_format += 'B'*nr_pt
            elif bn_fmt == b'RP' and byt_nr == 2:
                # unsigned char
                struct_format += 'H'*nr_pt
            else:
                print('Invalid binary struct_format BN_FMT(%s) BYT_NR(%d).' % (bn_fmt, byt_nr))

            self.yarray = np.array(struct.unpack(struct_format, raw_data[2 + number_length:-1])) * self.get_vscale() + yzero - self.get_voffset()
            
        else:
            # ASCII data (???)
            pass

        self.xarray = (np.arange(nr_pt) - pt_off) * self.get_dt() + xzero

    def decode_tektronix_raw_data(self, raw_data):
        header = self.header = raw_data.split(b';')[:16]
        byt_nr =   int(header[ 0])
        bit_nr =   int(header[ 1])
        encdg  =       header[ 2]
        bn_fmt =       header[ 3]
        byt_or =       header[ 4]
        wfid   =       header[ 5]
        nr_pt  =   int(header[ 6])
        pt_fmt =       header[ 7]
        xunit  =       header[ 8]
        xincr  = float(header[ 9])
        xzero  = float(header[10])
        pt_off =  bool(int(header[11]))
        yunit  =       header[12]
        ymult  = float(header[13])
        yoff   = float(header[14])
        yzero  = float(header[15])

##        print(header[13])

##        print('ymult = {0:1.5E}'.format(ymult))
        header_length = 0
        for i in range(16):
            header_length += len(header[i]) + 1
        raw_data = raw_data[header_length:]
      
        if raw_data[0] == 35:    ##35 == '#' ASCII
            # binary data
            
            ##Length of the part in the string that represents the number of sample points
            ##This length is read as ASCII, subtract ASCII zero character
            number_length = raw_data[1] - ord('0') 
            sample_pts = int(raw_data[2:2 + number_length])
            if sample_pts != byt_nr*nr_pt:
                print('NR_PT (numbr of point) is invalid.')
            if len(raw_data[2 + number_length:-1]) > byt_nr*nr_pt:
                print('Received data length is too long.')
            elif len(raw_data[2 + number_length:-1]) < byt_nr*nr_pt:
                print('Received data length is too short.')
            
            # RI means signed integer
            # RP means positive (unsigned) integer
            # MSB means the MSB is transmitted first
            # LSB means the LSB is transmitted first
            struct_format = ''  ## Has to be an ordinary string. For details refer to Python 'struct' library description
            if byt_or == b'MSB':
                struct_format += '>'
            elif byt_or == b'LSB':
                struct_format += '<'
            else:
                print('Invalid BYT_OR (%s)' % byt_or)


            if bn_fmt == b'RI' and byt_nr == 1:
                # signed char
                struct_format += 'b'*nr_pt
            elif bn_fmt == b'RI' and byt_nr == 2:
                # signed short
                struct_format += 'h'*nr_pt
            elif bn_fmt == b'RP' and byt_nr == 1:
                # unsigned char
                struct_format += 'B'*nr_pt
            elif bn_fmt == b'RP' and byt_nr == 2:
                # unsigned char
                struct_format += 'H'*nr_pt
            else:
                print('Invalid binary struct_format BN_FMT(%s) BYT_NR(%d).' % (bn_fmt, byt_nr))

            self.yarray = np.array(struct.unpack(struct_format, raw_data[2 + number_length:-1])) * self.get_vscale() + yzero - self.get_voffset()
            
        else:
            # ASCII data (???)
            pass

        self.xarray = (np.arange(nr_pt) - pt_off) * self.get_dt() + xzero

    def decode_target_raw_data(self, raw_data):
        pass

    def filter_moving_average(self, n):
        """
        Calculate movinge average of waveform by using np.convolve
        https://docs.scipy.org/doc/np/reference/generated/np.convolve.html
        """
        weight = np.ones(n) / n
        return np.convolve(self.yarray, weight, 'same')

    def get_dt(self):
        if self.instrument == InstrumentType.LE_CROY:
            return float(self.header[b'HORIZ_INTERVAL'])
        elif self.instrument == InstrumentType.TEKTRONIX:
            return float(self.header[9])
        elif self.instrument == InstrumentType.TEKTRONIX_newFW:
            return float(self.header[10])
        else:
            raise 'Not implemented yet'

    def get_dv(self):
        if self.instrument == InstrumentType.LE_CROY:
            return float(self.header[b'VERTICAL_GAIN']) * 2 ** (16 - int(self.header[b'NOMINAL_BITS']))
        elif self.instrument == InstrumentType.TEKTRONIX:
            return float(self.get_vscale() * 2. ** 8.)
        elif self.instrument == InstrumentType.TEKTRONIX_newFW:
            return float(self.get_vscale() * 2. ** 8.)
        else:
            raise 'Not implemented yet'

    def get_vscale(self):
        if self.instrument == InstrumentType.LE_CROY:
            return float(self.header[b'VERTICAL_GAIN'])
        elif self.instrument == InstrumentType.TEKTRONIX:
            return float(self.header[13])
        elif self.instrument == InstrumentType.TEKTRONIX_newFW:
            return float(self.header[14])
        else:
            raise 'Not implemented yet'

    def get_voffset(self):
        if self.instrument == InstrumentType.LE_CROY:
            return float(self.header[b'VERTICAL_OFFSET'])
        elif self.instrument == InstrumentType.TEKTRONIX:
            return float(self.header[14]) * self.get_vscale()
        elif self.instrument == InstrumentType.TEKTRONIX_newFW:
            return float(self.header[15]) * self.get_vscale()
        else:
            raise 'Not implemented yet'

    def get_n(self):
        return len(self.xarray)
