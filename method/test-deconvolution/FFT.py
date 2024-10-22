    def fft_deconvolution(signal, kernel):
        signal_fft = np.fft.fft(signal)
        kernel_fft = np.fft.fft(kernel)
        recover_fft = signal_fft / kernel_fft
        recover = np.fft.fftshift(np.fft.ifft(recover_fft))
        return np.real(recover)

    def power_wiener_deconvolution(signal, kernel, noise_variance):
        # ノイズの分散を計算
        noise_power = noise_variance * np.ones_like(input_signal)

        signal_fft = fft(signal)
        kernel_fft = fft(kernel)

        # Wiener deconvolutionをFFT領域で実行
        recover_fft = ( np.conj(kernel_fft) * signal_fft ) / ( np.conj(kernel_fft) * kernel_fft * signal_fft + noise_power)

        # 逆FFTを適用して復元
        recover = np.fft.fftshift(np.fft.ifft(recover_fft))
        return np.real(recover)
        
    def snr_wiener_deconvolution(signal, kernel, snr):
        signal_fft = np.fft.fft(signal)
        kernel_fft = np.fft.fft(kernel)
        
        # スペクトルの推定
        estimated_spectrum = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + 1 / snr)
        
        # Wiener deconvolution を適用
        recover_fft = signal_fft * estimated_spectrum
        # 逆FFT を使用して結果を時系列領域に変換
        recover = np.fft.fftshift(np.fft.ifft(recover_fft))
        return np.real(recover)
