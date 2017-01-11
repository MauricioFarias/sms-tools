import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF
eps = np.finfo(float).eps


"""
A4-Part-2: Measuring noise in the reconstructed signal using the STFT model 

Write a function that measures the amount of noise introduced during the analysis and synthesis of a 
signal using the STFT model. Use SNR (signal to noise ratio) in dB to quantify the amount of noise. 
Use the stft() function in stft.py to do an analysis followed by a synthesis of the input signal.

A brief description of the SNR computation can be found in the pdf document (A4-STFT.pdf, in Relevant 
Concepts section) in the assignment directory (A4). Use the time domain energy definition to compute
the SNR.

With the input signal and the obtained output, compute two different SNR values for the following cases:

1) SNR1: Over the entire length of the input and the output signals.
2) SNR2: For the segment of the signals left after discarding M samples from both the start and the 
end, where M is the analysis window length. Note that this computation is done after STFT analysis 
and synthesis.

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N), and hop size (H). The function should return a python 
tuple of both the SNR values in decibels: (SNR1, SNR2). Both SNR1 and SNR2 are float values. 

Test case 1: If you run your code using piano.wav file with 'blackman' window, M = 513, N = 2048 and 
H = 128, the output SNR values should be around: (67.57748352378475, 304.68394693221649).

Test case 2: If you run your code using sax-phrase-short.wav file with 'hamming' window, M = 512, 
N = 1024 and H = 64, the output SNR values should be around: (89.510506656299285, 306.18696700251388).

Test case 3: If you run your code using rain.wav file with 'hann' window, M = 1024, N = 2048 and 
H = 128, the output SNR values should be around: (74.631476225366825, 304.26918192997738).

Due to precision differences on different machines/hardware, compared to the expected SNR values, your 
output values can differ by +/-10dB for SNR1 and +/-100dB for SNR2.
"""

def computeSNR(inputFile, window, M, N, H):
    """
    Input:
            inputFile (string): wav file name including the path 
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming, 
                    blackman, blackmanharris)
            M (integer): analysis window length (odd positive integer)
            N (integer): fft size (power of two, > M)
            H (integer): hop size for the stft computation
    Output:
            The function should return a python tuple of both the SNR values (SNR1, SNR2)
            SNR1 and SNR2 are floats.
    """
    ## your code here
    
    # read input sound (monophonic with sampling rate of 44100)
    fs, x = UF.wavread(inputFile)
    w = get_window(window, M)
    y = stft.stft(x, w, N, H)
    if len(x) <> len(y):
        print ' x ' + str(len(x)) + ' y ' + str(len(y))
        return 0 ,0 

    #********************SNR1************************
    Ex = sum(x**2)

    noise = abs(x-y)
    Enoise = sum(noise**2)
    SNR1 = 10*np.log10(Ex/Enoise)

    #**********************SNR2***********************
    xp = x[M:-M] 
    yp = y[M:-M]
    
    Exp = sum(xp**2)
    
    if len(xp) <> len(yp):
        print ' x ' + str(len(x)) + ' xp ' + str(len(xp)) + ' yp ' + str(len(yp))
        return 0 ,0 
    noisep = abs(xp-yp)
    Enoisep = sum(noisep**2)
    SNR2 = 10*np.log10(Exp/Enoisep)

    return SNR1, SNR2

'''
def main(inputFile = '../../sounds/piano.wav', window = 'hamming', M = 1024, N = 1024, H = 512):
	"""
	analysis/synthesis using the STFT
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (choice of rectangular, hanning, hamming, blackman, blackmanharris)
	M: analysis window size
	N: fft size (power of two, bigger or equal than M)
	H: hop size (at least 1/2 of analysis window size to have good overlap-add)
	"""

	# read input sound (monophonic with sampling rate of 44100)
	fs, x = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# compute the magnitude and phase spectrogram

	mX, pX = STFT.stftAnal(x, w, N, H)
	 
	# perform the inverse stft
	y = STFT.stftSynth(mX, pX, M, H)

	# output sound file (monophonic with sampling rate of 44100)
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_stft.wav'

	# write the sound resulting from the inverse stft
	UF.wavwrite(y, fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(12, 9))

	# frequency range to plot
	maxplotfreq = 5000.0

	# plot the input sound
	plt.subplot(4,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')

	# plot magnitude spectrogram
	plt.subplot(4,1,2)
	numFrames = int(mX[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = fs*np.arange(N*maxplotfreq/fs)/N
	plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('magnitude spectrogram')
	plt.autoscale(tight=True)

	# plot the phase spectrogram
	plt.subplot(4,1,3)
	numFrames = int(pX[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = fs*np.arange(N*maxplotfreq/fs)/N
	plt.pcolormesh(frmTime, binFreq, np.transpose(np.diff(pX[:,:N*maxplotfreq/fs+1],axis=1)))
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('phase spectrogram (derivative)')
	plt.autoscale(tight=True)

	# plot the output sound
	plt.subplot(4,1,4)
	plt.plot(np.arange(y.size)/float(fs), y)
	plt.axis([0, y.size/float(fs), min(y), max(y)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('output sound: y')

	plt.tight_layout()
        plt.ion()
	plt.show()

'''
