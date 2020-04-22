import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import pandas as pd

# Import csv file
df = pd.read_csv("C- samajavaragamana_output_wav_Output.csv")
print(df.head())

# #plot data
# plt.figure(figsize=(12,4))
# df.plot(linestyle = '', marker = '*', color='r')
# #plt.savefig('rsam_2016_2017_snippetforfft.jpg')
# plt.show()

#FFT
#number of sample points
N = 407
#frequency of signal (in days)
T = 1
#create x-axis for time length of signal
#x = np.linspace(0, N*T, N)
#create array that corresponds to values in signal
y = df
#perform FFT on signal
yf = fft(y)

np.savetxt('outputTraining.csv',yf, delimiter=',')
# print(yf)
#create new x-axis: frequency from signal
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# #plot results
# plt.plot(xf, yf[0:N//2], label = 'signal')
# plt.grid()
# plt.xlabel('Frequency (days)')
# plt.ylabel(r'Spectral Amplitude')
# plt.legend(loc=1)
# #plt.savefig('rsam_2016_2017_snippet_fft_firstresult.png')
# plt.show()