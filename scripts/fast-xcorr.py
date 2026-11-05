import numpy as np
import h5py
from glob import glob
from scipy.fft import fft,fft2, ifft2, fftshift
from tqdm import tqdm
import os
import multiprocessing

def main():
    '''
    Load template waveform
    '''
    template = np.load('template.npz')['data']
    template_time = np.load('template.npz')['time']

    '''
    Load DAS data
    '''
    filelist = glob('/1-fnp/petasaur/p-jbod1/das4orcas/incoming/decimator_2024-11-0*.h5')
    filelist.sort()

    ''' Serial version'''
    # for file in filelist:
        # xc_workflow(file,template)

    ''' Parallel version'''
    # Build argument tuples
    args_list = [(file, template) for file in filelist]

    with multiprocessing.Pool(processes=24) as pool:
        pool.map(xc_workflow_wrapper, args_list)

def xc_workflow_wrapper(args):
    file, template = args
    xc_workflow(file, template)

def xc_workflow(file,template):
    '''
    Load DAS data and correlate with template
    '''
    # data = h5py.File(file, locking=False, mode='r')
    try:
        data = h5py.File(file, locking=False, mode='r')
    except Exception as e:
        print(f"Error opening file: {e}")
        return None

    attrs = dict(data['Acquisition'].attrs)
    dt = 1 / attrs['MaximumFrequency'] / 2
    dx = attrs['SpatialSamplingInterval']

    das = np.array(data['Acquisition/Raw[0]/RawData'])
    time = np.array(data['Acquisition/Raw[0]/RawDataTime'])

    data.close()

    # Get dimensions
    nt, nx = das.shape

    # Create coordinate arrays
    x = np.linspace(0, nx * dx, nx)
    t = np.linspace(0, nt * dt, nt)

    '''
    Correlate template waveform over each channel of the DAS data
    '''
    xc = window_and_correlate(template,das)
    max_xc = np.max(xc, axis=1)

    '''
    Save output as npz
    '''
    path = '/data/fast1/orcas/'
    savename = os.path.splitext(os.path.basename(file))[0]
    np.savez(f'{path}{savename}_xcorr.npz', xc=max_xc, time=time, x=x, dt=dt, dx=dx)


def correlate(s1,s2,mode="same",verbose=False):
    '''
    Cross correlate two signals using FFT
    s1: first signal
    s2: second signal
    mode: "full" or "same"
    verbose: print debug information
    '''

    # throw an error of input sizes are inconsistent
    # if s1.shape != s2.shape:
    #     raise ValueError("s1 and s2 must have the same size!")



    # Check shape match along time axis
    #nt1, nc1 = s1.shape
    if verbose: 
        print(f"Shape of s1: {s1.shape}")
        print(f"Shape of s2: {s2.shape}")
        if s1.ndim == 1:
            nt1 = s1.shape[0]
            nc1 = 1
        else: 
            nt1, nc1 = s1.shape
        nt2, nc2 = s2.shape
        print(f"nt1: {nt1}, nc1: {nc1}")
        print(f"nt2: {nt2}, nc2: {nc2}")



    if s1.shape[0] != s2.shape[0]:
        raise ValueError("s1 and s2 must have the same number of time samples!")

    # Broadcast template if needed

    if s1.ndim == 1 and s2.shape[1] > 1:
        s1 = np.tile(s1.reshape(-1,1), (1, s2.shape[1]))
    elif s1.shape[1] != s2.shape[1]:
        raise ValueError(f"s1 and s2 must have the same number of channels or s1 must be 1D. Instead, s1 has {s1.shape[1]} channels and s2 has {s2.shape[1]} channels.")
    


    # get fft size
    sz = s1.shape[0]
    n_bits = 1+int(np.log2(2*sz-1))
    fft_sz = 2**n_bits
    


    # take FFT along time axis for both
    fft_s1 = np.fft.fft(s1, fft_sz, axis=0)
    fft_s2 = np.fft.fft(s2, fft_sz, axis=0)

    # take complex conjugate of second signal
    fft_s2_conj = np.conj(fft_s2)

    # multiply to get correlation function
    corr_fft = fft_s1*fft_s2_conj

    # take inverse fourier transform
    corr = np.fft.ifft(corr_fft, axis=0)

    # # normalize using the magnitude of both input data
    norm1 = np.linalg.norm(s1,axis=0)
    norm2 = np.linalg.norm(s2,axis=0)
    norm_factor = norm1*norm2
    corr = np.vstack((corr[-(sz-1) :], corr[:sz]))
    norm_corr = np.real(corr) / norm_factor

    



    # return desired part of correlation function
    if mode == "full":
        pass
    elif mode == "same":
        norm_corr = norm_corr[int(sz/2):-int(sz/2)+1]
    return norm_corr

def window_and_correlate(template,data,verbose=False):
    # define container
    all_corr = []

    # get some helpful values
    window_length = template.shape[0]
    num_windows = int(data.shape[0]/window_length)
    if verbose: print(f'number of windows: {num_windows}')

    # iterate through time windows
    for i in tqdm(range(num_windows)):

        # pull out a time window of data - divides into non-overlapping windows corresponding to the template length
        start_index = i*window_length
        end_index = start_index + window_length
        window = data[start_index:end_index,:]

        # call cross correlation function
        corr = correlate(template,window)

        # save value
        all_corr.append(corr)

    # reshape output
    all_corr = np.stack(all_corr)

    return all_corr

if __name__ == "__main__":
    main()