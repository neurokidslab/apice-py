import numpy as np
from scipy import interpolate
from scipy.stats import yeojohnson
import warnings


from apice.utils import (get_data_size,
                         mask_bad_segments, 
                         include_short_bad_segments, 
                         reject_short_good_segments,
)


# %% CLASSES FOR ARTIFACTS DETECTION

class DetectionMethod:

    def __init__(self, bad_data=None, do_reference_data=False, do_zscore=False, 
                 mask=0, update_artifacts=True,
                 remove_bct=True, remove_bt=True, remove_bc=True,
                 name=None, group_name=None, verbose=True):

        # Arreange parameters
        self.params_compute = dict(bad_data=bad_data, do_reference_data=do_reference_data, do_zscore=do_zscore)
        self.params_reject = dict(mask=mask, update_artifacts=update_artifacts, remove_bct=remove_bct, remove_bt=remove_bt, remove_bc=remove_bc)
        
        self.name = name
        self.group_name = group_name
        self.verbose = verbose
        
    def steps_pre_compute(self, raw):
        
        # print info
        if self.verbose:
            print(f'Computing signal {self.name}...')
            for k in self.params_compute.keys():
                print(f'-- {k}: {self.params_compute[k]}')

        # Apply average reference or Z-score if requiered
        if self.params_compute['do_reference_data']:
            raw, self.ref = _set_reference(raw, bad_data=self.params_compute['bad_data'])
        if self.params_compute['do_zscore']:
            raw, self.sd = _compute_z_score(raw, bad_data=self.params_compute['bad_data'])
        
        return raw
    
    def steps_post_compute(self, raw):
        
        # save some info
        self.sfreq = raw.info['sfreq']
        self.data_shape = np.shape(raw._data)
        
        # Get data back
        if self.params_compute['do_zscore']:
            raw = _return_data_after_zscore(raw, self.sd)
        if self.params_compute['do_reference_data']:
            raw = _return_data_after_referencing(raw, self.ref)
        
        return raw
        
    def steps_pre_reject(self):
                
        if self.verbose:
            print('Rejecting data based on the signal {}...'.format(self.name))
            for k in self.params_reject.keys():
                print('-- {}: '.format(k), self.params_reject[k])
                
    def steps_post_reject(self, raw, bct, show_rej=None):
        
        # Mask around artifacts
        if self.params_reject['mask']:
            mask_length = np.round(self.params_reject['mask'] * raw.info['sfreq'])
            if len(np.shape(bct)) == 2:
                bct, _ = mask_bad_segments(bct, mask_length, axis=1)
            else:
                bct, _ = mask_bad_segments(bct, mask_length, axis=2)

        # keep some info
        n = np.size(bct)
        self.rej_new = np.sum(np.logical_and(raw.artifacts.BCT==0, bct) )/ n
        self.rej = np.sum(bct) / n 
        
        # Update rejection matrix
        if self.params_reject['update_artifacts']:
            raw.artifacts.BCT[bct] = 1 

        # Display the rejected data
        if self.verbose:
            print('-- Rejected data: ', np.round(self.rej * 100, 2), '%')
            if show_rej:
                for k,v in show_rej.items():
                        print('   - {} '.format(k), np.round(v / n * 100, 2), '%')
            print('-- Newly rejected data: ', np.round(self.rej_new * 100, 2), '%')
        
        return bct, raw
    

class Amplitude(DetectionMethod):

    """
    Class for artifacts detection base on amplitude
    """
    
    def __init__(self, bad_data=None, do_reference_data=False, do_zscore=False, name=None, group_name=None, verbose=True, 
                 thresh_type='outliers_per_channel', thresh=2, mask=0, update_artifacts=True,
                 remove_bct=True, remove_bt=True, remove_bc=True):
        
        super().__init__(bad_data=bad_data, 
                         do_reference_data=do_reference_data, 
                         do_zscore=do_zscore, 
                         mask=mask,
                         update_artifacts=update_artifacts,
                         remove_bct=remove_bct,
                         remove_bt=remove_bt,
                         remove_bc=remove_bc,
                         name=name,
                         group_name=group_name,
                         verbose=verbose)
        
        # Arreange parameters for rejection
        self.params_reject['thresh_type'] = thresh_type
        self.params_reject['thresh'] = thresh
                
    
    def compute(self, raw):
        
        # steps before computation
        raw = self.steps_pre_compute(raw)
        
        # Compute
        self._data = self.amplitude(raw)
        
        # steps after computation
        raw = self.steps_post_compute(raw)
        

    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Reject data 
        if self.params_reject['thresh_type']=='absolute':
            transform=None
        else:
            transform=None
        if isinstance(self.params_reject['thresh'], list) and len(self.params_reject['thresh'])==2:
            iq_half=False
        else:
            iq_half=True
        bct, rl_sum, ru_sum = reject_data(self._data.copy(), raw, self.params_reject['thresh'], 
                                          rejection=self.params_reject['thresh_type'], 
                                          transform=transform,
                                          remove_bct=self.params_reject['remove_bct'],
                                          remove_bt=self.params_reject['remove_bt'],
                                          remove_bc=self.params_reject['remove_bc'],
                                          iq_half=iq_half)
        
        # steps after rejection
        bct, raw = self.steps_post_reject(raw, bct, show_rej={'threshold':rl_sum+ru_sum})
        
        return raw, bct

    @staticmethod
    def amplitude(raw): 
        amp = raw.get_data()
        return amp
    


class RunningAverage(DetectionMethod):

    """
    Class for artifacts detection base of distance
    """

    def __init__(self, bad_data=None, fast_wind=0.05, slow_wind=0.15, do_reference_data=False, do_zscore=False, name=None, group_name=None, verbose=True, 
                 thresh_type='outliers_per_channel', thresh_fast=2.0, thresh_diff=3.0, mask=0, update_artifacts=True,
                 remove_bct=True, remove_bt=True, remove_bc=True):
        
        super().__init__(bad_data=bad_data, 
                         do_reference_data=do_reference_data, 
                         do_zscore=do_zscore, 
                         mask=mask,
                         update_artifacts=update_artifacts,
                         remove_bct=remove_bct,
                         remove_bt=remove_bt,
                         remove_bc=remove_bc,
                         name=name,
                         group_name=group_name,
                         verbose=verbose)
        
        # Arreange parameters for computation
        self.params_compute['fast_wind'] = fast_wind
        self.params_compute['slow_wind'] = slow_wind
        
        # Arreange parameters for rejection
        self.params_reject['thresh_type'] = thresh_type
        self.params_reject['thresh_fast'] = thresh_fast
        self.params_reject['thresh_diff'] = thresh_diff
                 
    
    def compute(self, raw):
        
        # steps before computation
        raw = self.steps_pre_compute(raw)
        
        # Compute
        self._data_fast, self._data_diff = self.runningaverage(raw, self.params_compute['fast_wind'], self.params_compute['slow_wind'])
        
        # steps after computation
        raw = self.steps_post_compute(raw)
        
     
    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Reject data 
        if self.params_reject['thresh_type']=='absolute':
            transform=None
        else:
            transform=None

        if isinstance(self.params_reject['thresh_fast'], list) and len(self.params_reject['thresh_fast'])==2:
            iq_half=False
        else:
            iq_half=True
        bct_fast, rl_sum_fast, ru_sum_fast = reject_data(self._data_fast.copy(), raw, self.params_reject['thresh_fast'], 
                                                         rejection=self.params_reject['thresh_type'], 
                                                         transform=transform,
                                                         remove_bct=self.params_reject['remove_bct'],
                                                         remove_bt=self.params_reject['remove_bt'],
                                                         remove_bc=self.params_reject['remove_bc'],
                                                         iq_half=iq_half)
        if isinstance(self.params_reject['thresh_diff'], list) and len(self.params_reject['thresh_diff'])==2:
            iq_half=False
        else:
            iq_half=True
        bct_diff, rl_sum_diff, ru_sum_diff = reject_data(self._data_diff.copy(), raw, self.params_reject['thresh_diff'], 
                                                         rejection=self.params_reject['thresh_type'], 
                                                         transform=transform,
                                                         remove_bct=self.params_reject['remove_bct'],
                                                         remove_bt=self.params_reject['remove_bt'],
                                                         remove_bc=self.params_reject['remove_bc'],
                                                         iq_half=iq_half)
        bct = np.logical_or(bct_fast, bct_diff)

        # steps after rejection
        show_rej={'fast threshold':rl_sum_fast+ru_sum_fast,
                  'diff threshold':rl_sum_diff+ru_sum_diff}
        bct, raw = self.steps_post_reject(raw, bct, show_rej=show_rej)
    
        return raw, bct
            


    @staticmethod
    def runningaverage(raw, fast_wind, slow_wind):
        
        # get data 
        n_channels, n_samples, n_epochs = get_data_size(raw)
        eeg_data = raw._data.copy()  
        if len(np.shape(raw._data))==2:
            eeg_data = np.reshape(eeg_data, (n_epochs, n_channels, n_samples))
    
        # intialize varaibles
        fast_average = np.empty((n_epochs, n_channels, n_samples))
        fast_average[:] = np.nan
        slow_average = np.empty((n_epochs, n_channels, n_samples))
        slow_average[:] = np.nan
    
        # Compute the running average
        n_fast = int(raw.info['sfreq']*fast_wind)
        n_slow = int(raw.info['sfreq']*slow_wind)
        for iep in range(n_epochs):
            for ich in range(n_channels):
                fast_average[iep, ich, :] = running_mean(eeg_data[iep,ich,:], n_fast)
                slow_average[iep, ich, :] = running_mean(eeg_data[iep,ich,:], n_slow)
        diff_average = fast_average - slow_average    
            
        # reshape if necessary
        if len(np.shape(raw._data))==2:
            fast_average = np.squeeze(np.transpose(fast_average,(1,2,0)), axis=2)
            slow_average = np.squeeze(np.transpose(slow_average,(1,2,0)), axis=2)
            diff_average = np.squeeze(np.transpose(diff_average,(1,2,0)), axis=2)  
                
        return fast_average, diff_average

def running_mean(x, N):
    if N % 2 ==0:
        x_ini = np.full(int(N/2), x[0])
        x_end = np.full(int(N/2), x[-1])
    else:
        x_ini = np.full(int(N/2)+1, x[0])
        x_end = np.full(int(N/2), x[-1])
    x = np.concatenate((x_ini, x, x_end),axis=0)
    cumsum = np.cumsum(x) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class TimeVariance(DetectionMethod):

    """
    Class for artifacts detection base on time variance
    """
    
    def __init__(self, bad_data=None, do_reference_data=False, do_zscore=False, name=None, group_name=None, verbose=True, 
                 time_window=0.5, time_window_step=0.1, 
                 thresh_type='outliers_per_channel', thresh=[None, 2.5], mask=0, update_artifacts=True,
                 remove_bct=True, remove_bt=True, remove_bc=True):
        
        super().__init__(bad_data=bad_data, 
                         do_reference_data=do_reference_data, 
                         do_zscore=do_zscore, 
                         mask=mask,
                         update_artifacts=update_artifacts,
                         remove_bct=remove_bct,
                         remove_bt=remove_bt,
                         remove_bc=remove_bc,
                         name=name,
                         group_name=group_name,
                         verbose=verbose)
        
        # Arreange parameters for computation
        self.params_compute['time_window'] = time_window
        self.params_compute['time_window_step'] = time_window_step
        
        # Arreange parameters for rejection
        self.params_reject['thresh_type'] = thresh_type
        self.params_reject['thresh'] = thresh
               
    
    def compute(self, raw):
        
        # steps before computation
        raw = self.steps_pre_compute(raw)
        
        # Compute
        self._data = self.timevariance(raw, self.params_compute['time_window'], self.params_compute['time_window_step'])
        
        # steps after computation
        raw = self.steps_post_compute(raw)
        

    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Reject data 
        if self.params_reject['thresh_type']=='absolute':
            transform=None
        else:
            transform='sqrt'
        bct, rl_sum, ru_sum = reject_data(self._data.copy(), raw, self.params_reject['thresh'], 
                                          transform=transform, 
                                          rejection=self.params_reject['thresh_type'],
                                          remove_bct=self.params_reject['remove_bct'],
                                          remove_bt=self.params_reject['remove_bt'],
                                          remove_bc=self.params_reject['remove_bc'],
                                          )
            
          
        # steps after rejection
        show_rej={'lower threshold':rl_sum,
                  'upper threshold':ru_sum}
        bct, raw = self.steps_post_reject(raw, bct, show_rej=show_rej)
    
        return raw, bct
    
    @staticmethod
    def timevariance(raw, time_window, time_window_step):
        
        n_channels, n_samples, n_epochs = get_data_size(raw)
        
        # find the time windows
        if time_window and time_window<(n_samples/raw.info['sfreq']):
            i_t, time_tw = define_time_window(raw, time_window,time_window_step)
            n_tw = np.shape(i_t)[1]
        else:
            i_t = np.arange(0,n_samples)
            n_tw = 1
        
        # get the data
        eeg_data = raw._data.copy()  
        if len(np.shape(raw._data))==2:
            eeg_data = np.reshape(eeg_data, (n_epochs, n_channels, n_samples))
    
        # compute time variance in time windows
        std_tw = np.empty((n_epochs, n_channels, n_tw))
        std_tw[:] = np.nan
        for ep in np.arange(n_epochs):
            for el in np.arange(n_channels):
                data = eeg_data[ep, el, :]
                data_time_window = data[i_t]
                std_tw[ep, el, :] = np.std(data_time_window, axis=0)
        
        # interpolate to have it in the data size
        if time_window and time_window<(n_samples/raw.info['sfreq']):
            std = interpolate_tw(raw, std_tw, time_tw)
        else:
            std = np.tile(std_tw, (1,1,n_samples))
        
        # reshape
        if len(np.shape(raw._data))==2:
            std = np.squeeze(np.transpose(std,(1,2,0)), axis=2)
                      
        return std
    
    
    
class MaxChange(DetectionMethod):

    """
    Class for artifacts detection base on maximun change
    """
    
    def __init__(self, bad_data=None, do_reference_data=False, do_zscore=False, name=None, group_name=None, verbose=True, 
                 time_window=0.100, time_window_step=0.01, 
                 thresh_type='outliers_per_channel', thresh=[None, 2.0], mask=0, update_artifacts=True,
                 remove_bct=True, remove_bt=True, remove_bc=True):
        
        super().__init__(bad_data=bad_data, 
                         do_reference_data=do_reference_data, 
                         do_zscore=do_zscore, 
                         mask=mask,
                         update_artifacts=update_artifacts,
                         remove_bct=remove_bct,
                         remove_bt=remove_bt,
                         remove_bc=remove_bc,
                         name=name,
                         group_name=group_name,
                         verbose=verbose)
        
        # Arreange parameters for computation
        self.params_compute['time_window'] = time_window
        self.params_compute['time_window_step'] = time_window_step
        
        # Arreange parameters for rejection
        self.params_reject['thresh_type'] = thresh_type
        self.params_reject['thresh'] = thresh
                 
    
    def compute(self, raw):
        
        # steps before computation
        raw = self.steps_pre_compute(raw)
        
        # Compute
        self._data = maxchange(raw, self.params_compute['time_window'], self.params_compute['time_window_step'])
        
        # steps after computation
        raw = self.steps_post_compute(raw)
        

    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Reject data 
        if self.params_reject['thresh_type']=='absolute':
            transform=None
        else:
            transform='cbrt'
        bct, rl_sum, ru_sum = reject_data(self._data.copy(), raw, self.params_reject['thresh'], 
                                          transform=transform, 
                                          rejection=self.params_reject['thresh_type'],
                                          remove_bct=self.params_reject['remove_bct'],
                                          remove_bt=self.params_reject['remove_bt'],
                                          remove_bc=self.params_reject['remove_bc'],
                                          )
         
        # steps after rejection
        show_rej={'lower threshold':rl_sum,
                  'upper threshold':ru_sum}
        bct, raw = self.steps_post_reject(raw, bct, show_rej=show_rej)
        
        return raw, bct
    

class CrossElectrodesOutlier(DetectionMethod):

    """
    Class for artifacts detection base on between electrodes variance
    """
    
    def __init__(self, bad_data=None, do_reference_data=False, do_zscore=True, name=None, group_name=None, verbose=True, 
                 time_window=0.020, time_window_step=0.005, 
                 thresh=2.5, mask=0, update_artifacts=True,
                 remove_bct=True, remove_bt=True, remove_bc=True):
        
        # data cannot be referenced to the average before computation
        if do_reference_data:
            do_reference_data = False
            warnings.warn('No reference average before computation allowed, set to False')
            
        # data must z-scored before computation
        if not do_zscore:
            do_zscore = True
            warnings.warn('z-score before computation is mandatory, set to True')
            
        super().__init__(bad_data=bad_data, 
                         do_reference_data=do_reference_data, 
                         do_zscore=do_zscore, 
                         mask=mask,
                         update_artifacts=update_artifacts,
                         remove_bct=remove_bct,
                         remove_bt=remove_bt,
                         remove_bc=remove_bc,
                         name=name,
                         group_name=group_name,
                         verbose=verbose)
        
        # Arreange parameters for computation
        self.params_compute['time_window'] = time_window
        self.params_compute['time_window_step'] = time_window_step
        
        # Arreange parameters for rejection
        self.params_reject['thresh'] = thresh
                 
    
    def compute(self, raw):
        
        # steps before computation
        raw = self.steps_pre_compute(raw)
        
        # Compute
        self._data = self.acrosselectrodesoutlier(raw, self.params_compute['time_window'], self.params_compute['time_window_step'])
        
        # steps after computation
        raw = self.steps_post_compute(raw)
        
        
    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        
        # Reject data 
        bct, rl_sum, ru_sum = reject_data(self._data.copy(), raw, self.params_reject['thresh'], rejection='absolute')
        
        # steps after rejection
        bct, raw = self.steps_post_reject(raw, bct, show_rej={'threshold': ru_sum})
        
        return raw, bct
    
    @staticmethod
    def acrosselectrodesoutlier(raw, time_window, time_window_step):
        
        n_channels, n_samples, n_epochs = get_data_size(raw)
        
        # find the time windows
        if time_window and time_window<(n_samples/raw.info['sfreq']):
            i_t, time_tw = define_time_window(raw, time_window,time_window_step)
            n_tw = np.shape(i_t)[1]
        else:
            i_t = np.arange(0,n_samples)
            n_tw = 1
            
        # get the data
        eeg_data = raw._data.copy()  
        bct = raw.artifacts.BCT.copy()
        if len(np.shape(raw._data))==2:
            eeg_data = np.reshape(eeg_data, (n_epochs, n_channels, n_samples))
            bct = np.reshape(bct, (n_epochs, n_channels, n_samples))
        
        # clean data
        eeg_data_clean = eeg_data.copy()
        eeg_data_clean[bct==1] = np.nan
        
        # comput electrodes activity relatively to the variance across electrodes in time windows
        el_desviation_tw = np.empty((n_epochs, n_channels, n_tw))
        el_desviation_tw[:] = 0
        for ep in np.arange(n_epochs):
            for itw in range(np.shape(i_t)[1]):
                d = eeg_data_clean[ep, :, i_t[:,itw]].flatten()
                idx = np.isnan(d)==0
                if np.sum(idx)>10:
                    q3 = np.percentile(abs(d[np.isnan(d)==0]), 50)
                    IQ = 2*q3
                    dd = eeg_data[ep, :, i_t[:,itw]]
                    mu = np.mean(dd, axis=0)
                    el_desviation_tw[ep,:,itw] = (abs(mu)-q3) / IQ
                    
        # interpolate to have it in the data size
        if time_window and time_window<(n_samples/raw.info['sfreq']):
            el_desviation = interpolate_tw(raw, el_desviation_tw, time_tw)
        else:
            el_desviation = np.tile(el_desviation_tw, (1,1,n_samples))
                     
        # reshape
        if len(np.shape(raw._data))==2:
            el_desviation = np.squeeze(np.transpose(el_desviation,(1,2,0)), axis=2)
                         
        return el_desviation
    
      

class Power(DetectionMethod):

    """
    Class for artifacts detection base on power
    """
    
    def __init__(self, bad_data=None, do_reference_data=False, do_zscore=False, name=None, group_name=None, verbose=True, 
                 time_window=10, time_window_step=5, freq_band=[20, 40],
                 thresh=[None,2.0], mask=0.05, update_artifacts=True,
                 remove_bct=True, remove_bt=True, remove_bc=True):
        
        super().__init__(bad_data=bad_data, 
                         do_reference_data=do_reference_data, 
                         do_zscore=do_zscore, 
                         mask=mask,
                         update_artifacts=update_artifacts,
                         remove_bct=remove_bct,
                         remove_bt=remove_bt,
                         remove_bc=remove_bc,
                         name=name,
                         group_name=group_name,
                         verbose=verbose)
        
        # Arreange parameters for computation
        self.params_compute['time_window'] = time_window
        self.params_compute['time_window_step'] = time_window_step
        self.params_compute['freq_band'] = freq_band
        
        # Arreange parameters for rejection
        self.params_reject['thresh'] = thresh
                 
    
    def compute(self, raw):
        
        # steps before computation
        raw = self.steps_pre_compute(raw)
        
        # Compute
        self._data = self.powerperband(raw, self.params_compute['time_window'], self.params_compute['time_window_step'], self.params_compute['freq_band'])
        
        # steps after computation
        raw = self.steps_post_compute(raw)
        

    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Reject data 
        transform=None
        bct, rl_sum, ru_sum = reject_data(self._data.copy(), raw, self.params_reject['thresh'], 
                                          rejection='outliers_all',
                                          transform=transform,
                                          remove_bct=self.params_reject['remove_bct'],
                                          remove_bt=self.params_reject['remove_bt'],
                                          remove_bc=self.params_reject['remove_bc'],
                                          )
        
        # steps after rejection
        show_rej={'lower threshold':rl_sum,
                  'upper threshold':ru_sum}
        bct, raw = self.steps_post_reject(raw, bct, show_rej=show_rej)
        
        return raw, bct

    @staticmethod
    def powerperband(raw, time_window, time_window_step, freq_band):

        n_channels, n_samples, n_epochs = get_data_size(raw)
        
        # find the time windows
        i_t, time_tw = define_time_window(raw, time_window,time_window_step)
        n_tw = np.shape(i_t)[1]
        
        # Initialize variables
        powerband_tw = np.empty((n_epochs, n_channels, n_tw))
        powerband_tw[:] = np.nan

        # compute
        # raw_mne = raw.to_mne_raw(annotate_channels=False, annotate_times=False, annotate_data=False, annotate_corrected=False)
        for i in range(n_tw):
            dat = raw.copy().crop(tmin=raw.times[i_t[0,i]], tmax=raw.times[i_t[-1,i]], verbose='ERROR')
            spc = dat.compute_psd(method='multitaper', fmin=freq_band[0], fmax=freq_band[1], verbose='ERROR')
            p = spc.get_data()
            if len(np.shape(raw._data))==2:
                m = np.mean(p, axis=1)
                # p_band = np.where(m > 0, np.log10(np.maximum(m, np.finfo(float).tiny)), np.nan)
                p_band = np.log10(np.maximum(m, np.finfo(float).tiny))
                p_base = np.nanmedian(p_band)
                p_band = p_band - p_base
            else:
                m = np.mean(p, axis=2)
                p_band = np.log10(np.maximum(m, np.finfo(float).tiny))
                p_base = np.nanmedian(p_band.flatten())
                p_band = p_band - p_base
            powerband_tw[:,:,i] = p_band
            
        # interpolate to have it in the data size
        powerband = interpolate_tw(raw, powerband_tw, time_tw)
        
        # reshape
        if len(np.shape(raw._data))==2:
            powerband = np.squeeze(np.transpose(powerband,(1,2,0)), axis=2)   
               
        return powerband




class ChannelCorr(DetectionMethod):
    
    """
    Class for artifacts detection base on between channels correlation
    """
    
    def __init__(self, bad_data=None, do_reference_data=False, do_zscore=False, name=None, group_name=None, verbose=True, 
                 time_window=10, time_window_step=5, top_channel_corr=5,
                 thresh=0.4, mask=0.05, update_artifacts=True,
                 remove_bct=True, remove_bt=True, remove_bc=True):
        
        super().__init__(bad_data=bad_data, 
                         do_reference_data=do_reference_data, 
                         do_zscore=do_zscore, 
                         mask=mask,
                         update_artifacts=update_artifacts,
                         remove_bct=remove_bct,
                         remove_bt=remove_bt,
                         remove_bc=remove_bc,
                         name=name,
                         group_name=group_name,
                         verbose=verbose)
        
        # Arreange parameters for computation
        self.params_compute['time_window'] = time_window
        self.params_compute['time_window_step'] = time_window_step
        self.params_compute['top_channel_corr'] = top_channel_corr
        
        # Arreange parameters for rejection
        self.params_reject['thresh'] = thresh
                 
    
    def compute(self, raw):
        
        # steps before computation
        raw = self.steps_pre_compute(raw)
        
        # Compute
        self._data = self.chennelscorr(raw, self.params_compute['time_window'], self.params_compute['time_window_step'], self.params_compute['top_channel_corr'])
        
        # steps after computation
        raw = self.steps_post_compute(raw)
        

    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Reject data 
        thresh = [self.params_reject['thresh'], None]
        bct, rl_sum, ru_sum = reject_data(self._data.copy(), raw, thresh, 
                                          rejection='absolute',
                                          remove_bct=self.params_reject['remove_bct'],
                                          remove_bt=self.params_reject['remove_bt'],
                                          remove_bc=self.params_reject['remove_bc'],
                                          )
        
        # steps after rejection
        bct, raw = self.steps_post_reject(raw, bct, show_rej={'lower threshold': rl_sum})
        
        return raw, bct

    @staticmethod
    def chennelscorr(raw, time_window, time_window_step, top_channel_corr):

        # get the data
        n_channels, n_samples, n_epochs = get_data_size(raw)
        eeg_data = raw._data.copy()  
        if len(np.shape(raw._data))==2:
            eeg_data = np.reshape(eeg_data, (n_epochs, n_channels, n_samples))
    
        # find the time windows
        i_t, time_tw = define_time_window(raw, time_window,time_window_step)
        n_tw = np.shape(i_t)[1]
        
        # Initialize variables
        chacorr_tw = np.empty((n_epochs, n_channels, n_tw))
        chacorr_tw[:] = np.nan

        # compute excluding from the computation dead channels (all zeros)
        idx_cha = np.sum(eeg_data, axis=(0,2))!=0
        for ep in range(n_epochs):
            for itw in range(n_tw):
                d = eeg_data[ep][idx_cha][:, i_t[:, itw]]  
                # compute the correlation with all channels
                channel_corr = np.abs(np.corrcoef(d))
                # remove correlation with self
                channel_corr[np.identity(np.shape(channel_corr)[0], dtype=bool)] = np.nan
                # average of the top correlation
                ptop = np.nanpercentile(channel_corr, 100 - top_channel_corr, axis=0, method='hazen')
                channel_corr[channel_corr <= np.tile(ptop, (np.sum(idx_cha), 1))] = np.nan
                average_corr = np.nanmean(channel_corr, axis=0)
                # store the data
                chacorr_tw[ep, idx_cha, itw] = average_corr


        # interpolate to have it in the data size
        chacorr = interpolate_tw(raw, chacorr_tw, time_tw)
    
        # reshape
        if len(np.shape(raw._data))==2:
            chacorr = np.squeeze(np.transpose(chacorr,(1,2,0)), axis=2)    
               
        return chacorr



class FlatChannel(DetectionMethod):
    
    """
    Class for detecting flat channels
    """
    
    def __init__(self, bad_data=None, do_reference_data=False, do_zscore=False, name=None, group_name=None, verbose=True, 
                 time_window=10, time_window_step=5, min_change=1e-7, 
                 thresh=5, mask=0, update_artifacts=True,
                 remove_bct=True, remove_bt=True, remove_bc=True):
        
        super().__init__(bad_data=bad_data, 
                         do_reference_data=do_reference_data, 
                         do_zscore=do_zscore, 
                         mask=mask,
                         update_artifacts=update_artifacts,
                         remove_bct=remove_bct,
                         remove_bt=remove_bt,
                         remove_bc=remove_bc,
                         name=name, 
                         group_name=group_name,
                         verbose=verbose)
        
        # Arreange parameters for computation
        self.params_compute['time_window'] = time_window
        self.params_compute['time_window_step'] = time_window_step
        self.params_compute['min_change'] = min_change
        
        # Arreange parameters for rejection
        self.params_reject['thresh'] = thresh
                 
    
    def compute(self, raw):
        
        # steps before computation
        raw = self.steps_pre_compute(raw)
        
        # Compute
        self._data = self.flatchannel(raw, self.params_compute['time_window'], 
                                      self.params_compute['time_window_step'],
                                      self.params_compute['min_change'])
        
        # steps after computation
        raw = self.steps_post_compute(raw)
        

    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Reject data 
        thresh = [None, self.params_reject['thresh']]
        bct, rl_sum, ru_sum = reject_data(self._data.copy(), raw, thresh, 
                                          rejection='absolute',
                                          remove_bct=self.params_reject['remove_bct'],
                                          remove_bt=self.params_reject['remove_bt'],
                                          remove_bc=self.params_reject['remove_bc'],
                                          )
        
        # steps after rejection
        bct, raw = self.steps_post_reject(raw, bct, show_rej={'lower threshold': rl_sum})
        
        return raw, bct

    @staticmethod
    def flatchannel(raw, time_window, time_window_step, min_change):
        
        # maximiun change in time windows of 5 samples 
        twind = 5/raw.info['sfreq']
        change = maxchange(raw, twind, twind/2)
        
        # check when the change is too small
        small_change = change<min_change
        
        # reshape
        n_channels, n_samples, n_epochs = get_data_size(raw)
        if len(np.shape(raw._data))==2:
            small_change = small_change[np.newaxis,:,:]
            
        # find the time windows
        i_t, time_tw = define_time_window(raw, time_window, time_window_step)
        n_tw = np.shape(i_t)[1]
        
        # Initialize variables
        proportion_small_change_tw = np.empty((n_epochs, n_channels, n_tw))
        proportion_small_change_tw[:] = np.nan

        # compute the proportion of data with a small chnage in the time windows
        for ep in range(n_epochs):
            for itw in range(n_tw):
                d = small_change[ep, :, i_t[:, itw]]  
                # propotion of data with small changes
                p = 100*(np.sum(d, axis=0)/d.shape[0])
                # store the data
                proportion_small_change_tw[ep, :, itw] = p


        # interpolate to have it in the data size
        proportion_small_change = interpolate_tw(raw, proportion_small_change_tw, time_tw)
    
        # reshape
        if len(np.shape(raw._data))==2:
            proportion_small_change = np.squeeze(np.transpose(proportion_small_change,(1,2,0)), axis=2)    
               
        return proportion_small_change


# %% CLASSES FOR MODIFYING REJECTION

class ModifyRejection:

    def __init__(self, update_artifacts=True, name=None, group_name=None, verbose=True):

        # Arreange parameters
        self.params = dict(update_artifacts=update_artifacts)
        self.name = name
        self.group_name = group_name
        self.verbose = verbose


    def steps_pre_reject(self):
                
        if self.verbose:
            print('Modifying rejection matrix by {}...'.format(self.name))
            for k in self.params.keys():
                print('-- {}: '.format(k), self.params[k])

    def steps_post_reject(self, bct, raw, change_to=1):
        
        # keep some info
        n = np.size(bct)
        self.rej_new = np.sum(np.logical_and(raw.artifacts.BCT==0, bct) )/ n
        self.rej = np.sum(bct) / n 
        
        # Update rejection matrix
        if self.params['update_artifacts']:
            raw.artifacts.BCT[bct==1] = change_to

        # Display the rejected data
        if self.verbose:
            print('-- Modified data: ', np.round(self.rej * 100, 2), '%')
        
        return bct, raw

class Mask(ModifyRejection):

    """
    Class for masking rejection 
    """
    
    def __init__(self, mask_length=0.5, update_artifacts=True, name=None, group_name=None, verbose=True):
        
        super().__init__(update_artifacts=update_artifacts, name=name, group_name=group_name, verbose=verbose)
        
        # Arreange parameters 
        self.params['mask_length'] = mask_length
        
    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Mask rejection matrix
        bct = self.apply_mask(raw, self.params['mask_length'])
          
        # steps after rejection
        bct, raw = self.steps_post_reject(bct, raw)
        
        return raw, bct

    @staticmethod
    def apply_mask(raw, mask_length):

        n_channels, n_samples, n_epochs = get_data_size(raw)
        mask_length = np.round(mask_length * raw.info['sfreq'])
        
        # Mask
        if len(np.shape(raw.artifacts.BCT))==2:
            _, bct = mask_bad_segments(raw.artifacts.BCT.copy(), mask_length, axis=1)
        else:
            _, bct = mask_bad_segments(raw.artifacts.BCT.copy(), mask_length, axis=2)
        
        return bct



class ShortGoodSegments(ModifyRejection):

    """
    Class for masking rejection 
    """
    
    def __init__(self, time_limit=2, update_artifacts=True, name=None, group_name=None, verbose=True):
        
        super().__init__(update_artifacts=update_artifacts, name=name, group_name=group_name, verbose=verbose)
        
        # Arreange parameters 
        self.params['time_limit'] = time_limit
        
    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Reject short good segments
        bct = self.apply_rejection_short_good(raw, self.params['time_limit'])
                   
        # steps after rejection
        bct, raw = self.steps_post_reject(bct, raw)
        
        return raw, bct


    @staticmethod
    def apply_rejection_short_good(raw, time_limit):

        n_channels, n_samples, n_epochs = get_data_size(raw)
        time_limit = np.round(time_limit * raw.info['sfreq'])
        if n_samples <= time_limit:
            time_limit = n_samples - 1

        # Reject short good segments
        if len(np.shape(raw.artifacts.BCT))==2:
            _, bct = reject_short_good_segments(raw.artifacts.BCT.copy(), time_limit, axis=1)
        else:
            _, bct = reject_short_good_segments(raw.artifacts.BCT.copy(), time_limit, axis=2)
        
            
        return bct


class ShortBadSegments(ModifyRejection):

    """
    Class for masking rejection 
    """
    
    def __init__(self, time_limit=2, update_artifacts=True, name=None, group_name=None, verbose=True):
        
        super().__init__(update_artifacts=update_artifacts, name=name, group_name=group_name, verbose=verbose)
        
        # Arreange parameters 
        self.params['time_limit'] = time_limit
        
    def reject(self, raw):
        
        # steps before rejection
        self.steps_pre_reject()
        
        # Keep short bad segments
        include_segments = self.apply_include_short_bad(raw, self.params['time_limit'])
                   
        # steps after rejection
        bct, raw = self.steps_post_reject(include_segments, raw, change_to=0)
        
        return raw, include_segments     


    @staticmethod
    def apply_include_short_bad(raw, time_limit):
        
        n_channels, n_samples, n_epochs = get_data_size(raw) 
        time_limit = np.round(time_limit * raw.info['sfreq'])
    
        # Determine short bad segments
        if len(np.shape(raw.artifacts.BCT))==2:
           _, include_segments = include_short_bad_segments(raw.artifacts.BCT.copy(), time_limit, axis=1)
        else:
           _, include_segments = include_short_bad_segments(raw.artifacts.BCT.copy(), time_limit, axis=2)
        
        
        return include_segments



# %% FUNCTIONS

def maxchange(raw, time_window, time_window_step):
    
    n_channels, n_samples, n_epochs = get_data_size(raw)
    
    # find the time windows
    if time_window and time_window<(n_samples/raw.info['sfreq']):
        i_t, time_tw = define_time_window(raw, time_window,time_window_step)
        n_tw = np.shape(i_t)[1]
    else:
        i_t = np.arange(0,n_samples)
        n_tw = 1
    
    # get the data
    eeg_data = raw._data.copy()  
    if len(np.shape(raw._data))==2:
        eeg_data = np.reshape(eeg_data, (n_epochs, n_channels, n_samples))

    # comput max-min difference in time windows
    maxmindiff_tw = np.empty((n_epochs, n_channels, n_tw))
    maxmindiff_tw[:] = np.nan
    for ep in np.arange(n_epochs):
        for el in np.arange(n_channels):
            data = eeg_data[ep, el, :]
            data_time_window = data[i_t]
            maxmindiff_tw[ep, el, :] = np.max(data_time_window, axis=0) - np.min(data_time_window, axis=0)
                
    # interpolate to have it in the data size
    if time_window and time_window<(n_samples/raw.info['sfreq']):
        maxmindiff = interpolate_tw(raw, maxmindiff_tw, time_tw)
    else:
        maxmindiff = np.tile(maxmindiff_tw, (1,1,n_samples))
         
    # reshape
    if len(np.shape(raw._data))==2:
        maxmindiff = np.squeeze(np.transpose(maxmindiff,(1,2,0)), axis=2)
               
    return maxmindiff


def interpolate_tw(raw, data_tw, time_tw):
    n_channels, n_samples, n_epochs = get_data_size(raw)
    
    data = np.empty((n_epochs, n_channels, n_samples))
    data[:]= np.nan
    idx_t = np.logical_and(raw.times>=time_tw[0], raw.times<=time_tw[-1])
    idx_ini = raw.times<time_tw[0] 
    idx_end = raw.times>time_tw[-1]
    for ep in np.arange(n_epochs):
        for el in np.arange(n_channels):
            if np.sum(np.isnan(data_tw[ep, el, :]))==len(time_tw):
                continue
            d = data_tw[ep, el, :].flatten()
            f = interpolate.interp1d(time_tw, d, kind='linear')
            dnew = f(raw.times[idx_t])
            data[ep,el,idx_t] = dnew
            data[ep,el,idx_ini] = dnew[0]
            data[ep,el,idx_end] = dnew[-1]
 
    return data

def define_time_window(raw, time_window, time_window_step):
    """
    This function creates a matrix that divides the data into segments whose length is defined by the
    time window duration sliding based on the number of step length.
    :param raw: object containing the eeg data
    :param time_window: window duration in seconds
    :param time_window_step: stride in seconds
    :return: i_t: windows limits in samples
             n_time_window: number of generated time windows
    """
    n_channels, n_samples, n_epochs = get_data_size(raw)

    time_window = int(np.round(time_window * raw.info['sfreq']))
    time_window_step = int(np.round(time_window_step * raw.info['sfreq']))
    n_time_window = int(np.round((n_samples - time_window + 1) / time_window_step) + 1)
    if n_time_window <= 0:
        warnings.warn('The time window is too long.')
        i_t = np.arange(0,n_samples)
        i_t = i_t[:,np.newaxis]
    else:
        # Indices of the window limits
        i_t = np.asarray((np.round(np.linspace(0, n_samples - time_window, n_time_window))), dtype=int)
        i_t = np.asarray(np.tile(i_t, (time_window, 1)) + np.tile(np.arange(time_window), (len(i_t), 1)).T,
                         dtype=int)
    
    # time verctor for the half of the time window
    time_tw = (i_t[0,:] + time_window/2)/raw.info['sfreq'] + raw.times[0]
    
    return i_t, time_tw

def data_transformation(data, transform, yeojohnson_lambda=None):

    if transform not in ['sqrt', 'cbrt', 'log', 'yeojohnson', None]:
        raise Exception("transform can take one of these values ['sqrt', 'cbrt', 'log', 'yeojohnson', None]")
    
    if transform is None:
        return data
    if transform=='sqrt':
        print("Applying square root transformation")
        data = np.sqrt(data)
    if transform=='cbrt':
        print("Applying cube root transformation")
        data = np.cbrt(data)
    if transform=='log':
        print("Applying log transformation")
        data = np.log(data)
    if transform=='yeojohnson':
        if yeojohnson_lambda is None:
            raise Exception("yeojohnson_lambda is required when using yeojohnson transformation")
        print(f"Applying Yeo-Johnson transformation with lambda={yeojohnson_lambda}")
        data = yeojohnson(data, lmbda=yeojohnson_lambda)
        
    return data

def reject_data(data, raw, thresh, transform=None, yeojohnson_lambda=None, iq_half=False, rejection='absolute', remove_bt=True, remove_bct=True, remove_bc=True):
    
    possible_rejection = ['absolute', 'outliers_all', 'outliers_per_channel']
    if rejection not in possible_rejection:
        raise Exception("rejection can take one of these values {}".format(possible_rejection))
            
    if rejection=='absolute':
        return reject_absolute(data, thresh, iq_half=iq_half)
  
    if rejection=='outliers_per_channel':
        data = data_transformation(data, transform, yeojohnson_lambda)
        return reject_relative_per_cha(data, raw.artifacts, thresh, iq_half=iq_half, remove_bt=remove_bt, remove_bct=remove_bct)
    
    if rejection=='outliers_all':
        data = data_transformation(data, transform, yeojohnson_lambda)
        return reject_relative(data, raw.artifacts, thresh, iq_half=iq_half, remove_bt=remove_bt, remove_bct=remove_bct, remove_bc=remove_bc)
    


def reject_relative_per_cha(data, art, thresh, iq_half=False, remove_bt=True, remove_bct=True):
        
    if iq_half:
        if type(thresh)==float or type(thresh)==int:
            thresh= [thresh]
        if len(thresh)>1:
            raise Exception("a unique threshold is required when using iq_half=True")
        
    else:
        if type(thresh)==float or type(thresh)==int:
            raise Exception("two thresholds are required")
        if len(thresh)!=2:
            raise Exception("two thresholds are required")
        
            
    n_channels = art.n_channels
    n_samples = art.n_samples
    n_epochs = art.n_epochs
    
    threshold_matrix = np.empty((n_channels,2))
    threshold_matrix[:] = np.nan
    
    data_to_reject = np.full((n_epochs, n_channels, n_samples), False)
    bct = art.BCT.copy()
    bt = art.BT.copy()
    
    if len(np.shape(art.BCT))==2:
        data = np.reshape(data,(n_epochs,n_channels,n_samples))
        bct = np.reshape(bct,(n_epochs,n_channels,n_samples))
        bt = np.reshape(bt,(n_epochs,1,n_samples))
        
    # mark as nan data already rejected
    data_rej = data.copy()
    if remove_bct:
        data_rej[bct==1] = np.nan    
    if remove_bt:
        data_rej[np.tile(bt,(1,n_channels,1))==1] = np.nan    
    
    # reject
    ru_sum = 0  # upper threshold
    rl_sum = 0  # upper threshold
    for el in np.arange(n_channels):
        
        d = data_rej[:,el,:].flatten()
        if iq_half:
            perc = np.nanpercentile(np.abs(d), 50, method='hazen')
            IQ = 2 * perc # Get the half distribution centered at zero
            t_u = perc + thresh[0] * IQ
            t_l = -t_u
        else:
            perc = np.nanpercentile(d, [25, 75], method='hazen')
            IQ = perc[1] - perc[0]
            if thresh[0]:
                t_l = perc[0] + thresh[0] * IQ   
            else:
                t_l = -np.inf   
            if thresh[1]:
                t_u = perc[1] + thresh[1] * IQ
            else:
                t_u = np.inf   
                   
        data_to_reject[:,el,:] = np.logical_or(data[:,el,:] < t_l, data[:,el,:] > t_u)
        threshold_matrix[el,0] = t_l
        threshold_matrix[el,1] = t_u
        
        rl_sum = rl_sum + np.sum(data[:,el,:].copy().flatten() < t_l)
        ru_sum = ru_sum + np.sum(data[:,el,:].copy().flatten() > t_u)

    if len(np.shape(art.BCT))==2:
        data_to_reject = np.squeeze(np.transpose(data_to_reject,(1,2,0)))
        data = np.squeeze(np.transpose(data,(1,2,0)))
        
    return data_to_reject, rl_sum, ru_sum


def reject_relative(data, art, thresh, iq_half=False, remove_bt=True, remove_bct=True, remove_bc=True):
        
    if iq_half:
        if type(thresh)==float or type(thresh)==int:
            thresh= [thresh]
        if len(thresh)>1:
            raise Exception("a unique threshold is required when using iq_half=True")
        
    else:
        if type(thresh)==float or type(thresh)==int:
            raise Exception("two thresholds are required")
        if len(thresh)!=2:
            raise Exception("two thresholds are required")
        
            
    n_channels = art.n_channels
    n_samples = art.n_samples
    n_epochs = art.n_epochs
    
    threshold_matrix = np.empty((1,2))
    threshold_matrix[:] = np.nan
    
    bct = art.BCT.copy()
    bt = art.BT.copy()
    bc = art.BC.copy()
    
    if len(np.shape(art.BCT))==2:
        data = np.reshape(data,(n_epochs,n_channels,n_samples))
        bct = np.reshape(bct,(n_epochs,n_channels,n_samples))
        bt = np.reshape(bt,(n_epochs,1,n_samples))
        bc = np.reshape(bc,(n_epochs,n_channels,1))
        
    # mark as nan data already rejected
    data_rej = data.copy()
    if remove_bct:
        data_rej[bct==1] = np.nan    
    if remove_bt:
        data_rej[np.tile(bt,(1,n_channels,1))==1] = np.nan    
    if remove_bc:
        data_rej[np.tile(bc,(1,1,n_samples))==1] = np.nan       
    
    # reject
    if iq_half:
        perc = np.nanpercentile(np.abs(data_rej.flatten()), 50, method='hazen')
        IQ = 2 * perc # Get the half distribution centered at zero
        t_u = perc + thresh[0] * IQ
        t_l = -t_u
    else:
        perc = np.nanpercentile(data_rej.flatten(), [25,75], method='hazen')
        IQ = perc[1] - perc[0]
        if thresh[0]:
            t_l = perc[0] + thresh[0] * IQ   
        else:
            t_l = -np.inf   
        if thresh[1]:
            t_u = perc[1] + thresh[1] * IQ
        else:
            t_u = np.inf  
    
    data_to_reject = np.logical_or(data < t_l, data > t_u)
    threshold_matrix[0,0] = t_l
    threshold_matrix[0,1] = t_u
    
    rl_sum = np.sum(data.copy().flatten() < t_l)
    ru_sum = np.sum(data.copy().flatten() > t_u)
    
    if len(np.shape(art.BCT))==2:
        data_to_reject = np.squeeze(np.transpose(data_to_reject,(1,2,0)))
        data = np.squeeze(np.transpose(data,(1,2,0)))
      
    return data_to_reject, rl_sum, ru_sum


def reject_absolute(data, thresh, iq_half=False):
        
    if iq_half:
        if type(thresh)==float or type(thresh)==int:
            thresh= [thresh]
        if len(thresh)>1:
            raise Exception("a unique threshold is required when using iq_half=True")
        if not thresh[0]:
            thresh[0] = np.inf
    else:
        if type(thresh)==float or type(thresh)==int:
            raise Exception("two thresholds are required")
        if len(thresh)!=2:
            raise Exception("two thresholds are required")
        
    threshold_matrix = np.empty((1,2))
    threshold_matrix[:] = np.nan
    
    # reject
    if iq_half:
        t_u = thresh[0]
        t_l = -t_u
    else:
        if thresh[0]:
            t_l = thresh[0]
        else:
            t_l = -np.inf
        if thresh[1]:
            t_u = thresh[1]
        else:
            t_u = np.inf
    
    data_to_reject = np.logical_or(data < t_l, data > t_u)
    threshold_matrix[0,0] = t_l
    threshold_matrix[0,1] = t_u
    
    rl_sum = np.sum(data.copy().flatten() < t_l)
    ru_sum = np.sum(data.copy().flatten() > t_u)
    
    return data_to_reject, rl_sum, ru_sum


def remove_bad_data(raw, bad_data=None, artifact_type='all', verbose=True):
    """
    This function replaces the bad data in the raw data as defined in the artifacts' matrix.
    :param raw: object containing the eeg data
    :param bad_data: replaces the bad data by
                        None : the bad data will be retained
                        'replace by zero': the bad data will be replaced by 0
                        'replace by nan': the bad data will be replaced by 'NaNs
                        'replace by mean': the bad data will be replaced by the mean over all epochs
    :param artifact_type: artifact to be removed
    :param verbose: warning message
    :return:
    """
    if bad_data is None:
        return raw._data.copy()
    
    if not hasattr(raw, 'artifacts'):
        raise Exception("The raw/epochs object should have an 'artifacts' attribute to use the bad_data parameter.")
    
    n_channels, n_samples, n_epochs = get_data_size(raw)
    data_to_remove = np.full(np.shape(raw._data), False)

    # Setting up the indexes of the data to be removed
    if artifact_type in ['all', 'bct']:
        data_to_remove[raw.artifacts.BCT==1] = True
    if artifact_type in ['all', 'BTBC', 'BT'] and hasattr(raw.artifacts,'BT'):
        if np.size(np.shape(raw.artifacts.BT)) == 2:
            bt = np.tile(raw.artifacts.BT, (n_channels, 1))
        elif np.size(np.shape(raw.artifacts.BT)) == 3:
            bt = np.tile(raw.artifacts.BT, (1, n_channels, 1))    
        data_to_remove[bt] = True
    if artifact_type in ['all', 'BTBC', 'BC'] and hasattr(raw.artifacts,'BC'):
        if np.size(np.shape(raw.artifacts.BC)) == 2:
            bc = np.tile(raw.artifacts.BC, (1, n_samples))
        elif np.size(np.shape(raw.artifacts.BC)) == 3:
            bc = np.tile(raw.artifacts.BC, (1, 1, n_samples)) 
        data_to_remove[bc] = True

    # Removing bad data
    n_bad_data = np.sum(data_to_remove)
    eeg_data = raw._data.copy()
    if n_bad_data > 0:
        # Warning message
        if verbose:
            print('\nPercentage of bad data from overall data: ',
                    n_bad_data, ' samples out of ', np.size(data_to_remove),
                    '(', np.round(n_bad_data / np.size(data_to_remove) * 100, 2), '%)')
            message = {
                None: '--> Bad data will be retained',
                'replace by nan': '--> Bad data will be replaced by NaNs',
                'replace by zero': '--> Bad data will be replaced by zeros',
                'replace by mean': '--> Bad data will be replaced by the mean over all epochs',
                'replace by mean  per condition': '--> Bad data will be replaced by the mean per condition'
            }
            print(message[bad_data])
        # Replace bad data
        if bad_data == 'replace by nan':
            eeg_data[data_to_remove] = np.nan
        if bad_data == 'replace by zero':
            eeg_data[data_to_remove] = 0
        if bad_data == 'replace by mean':
            M = np.nanmean(eeg_data, axis=0)
            sdD = np.nanstd(M)
            sdM = np.nanstd(eeg_data)
            M = M * sdD / sdM
            M = np.tile(M, (n_epochs, 1, 1))
            eeg_data[data_to_remove] = M[data_to_remove]
    
    return eeg_data


def _set_reference(raw, bad_data=None):
    """
    Set the reference of the eeg data
    :param raw: object containing the eeg data and related information
    :param save_reference: whether to return the reference values, 'True' | 'False'
    :return:
    """
    n_channels, n_samples, n_epochs = get_data_size(raw)
    
    # Remove bad data
    good_data = remove_bad_data(raw, bad_data=bad_data)

    # Reference to the mean
    if len(np.shape(raw._data))==2:
        reference = np.zeros(good_data.shape[1])
        idx_t_good = np.sum(np.isnan(good_data)==False, axis=0)>1
        reference[idx_t_good] = np.nanmean(good_data[:, idx_t_good], axis=0)
        eeg_data = raw._data.copy() - np.tile(reference, (n_channels, 1))
    
    else:
        eeg_data = raw._data.copy()
        for ep in range(n_epochs):
            reference = np.zeros(good_data.shape[1])
            idx_t_good = np.sum(np.isnan(good_data[ep])==False, axis=0)>1
            reference[idx_t_good] = np.nanmean(good_data[ep][:, idx_t_good], axis=0)
            eeg_data[ep] = eeg_data[ep] - np.tile(reference, (n_channels, 1))
    
    raw._data = eeg_data.copy()
    if np.isnan(reference).any():
        warnings.warn('The reference contains NaN values, that propagate to the EEG data.')

    return raw, reference


def _compute_z_score(raw, bad_data=None):
    """
    Computes for the z-score on the artifacts
    :param raw: object containing the eeg data and related information
    :return: eeg_data: z-score data
             sd: standard deviation
    """

    warnings.filterwarnings("ignore")

    n_channels, n_samples, n_epochs = get_data_size(raw)
    
    # Remove bad data
    good_data = remove_bad_data(raw, bad_data=bad_data)

    # Compute the standard deviation
    if len(np.shape(good_data))==3:
        good_data = np.transpose(good_data,(1,2,0))
    good_data = np.reshape(good_data, (n_channels, n_samples * n_epochs))
    sd = np.nanstd(good_data, axis=1)
    sd[np.isnan(sd)] = np.nanmean(sd)

    # Reshape and replicate variables
    if len(np.shape(good_data))==3:
        sd_ = np.reshape(sd, (1, n_channels, 1))
        sd_ = np.tile(sd_, (n_epochs, 1, n_samples))
    else:    
        sd_ = np.reshape(sd, (n_channels, 1))
        sd_ = np.tile(sd_, (1, n_samples))
    
    # raw data with z-score
    raw._data = np.divide(raw._data.copy(), sd_)

    if np.isnan(sd).any():
        warnings.warn('The standard deviation contains NaN values, that propagate to the EEG data.')
    
    return raw, sd


def _return_data_after_zscore(raw, sd):
    """
    If z_score was first applied on the eeg data, this function retrievs the previous data.
    :param raw: object containing the eeg data
    :param sd: standard deviation (computed by the z-score function)
    :return:
    """
    n_channels, n_samples, n_epochs = get_data_size(raw)
    
    if len(np.shape(raw._data))==3:
        sd_ = np.reshape(sd, (1, n_channels, 1))
        sd_ = np.tile(sd_, (n_epochs, 1, n_samples))
    else:    
        sd_ = np.reshape(sd, (n_channels, 1))
        sd_ = np.tile(sd_, (1, n_samples))
        
    # Compute for the original continuous data
    raw._data = np.multiply(raw._data, sd_) 
    
    return raw

def _return_data_after_referencing(raw, reference):
    """
    If the reference of the eeg was previously set, this function reset the data to its original reference.
    :param raw: object containing the eeg data
    :return:
    """
    n_channels, n_samples, n_epochs = get_data_size(raw)
    if len(np.shape(raw._data))==2:
        temp = np.tile(reference, (n_channels, 1))
        raw._data = raw._data + temp
    else:
        temp = np.tile(reference[:,np.newaxis,:], (1, n_channels, 1))
        raw._data = raw._data + temp
    return raw


