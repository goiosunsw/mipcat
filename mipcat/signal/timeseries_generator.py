import os
import argparse
import logging
from multiprocessing import Pool
import numpy as np
import yaml
from scipy.io import wavfile
import scipy.signal as sig
import librosa as lr
import traceback
import tgt
from timeseries import SampledTimeSeries
from pypevoc.Heterodyne import HeterodyneHarmonic
from pypevoc.SoundUtils import RMSWind 
from tqdm import tqdm, trange

def RMSts(x, sr=1, nwind=1024, nhop=512, windfunc=np.blackman, label='ampl'):
    '''
    Calculates the RMS amplitude amplitude of x, in frames of
    length nwind, and in steps of nhop. windfunc is used as
    windowing function.

    nwind should be at least 3 periods if the signal is periodic.
    '''

    npad = nwind//2
    xp = np.pad(x,npad)
    
    nsam = len(xp)
    ist = 0
    iend = ist+nwind

    t = []
    ret = []

    wind = windfunc(nwind)
    wsum2 = np.sum(wind**2)

    while (iend < nsam):
        thisx_a = xp[ist:iend]
        thisx = thisx_a - np.mean(thisx_a)
        xw = thisx*wind

        ret.append(np.sum(xw*xw/wsum2))
        t.append((float(ist+iend)/2.0-npad)/float(sr))

        ist = ist+nhop
        iend = ist+nwind

    ampl_ts = SampledTimeSeries(v=np.sqrt(np.array(ret)), t=np.array(t), label=label)
    return ampl_ts


def chunked_yin(w, sr, chunk_size=2**18, hop_length=512, label='', **kwargs):
    f0l = []
    nchunks = int(np.ceil(len(w)/chunk_size))
    for ii in range(nchunks):
        wi = w[ii*chunk_size:(ii+1)*chunk_size]
        f0i = lr.yin(wi, sr=sr, hop_length=hop_length, **kwargs)
        if ii > 0:
            f0i = f0i[1:]
        f0l.append(f0i)

    f0v = np.concatenate(f0l)
    t = lr.frames_to_time(np.arange(len(f0v)), sr=sr, hop_length=hop_length)
    
    if label:
        lab = label+'_f0'
    else:
        lab = 'f0'
    tsf = SampledTimeSeries(t=t, v=f0v, label=lab)

    return tsf


def chunked_pyin(w, sr, chunk_size=2**18, hop_length=512, label='', **kwargs):
    f0l = []
    vbooll = []
    vprobl = []
    nchunks = int(np.ceil(len(w)/chunk_size))
    for ii in range(nchunks):
        wi = w[ii*chunk_size:(ii+1)*chunk_size]
        ret = lr.pyin(wi, sr=sr, hop_length=hop_length, **kwargs)
        if ii > 0:
            ret = [r[1:] for r in ret]
        f0l.append(ret[0])
        vbooll.append(ret[1])
        vprobl.append(ret[2])
        logging.info(f'pyin: Finished chunk {ii} of {nchunks}')

    f0v = np.concatenate(f0l)
    vboolv = np.concatenate(vbooll)
    vprobv = np.concatenate(vprobl)
    t = lr.frames_to_time(np.arange(len(f0v)), sr=sr, hop_length=hop_length)
    
    if label:
        lab = label+'_f0'
    else:
        lab = 'f0'
    tsf = SampledTimeSeries(t=t, v=f0v, label=lab)

    if label:
        lab = label+'_isvoiced'
    else:
        lab = 'isvoiced'
    tsb = SampledTimeSeries(t=t, v=vboolv, label=lab)

    if label:
        lab = label+'_voiceprob'
    else:
        lab = 'voiceprob'
    tsp = SampledTimeSeries(t=t, v=vprobv, label=lab)
    
    return tsf, tsb, tsp

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def windowed_quantiles(x, pcts, nwind=1024, nhop=None, wind_func=np.hanning,
                       label=None, progress=True):
    """
    Performs a windowed quantile analysis in signal x

    Windowed samples are weighted according to wind_func
    """
    pcts = np.asarray(pcts)
    if nhop is None:
        nhop = nwind//2

    x = np.atleast_2d(x.T).T
    ret = np.zeros(((x.shape[0]-nwind)//nhop+1,x.shape[1],len(pcts)))
    pwind = wind_func(nwind)

    if progress:
        pbar = tqdm(total=x.shape[0])
    
    for iret, ii in enumerate(range(0, x.shape[0]-nwind, nhop)):
        xw = x[ii:ii+nwind,:]
        for jj in range(x.shape[1]):
            xx = xw[:,jj]
            qts = weighted_quantile(xx, pcts/100, sample_weight=pwind)
            ret[iret,jj,:] = qts
        if progress:
            pbar.update(nhop)
            
    if progress:
        pbar.close()
    return ret


def heterodyne_timeseries(w, sr, f, tf, wind_len=2**10, hop_length=2**9,
                          nharm=5, label=''):
    hh = HeterodyneHarmonic(w, sr=sr, nharm=nharm+1, 
                              f=f, tf=tf, 
                              nwind=wind_len, nhop=hop_length) 
    tsl = []
    for ii in range(nharm):
        v = (hh.camp[:,ii])
        ts = SampledTimeSeries(t=hh.t, v=v, label=f'{label}_h{ii+1:02d}')
        tsl.append(ts)
        if ii>0:
            rel_angles = np.mod(np.angle(hh.camp[:,ii])-
                                np.angle(hh.camp[:,0])*(ii+1)*np.pi,2*np.pi)-np.pi
            ts = SampledTimeSeries(t=hh.t, v=rel_angles,
                                   label=f'{label}_h{ii+1:02d}_rarg')
            tsl.append(ts)
        harm_freqs = hh.f[:-1,ii] - np.diff(np.unwrap(np.angle(hh.camp[:,ii])))/(hh.nhop/sr)/2/np.pi
        ts = SampledTimeSeries(t=(hh.t[:-1]+hh.t[1:])/2, v=harm_freqs,
                               label=f'{label}_h{ii+1:02d}_freq')
        tsl.append(ts)

    return tsl, hh


def process_wav_generic(w, sr, wind_len=2**10,
                hop_length=2**9, fmin=60, fmax=1500,
                f0_win_len=1024, f0_chan=0, 
                chan_labels=None, pcts=[5,25,50,75,95]):

    nchans = w.shape[1]

    wpct = windowed_quantiles(w, pcts, nhop=hop_length, nwind=wind_len)
    #tpct = (np.arange(0, w.shape[0] - wind_len, hop_length) - wind_len//2) / sr
    tpct = wpct.shape[0] * hop_length/sr
    
    if chan_labels is None:
        chan_labels = [f'ch{ii:02d}' for ii in range(nchans)]

    ww = w[:,f0_chan] 

    # f0
    freq_ts = chunked_yin(ww, sr=sr, hop_length=hop_length,
                          frame_length=f0_win_len, fmin=fmin, fmax=fmax,
                          label=chan_labels[f0_chan]) 

    ret = [freq_ts]

    for ii in range(nchans):
        ww = w[:, ii].astype('f')
        try:
            chlabel = chan_labels[ii]
        except IndexError:
            chlabel = None
        if chlabel is None:
            continue
        # RMS Amplitude

        ampl_ts = RMSts(ww, sr=sr, nwind=wind_len, nhop=hop_length, label=f'{chlabel}_ampl')

        # DC
        n_orig = w.shape[0]
        n_res = int(np.floor(n_orig/wind_len))
        wr = sig.resample(ww[:n_res*wind_len], n_res)
        dc_ts = SampledTimeSeries(wr, dt=wind_len/sr, label=f'{chlabel}_dc')

        # Spectral centroid 
        centroid = lr.feature.spectral_centroid(y=ww, sr=sr,
                                                hop_length=hop_length)[0]
        x_cent_ts = SampledTimeSeries(v=centroid, 
                                      t=lr.frames_to_time(np.arange(len(centroid)),
                                                          sr=sr,
                                                          hop_length=hop_length),
                                      label=f'{chlabel}_cent')
        this_tsl = [ampl_ts, dc_ts, x_cent_ts]
        ret.extend(this_tsl)

        # Heterodyne
        hts, _ = heterodyne_timeseries(ww, sr=sr, f=freq_ts.v, tf=freq_ts.t,
                                    label=chlabel)
        ret.extend(hts)

        # Percentiles
        for jj, pct in enumerate(pcts):
            ret.append(SampledTimeSeries(t=tpct, v=wpct[:,ii,jj],
                                         label=f'{chlabel}_pct{pct:02d}'))
        
     
    return ret


def read_wav_chan_info(filename, chan_info=None):

    w, sr = lr.load(filename, mono=False, sr=None)
    w = w.T
    
    sigs = {}
    
    for ii, chinfo in enumerate(chan_info):
        try:
            keep = chinfo['keep']
        except KeyError:
            keep = True

        # apply gain/offset
        try:
            gain = chinfo['gain']
        except KeyError:
            gain = 1.0
            
        try:
            offset = chinfo['offset']
        except KeyError:
            offset = 0.0

        label = chinfo['name']
        # print(label,gain)
        
        if keep:
            try:
                ww = (w[:,ii].astype('f') + offset)*gain
            except IndexError:
                logging.warning(f"Channel {ii} does not exist")
                continue
            sigs[label] = ww


    return sigs, sr


def process_wav_chan_info(w, sr, wind_len=2**10,
                hop_length=2**9, fmin=60, fmax=1500,
                f0_win_len=1024, nharm=5,
                f0_chan='barrel', 
                chan_info=None, pcts=[5,25,50,75,95]):

    sigs = []
    chan_labels = []
    print(f0_chan)
    for ii, chinfo in enumerate(chan_info):
        try:
            keep = chinfo['keep']
        except KeyError:
            keep = True

        # apply gain/offset
        try:
            gain = chan_info[ii]['gain']
        except KeyError:
            gain = 1.0
            
        try:
            offset = chan_info[ii]['offset']
        except KeyError:
            offset = 0.0

        label = chinfo['name']
        
        if keep:
            try:
                ww = (w[:,ii].astype('f') + offset)*gain
            except IndexError:
                logging.warning(f"Channel {ii} does not exist")
                break
            sigs.append(ww)
            chan_labels.append(label)

            if label == f0_chan:
                wf = ww
        
    w = np.asarray(sigs).T
    nchans = w.shape[1]

    wpct = windowed_quantiles(w, pcts, nhop=hop_length, nwind=wind_len)
    #tpct = (np.arange(0, w.shape[0] - wind_len, hop_length) - wind_len//2) / sr
    tpct = np.arange(wpct.shape[0])*hop_length/sr

    # f0
    freq_ts = chunked_yin(wf, sr=sr, hop_length=hop_length,
                          frame_length=f0_win_len, fmin=fmin, fmax=fmax,
                          label=f0_chan) 

    ret = [freq_ts]
    htsd = {}

    for ii in range(nchans):
        chlabel = chan_labels[ii]

        ww = w[:,ii]
        # RMS Amplitude
        # cme = lr.feature.rms(y=ww, frame_length=wind_len, hop_length=hop_length, 
        #                      center=True, pad_mode='reflect')[0]
        # ampl_ts = SampledTimeSeries(v=cme, t=lr.frames_to_time(np.arange(len(cme)),
        #                             sr=sr, hop_length=hop_length),
        #                             label=f'{chlabel}_ampl')
        ampl_ts = RMSts(ww, sr=sr, nwind=wind_len, nhop=hop_length, label=f'{chlabel}_ampl')

        # DC
        n_orig = w.shape[0]
        n_res = int(np.floor(n_orig/wind_len))
        wr = sig.resample(ww[:n_res*wind_len], n_res)
        dc_ts = SampledTimeSeries(wr, dt=wind_len/sr, label=f'{chlabel}_dc')

        # Spectral centroid 
        centroid = lr.feature.spectral_centroid(y=ww, sr=sr,
                                                hop_length=hop_length)[0]
        x_cent_ts = SampledTimeSeries(v=centroid, 
                                      t=lr.frames_to_time(np.arange(len(centroid)),
                                                          sr=sr,
                                                          hop_length=hop_length),
                                      label=f'{chlabel}_cent')
        this_tsl = [ampl_ts, dc_ts, x_cent_ts]
        ret.extend(this_tsl)

        # Heterodyne
        hts, hets = heterodyne_timeseries(ww, sr=sr, f=freq_ts.v, tf=freq_ts.t,
                                    label=chlabel, nharm=nharm)
        ret.extend(hts)
        htsd[chlabel] = hets

                

        # Percentiles
        for jj, pct in enumerate(pcts):
            ret.append(SampledTimeSeries(t=tpct, v=wpct[:,ii,jj],
                                         label=f'{chlabel}_pct{pct:02d}'))
        
    ## Derived signals:
    # Vocal tract / mouthpiece ratio
    try:
        camp_vt = htsd['mouth'].camp/htsd['mouthpiece'].camp
        hh = htsd['mouth']
        for ii in range(nharm):
            v = camp_vt[:,ii]
            ts = SampledTimeSeries(t=hh.t, v=v, label=f'vt_h{ii+1:02d}')
            ret.append(ts)
    except KeyError:
        pass
     
    return ret


# def process_wav(filename, channels, wind_len=2**10,
#                 hop_length=2**9, fmin=60, fmax=1500,
#                 f0_win_len=1024, base_chan='barrel'):
#     sr,w = wavfile.read(filename)
#     logging.info(f'Read file {filename}')

#     barrel_idx = [x['name'] for x in channels].index('barrel')
#     mouth_idx = [x['name'] for x in channels].index('mouth')
#     reed_idx = [x['name'] for x in channels].index('reed')
#     ext_idx = [x['name'] for x in channels].index('external') 
#     base_idx = [x['name'] for x in channels].index(base_chan)
    
#     # f0 is extracted from the base channel depending on instrument used
#     ww = w[:, base_idx].astype('f')
#     freq_ts = chunked_yin(ww, sr=sr, hop_length=hop_length,
#                           frame_length=f0_win_len, fmin=fmin, fmax=fmax,
#                           label=base_chan) 
#     # voice_ts = SampledTimeSeries(v=voiced, t=lr.frames_to_time(np.arange(len(f0)),
#     #                             sr=sr,hop_length=hop_length),label='voice_prob')

#     ww = w[:, barrel_idx].astype('f')
#     cme = lr.feature.rms(y=ww, frame_length=wind_len, hop_length=hop_length, 
#                          center=True, pad_mode='reflect')[0]
#     ampl_ts = SampledTimeSeries(v=cme, t=lr.frames_to_time(np.arange(len(cme)),
#                                 sr=sr, hop_length=hop_length), label='barrel_ampl')

#     # mouth: dc
#     ww = w[:, mouth_idx].astype('f')
#     n_orig = w.shape[0]
#     n_res = int(np.floor(n_orig/wind_len))
#     wr = sig.resample(ww[:n_res*wind_len], n_res)
#     mouth_ts = SampledTimeSeries(wr, dt=wind_len/sr, label='mouth_dc')

#     # mouth: ac
#     wl = sig.resample(wr, n_res*wind_len)
#     wh = ww[:len(wl)]-wl
#     cme = lr.feature.rms(y=wh, frame_length=wind_len, hop_length=hop_length, 
#                          center=True, pad_mode='reflect')[0]
#     m_ampl_ts = SampledTimeSeries(v=cme, t=lr.frames_to_time(np.arange(len(cme)),
#                                   sr=sr, hop_length=hop_length), label='mouth_ac')
 
#     # reed
#     ww = w[:, reed_idx].astype('f')
#     n_orig = w.shape[0]
#     n_res = int(np.floor(n_orig/wind_len))
#     wr = sig.resample(ww[:n_res*wind_len], n_res)
#     reed_ts = SampledTimeSeries(wr, dt=wind_len/sr, label='reed_dc')

#     # external: amplitude, freq and centroid
#     ww = w[:, ext_idx].astype('f')

#     cme = lr.feature.rms(y=ww, frame_length=wind_len, hop_length=hop_length, 
#                          center=True, pad_mode='reflect')[0]
#     x_ampl_ts = SampledTimeSeries(v=cme, t=lr.frames_to_time(np.arange(len(cme)),
#                                   sr=sr, hop_length=hop_length), 
#                                   label='external_ampl')

#     centroid = lr.feature.spectral_centroid(y=ww, sr=sr,
#                                             hop_length=hop_length)[0]
#     x_cent_ts = SampledTimeSeries(v=centroid, 
#                                   t=lr.frames_to_time(np.arange(len(centroid)),
#                                                       sr=sr,
#                                                       hop_length=hop_length),
#                                   label='external_cent')

#     # quantiles
#     qa = windowed_quantiles(w, pcts)

 
#     return [freq_ts, ampl_ts, mouth_ts, m_ampl_ts, reed_ts, x_ampl_ts, x_cent_ts]


def process_wav(filename, channels, wind_len=2**10,
                hop_length=2**9, fmin=60, fmax=1500,
                f0_win_len=1024, base_chan='barrel'):
    
    chan_labels = [x['name'] for x in channels]
    base_chno = chan_labels.index(base_chan)

    w, sr = lr.load(filename, mono=False, sr=None)
    w = w.T
    tsl = process_wav_chan_info(w, sr=sr, wind_len=wind_len,
                               hop_length=hop_length, f0_chan=base_chan,
                               chan_info=channels)

    return tsl


def ts_to_pickle(tsdict, filename):
    tsdata = {}
    uno = 1
    for ts in tsdict:
        tsd = {'dt':ts.dt,
               't0':ts.t_start,
               'v':ts.v}
        if len(ts.label)>0:
            label = ts.label
        else:
            label = f'Unnamed {uno}'
            uno += 1
        tsdata[label]=tsd
    import pickle
    with open(filename,'wb') as f:
        pickle.dump(tsdata, f)


def ts_from_pickle(filename):
    tss = []
    import pickle
    with open(filename,'rb') as f:
        tsl = pickle.load(f)
        for label, ts in tsl.items():
            tss.append(SampledTimeSeries(ts['v'],dt=ts['dt'],t_start=ts['t0'],label=label))
    return tss

class RunIterator(object):
    def __init__(self, rundict, tiername='clip', runsheet_root=None, output_root=None):
        if runsheet_root is None:
            try:
                self.root_folder = rundict['root_folder']
            except KeyError:
                self.root_folder = '.'
        else:
            self.root_folder = runsheet_root
        self.rundict = rundict
        try:
            self.data_folder = rundict['data_folder']
        except KeyError:
            self.data_folder = '.'

        self.tiername = tiername
   
    def iter_runs(self):
        for run in self.rundict['runsheets']:
            yield run

    def iter_files(self):
        for  run in self.iter_runs():
            yaml_path = os.path.join(self.root_folder, run['runsheet'])
            with open(yaml_path) as f:
                yml = yaml.full_load(f)
                root_dir = os.path.join(self.data_folder, yml['root_dir'])
                channels = yml['channel_desc']
                for wfd in yml['wave_files']:
                    try:
                        chdesc_lbl = wfd['channels']
                    except KeyError: 
                        chdesc_lbl = 'default'

                    channel_desc = channels[chdesc_lbl]
                    d = {'channels':channel_desc}
                    d['root_dir'] = root_dir
                    d.update(run)
                    d.update(wfd)
                    yield d 

    def iter_file_regions(self):
        for filedict in self.iter_files():
            tg_filename = filedict['excerpt_textgrid']
            
            tg = tgt.read_textgrid(os.path.join(filedict['root_dir'],
                                                tg_filename))
            tier = tg.get_tier_by_name(self.tiername)
            annots = []
            for annot in tier.annotations:
                d = {}
                d['start_time'] = annot.start_time
                d['end_time'] = annot.end_time
                d['label'] = annot.text
                annots.append(d)
            yield filedict, annots

    def iter_regions(self):
        for filedict in self.iter_files():
            tg_filename = filedict['excerpt_textgrid']
            
            tg = tgt.read_textgrid(tg_filename)
            tier = tg.get_tier_by_name(self.tiername)
            for annot in tier.annotations:
                d = {}
                d.update(filedict)
                d['start_time'] = annot.start_time
                d['end_time'] = annot.end_time
                d['label'] = annot.text
                yield d


def run_iterator_from_yaml(filename, output=None):
    runsheet_root, _ = os.path.split(os.path.abspath(filename))
    with open(filename) as f:
        yml = yaml.full_load(f)
        return RunIterator(yml, runsheet_root=runsheet_root, output_root=output)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="WAV filename or YAML runsheet")
    parser.add_argument("-r", '--recalc', action='store_true', 
                        help="Recalculate if pickle present")
    parser.add_argument("-b", '--base-chan',  
                        help="base channel for pitch analysis")
    parser.add_argument("-D", '--dry-run', action='store_true', 
                        help="Do not actually process")
    parser.add_argument("-P", '--no-parallel', action='store_true', 
                        help="Do not run parallel jobs")
    parser.add_argument("-w", "--wind-msec", default=20,
                        help="Window length for most analysis in msec")
    parser.add_argument("-p", "--hop-msec", default=10,
                        help="hop length for all analysis in msec")
    parser.add_argument("-f", "--f0-wind-msec", default=20,
                        help="frame length for f0 in msec")
    parser.add_argument("-n", "--fmin", default=50,
                        help="minimum f0")
    parser.add_argument("-x", "--fmax", default=2000,
                        help="maximum f0")
    parser.add_argument("-o", "--output",
                        help="output file name or folder (for runsheet)")
    parser.add_argument("-c", '--config',
                        help="YAML config file")

    return parser.parse_args()


class Processor():
    def __init__(self, recalc=False):
        self.recalc = recalc

    def extract_and_pickle(self, file_dict, output=None):

        logging.info(file_dict)
        
        tune = file_dict['tune']
        wfile = os.path.join(file_dict['root_dir'], file_dict['filename'])
        if output is None:
            output = file_dict['root_dir']

        pickfile = os.path.join(output, os.path.splitext(file_dict['filename'])[0]) + '_ts.pickle'

        try:
            os.stat(pickfile)
            if not self.recalc:
                logging.info(f'{pickfile} exists: skipping')
                return
        except FileNotFoundError:
            pass

        channel_desc = file_dict['channels']
        base_chan = 'barrel'
        try:
            if file_dict['instrument'] != 'lab':
                base_chan = 'external'
        except KeyError:
            pass

        try:
            ts = process_wav(wfile, channel_desc, base_chan=base_chan)
        except Exception:
            traceback.print_exc()
            return
        logging.info('Finished processing {}'.format(wfile))

        ts_to_pickle(ts, pickfile)

    def dry_run(self, file_dict):

        logging.info(file_dict)
        
        tune = file_dict['tune']
        wfile = os.path.join(file_dict['root_dir'], file_dict['filename'])
        pickfile = os.path.splitext(wfile)[0] + '_ts.pickle'

        logging.info(f'wavefile is {wfile}')
        logging.info(f'pickle stored to {pickfile}')
	
        try:
            os.stat(pickfile)
            if not self.recalc:
                logging.info(f'{pickfile} exists: skipping')
                return
        except FileNotFoundError:
            pass

        try:
            base_chan = args.base_chan
        except AttributeError:
            base_chan = None

        channel_desc = file_dict['channels']
        base_chan = 'barrel'
        try:
            if file_dict['instrument'] != 'lab':
                base_chan = 'external'
        except KeyError:
            pass

        logging.info(f'base channel is {base_chan}')
        try:
            pass
        except Exception:
            traceback.print_exc()
            return

        #ts_to_pickle(ts, pickfile)

def process_runsheet(ri):

    pool = Pool()
    p = Processor(recalc=recalc)

    if args.dry_run:
        pfunc = p.dry_run
    else:
        pfunc = p.extract_and_pickle
    
    if not args.no_parallel:
        pool.map(pfunc, ri.iter_files())
    else:
        for file_dict in ri.iter_files():
            pfunc(file_dict)

            
def process_single(filename, configfilename, output=None):
    with open(configfilename, 'r') as f:
        config = yaml.full_load(f)
        
    found = False
    channels = config['channel_desc']
    chdesc_lbl = 'default'
    for wfd in config['wave_files']:
        try:
            chdesc_lbl = wfd['channels']
        except KeyError: 
            chdesc_lbl = 'default'

    if not found:
        import sys
        print(f'Config for channels in {filename} not found. Using "default"')
        
    channel_desc = channels[chdesc_lbl]

    if output:
        pickfile = output
    else:
        pickfile = os.path.splitext(filename)[0]+'_ts.pickle'
    ts = process_wav(filename, channel_desc)
    ts_to_pickle(ts, pickfile)
    

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    args=parse_arguments()
    infile = args.infile
    recalc = args.recalc
    try:
        try:
            output = args.output
        except AttributeError:
            output = None
        ri = run_iterator_from_yaml(infile, output=output)
        process_runsheet(ri)
    except (ValueError, UnicodeDecodeError):
        try:
            output = args.output
        except AttributeError:
            output = os.path.splitext(infile) 
        process_single(infile, args.config, output=output)
