import sys
import pandas as pd
import numpy as np
import scipy.signal as sig
import yaml
import logging
import os
import traceback
import argparse
import gc

from mipcat.signal.timeseries_generator import read_wav_chan_info, ts_from_pickle

pctl = [1,5,25,50,75,95,99]
logging.basicConfig(level=logging.INFO)

class TransientProcessor(object):
    def __init__(self,  notes, nwind=2**10, nhop=2**7, chan_set=None, cutoff_hz=50, 
                    slope_db_sec=40, release_slope_db_sec = 80, slow_slope_db_sec=4):
        self.notes  = notes
        self.nhop= nhop
        self.nwind = nwind
        self.window = 'hann'
        self.channel_desc = chan_set
        self.cutoff_hz = cutoff_hz
        self.slope_db_sec = slope_db_sec 
        self.release_slope_db_sec = release_slope_db_sec
        self.slow_slope_db_sec = slow_slope_db_sec
        self.use_chans = ['barrel','external','farfield']

    def process_file(self, filename, chan_set_label):
        #filepath = os.path.join(self.original_root_dir, filename)
        filepath = self.original_root_dir + filename
        sigpath = self.ts_root_dir + filename.replace('.wav','_ts.pickle')
        chans, sr = read_wav_chan_info(filepath, 
                                       self.channel_desc[chan_set_label])

        try:
            self.msig = chans['mouth']
            self.tsig = chans['tongue']
            self.rsig = chans['reed']
        except KeyError:
            self.msig = None
            self.tsig = None
            self.rsig = None

        tsl = ts_from_pickle(sigpath)
        try:
            self.mouth_dc_ts = [ts for ts in tsl if ts.label=='mouth_dc'][0]
            self.reed_dc_ts = [ts for ts in tsl if ts.label=='reed_dc'][0]
        except IndexError:
            self.mouth_dc_ts = None
            self.reed_dc_ts = None
        del tsl

        file_mask = self.notes.wavfile.str.replace('\\','/',regex=False).str.contains(filename)
        notes = self.notes[file_mask].sort_values('start')

        logging.info(f"{sum(file_mask)} notes")
        self.sr = sr
        self.thop = self.nhop/sr
        chpcts = {}

        for lab, wch in chans.items():
            pcts = np.percentile(wch,pctl)
            # print(pcts)
            chpcts[lab] = {p:v for p,v in zip(pctl,pcts)}

        self.chpcts = chpcts
    
        try:
            self.m_thr = chpcts['mouth'][1] + (chpcts['mouth'][95]-chpcts['mouth'][1])/10
        except KeyError:
            self.m_thr = 0

        fundmin_hz = .8*notes.exp_freq.min()
        fundmax_hz = 1.25*notes.exp_freq.max()
        
        for ch_name in self.use_chans:
            try:
                ww = chans[ch_name]
                break
            except KeyError:
                logging.info(f'Channel {ch_name} not found')
                continue

        del chans
        self.ch_name = ch_name
        fss, tss, ss = sig.spectrogram(ww,fs=sr,nperseg=self.nwind,
                                    noverlap=self.nwind-self.nhop,
                                    window=self.window)
        del ww 
        self.tss = tss
        
        fundidx = np.flatnonzero((fss>fundmin_hz)&(fss<fundmax_hz))
        self.funddb = 10*np.log10(np.sum(ss[fundidx,:],axis=0))

        cutidx = np.flatnonzero(fss>self.cutoff_hz)[0]
        self.hpdb = 10*np.log10(np.sum(ss[cutidx:,:],axis=0))

        self.prev_max_idx=0

        self.sl = self.slope_db_sec*self.thop
        self.small_sl = self.slow_slope_db_sec*self.thop

        notes = self.notes[file_mask].sort_values('start')

        for irow, row in  notes.iterrows():
            try:
                self.process_attack(row)
            except Exception:
                print(f'Note {irow} at {row.start} channel {ch_name} -- attack')
                row['trans_err'] = traceback.format_exc()
                traceback.print_exc()
        
    
        self.prev_max_idx = len(self.hpdb)-1
        self.rel_sl = self.release_slope_db_sec*self.thop
        notes = self.notes[file_mask].sort_values('start',ascending=False)
        for irow, row in notes.iterrows():
            try:
                self.process_release(row)
            except Exception:
                print(f'Note {irow} at {row.start} channel {ch_name} -- release')
                row['trans_err'] = traceback.format_exc()
                traceback.print_exc()


        self.prev_max_idx = 0
        notes = self.notes[file_mask].sort_values('start')
        if self.msig is not None:
            for irow, row in notes.iterrows():
                try:
                    self.add_sensor_sig_info(row)
                except Exception:
                    row['trans_err'] = traceback.format_exc()
                    traceback.print_exc()
            
        del tss, fss, ss
        gc.collect()
        return notes

    def process_attack(self, row):
        
        irow = row.name
        start_idx = int(row.start/self.thop)
        end_idx = int(row.end/self.thop)
        max_idx = start_idx+np.argmax(self.hpdb[start_idx:end_idx])

        self.notes.loc[irow,self.ch_name+'_note_max_sec'] = self.tss[(max_idx)]
        svec = self.hpdb[self.prev_max_idx:max_idx]-self.small_sl*np.arange(max_idx-self.prev_max_idx)
        try:
            sidx = np.flatnonzero(svec>svec[0])[-1]
            if sidx>1:
                svec = svec[:sidx]
        except IndexError:
            pass
        prev_min = self.prev_max_idx+np.argmin(svec)
        self.notes.loc[irow,self.ch_name+'_prev_min_sec'] = self.tss[prev_min]
        attack_start = prev_min
        
        pks = sig.find_peaks(-self.hpdb[self.prev_max_idx:max_idx])[0] + self.prev_max_idx
        try:
            prev_local_min = pks[np.argmin(np.abs(pks-start_idx))]
            self.notes.loc[irow,self.ch_name+'_prev_local_min_sec'] = self.tss[prev_local_min]
            attack_start = prev_local_min
        except ValueError:
            pass
        svec = self.hpdb[prev_min:max_idx]-self.sl*np.arange(max_idx-prev_min)
        try:
            sidx = np.flatnonzero(svec<svec[-1])[0]
        except IndexError:
            sidx=0
        svec = svec[sidx:]
        attack_corner = prev_min+sidx+np.argmax(svec)
        self.notes.loc[irow,self.ch_name+'_attack_cnr_sec'] = self.tss[attack_corner]
        
        pks = pks[pks<attack_corner]
        try:
            prev_local_min = pks[np.argmin(np.abs(pks-start_idx))]
            #thisd['prev_local_min_sec'] = tss[prev_local_min]
            self.notes.loc[irow,self.ch_name+'_prev_min_before_crn_sec'] = self.tss[prev_local_min]
            attack_start = prev_local_min
        except ValueError:
            pass


        self.notes.loc[irow,self.ch_name+'_attack_start_val'] = self.hpdb[attack_start]
        self.notes.loc[irow,self.ch_name+'_attack_corner_val'] = self.hpdb[attack_corner]
        
        attack = self.hpdb[attack_start:attack_corner]
        if len(attack)>1:
            attack_sl = np.diff(attack)
            self.notes.loc[irow,self.ch_name+'_attack_max_slope'] = np.max(attack_sl)/self.thop
            self.notes.loc[irow,self.ch_name+'_attack_med_slope'] = np.median(attack_sl)/self.thop
        
        self.prev_max_idx = max_idx

    def process_release(self,row):
        irow = row.name
        start_idx = int(row.start/self.thop)
        end_idx = int(row.end/self.thop)
        max_idx = int(row[self.ch_name+'_note_max_sec']/self.thop)
        
        svec = self.hpdb[max_idx:self.prev_max_idx]+self.small_sl*np.arange(self.prev_max_idx-max_idx)
        next_min = max_idx+np.argmin(svec)
        self.notes.loc[irow,self.ch_name+'_following_min_sec'] = self.tss[next_min]
        
        pks = sig.find_peaks(-self.hpdb[max_idx:self.prev_max_idx])[0] + max_idx
        release_end_idx = next_min
        try:
            next_local_min = pks[np.argmin(np.abs(pks-end_idx))]
            self.notes.loc[irow,self.ch_name+'_following_local_min_sec'] = self.tss[next_local_min]
            release_end_idx = next_local_min
        except ValueError:
            pass
        
        svec = self.hpdb[max_idx:release_end_idx]+self.rel_sl*np.arange(release_end_idx-max_idx)
        try:
            attack_corner = max_idx+np.argmax(svec)
            self.notes.loc[irow,self.ch_name+'_release_cnr_sec'] = self.tss[attack_corner]
        except ValueError:
            pass
        self.prev_max_idx = max_idx

    def add_sensor_sig_info(self, row):
        irow = row.name
        start_samp = int(self.sr*row.start)
        end_samp = int(self.sr*row.end)
        max_samp = np.nan
        cnr_samp = np.nan
        min_samp = np.nan
        for ch_name in self.use_chans:
            if np.isnan(max_samp):
                max_samp = int(self.sr*row[ch_name+'_note_max_sec'])
            if np.isnan(cnr_samp):
                cnr_samp = int(self.sr*row[ch_name+'_attack_cnr_sec'])
            if np.isnan(min_samp):
                try:
                    min_samp = int(self.sr*row[ch_name+'_prev_min_before_crn_sec'])
                except ValueError:
                    try:
                        min_samp = int(self.sr*row[ch_name+'_prev_local_min_sec'])
                    except ValueError:
                        min_samp = int(self.sr*row[ch_name+'_prev_min_sec'])
        ms = self.msig[self.prev_max_idx:max_samp]
        tong_srch_idx_start = self.prev_max_idx
        tong_srch_t_start = tong_srch_idx_start/self.sr
        try:
            m_cr = np.flatnonzero((ms[:-1]<self.m_thr)&(ms[1:]>=self.m_thr))[-1] + self.prev_max_idx
            self.notes.loc[irow,'mouth_rise_sec'] = m_cr/self.sr
            tong_srch_idx_start = m_cr
        except IndexError:
            pass
        
        try:
            trel=np.argmax(self.tsig[tong_srch_idx_start:cnr_samp])+tong_srch_idx_start
            self.notes.loc[irow,'tongue_release_sec'] = trel/self.sr
            self.notes.loc[irow,'tongue_val'] = self.tsig[trel]
        except ValueError:
            pass

        for name, psig in [('mouth_p',self.msig),('reed', self.rsig)]:
            # Average mouth/reed during attack
            try:
                self.notes.loc[irow,f'attack_{name}_avg'] = np.mean(psig[min_samp:cnr_samp])
            except ValueError:
                pass
        for name, ts in [('mouth_p',self.mouth_dc_ts),('reed', self.reed_dc_ts)]:
            # Position of maximum of mouth/reed
            try:
                pmmaxidx = np.argmin(psig[start_samp:end_samp]) + start_samp
                self.notes.loc[irow,f'{name}_max_pos'] = ts.max_time(from_time=row.start, to_time=row.end)
                self.notes.loc[irow,f'{name}_max_val'] = ts.max(from_time=row.start, to_time=row.end)
            except ValueError:
                pass
            
    
        tong_srch_t_end = max_samp/self.sr
        try:
            self.notes.loc[irow,f'mouth_p_min_pos'] = self.mouth_dc_ts.min_time(from_time=tong_srch_t_start, to_time=tong_srch_t_end)
            self.notes.loc[irow,f'mouth_p_min_val'] = self.mouth_dc_ts.min(from_time=tong_srch_t_start, to_time=tong_srch_t_end)
        except ValueError:
            pass
        try:
            self.notes.loc[irow,f'reed_min_pos'] = self.reed_dc_ts.min_time(from_time=row.start, to_time=row.end)
            self.notes.loc[irow,f'reed_min_val'] = self.reed_dc_ts.min(from_time=row.start, to_time=row.end)
        except ValueError:
            pass
            
        self.prev_max_idx = max_samp

    def init_csv(self, csvfile,  original_root_dir='.', ts_root_dir='.'):
        
        
        df = pd.read_csv(csvfile, index_col=0)
        self.original_root_dir = original_root_dir
        self.ts_root_dir = ts_root_dir
        self.file_list = df

    def process_csv(self, csvfile,  original_root_dir='.', ts_root_dir='.'):

        self.init_csv( csvfile,  original_root_dir=original_root_dir, ts_root_dir=ts_root_dir)
        for irow, row in self.file_list.iterrows():
            #print(row)
            wav_path = original_root_dir + '/' + row.filename
            logging.info("Processing "+wav_path)
            try:
                rtrans = self.process_file(row.filename, chan_set_label=row.channel_set)
            except Exception as e:
                traceback.print_exc()
                
            

        return self.notes

    
def init_processor( csv_file, chan_desc_file, note_file, melody_file, original_root_dir='.', 
                   ts_root_dir='.',test=False):
    notedf = pd.read_csv(note_file, index_col=0)
    meldf = pd.read_csv(melody_file, index_col=0)
    meldf.index.name = 'gl_mel_note_idx'
    notes = notedf.merge(meldf,on=['tune','note_id'])
    with open(chan_desc_file, 'r') as f:
        chan_desc = yaml.safe_load(f)

    tp = TransientProcessor(notes, chan_set=chan_desc)
    tp.init_csv(csv_file, original_root_dir=original_root_dir, ts_root_dir=ts_root_dir)
    return tp

def parse_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("filename", help="CSV file with note segmentation")
    ap.add_argument("-c", "--csv-runsheet", help="CSV runsheet")
    ap.add_argument("-d", "--channel-desc", help="YAML channel description")
    ap.add_argument("-m", "--melody-file", help="CSV melody file (generated by build_note_database)")
    ap.add_argument("-o", "--output", help="Output file (pickle)")
    ap.add_argument("-r", "--root", help="root folder for the original recordings")
    ap.add_argument("-i", "--intermediate", help="root folder for intermediate calculations")
    return ap.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)
    # csv_file = "E:/Data/2021_SensorClarinet/original/wav_melody_list_manual.csv"
    # desc_file = "C:/Users/goios/Devel/sensor_clarinet_processing/runsheets/channel_desc.yaml"
    # note_file = "C:/Users/goios/cloudstor/Research/EmotionInMusic/SensorClarinetTests/played_notes.csv"
    # melody_file = "C:/Users/goios/cloudstor/Research/EmotionInMusic/SensorClarinetTests/melody_scores.csv" 
    csv_file = "E:/Data/2021_SensorClarinet/original/wav_melody_list_manual.csv"
    # csv_file = sys.argv[1]
    desc_file = "C:/Users/goios/Devel/sensor_clarinet_processing/runsheets/channel_desc.yaml"
    #note_file = "./played_notes.csv"
    note_file = sys.argv[1]
    melody_file = "./melody_scores.csv" 
    original_root_dir = "E:/Data/2021_SensorClarinet/original/"
    intermediate_root_dir = "E:/Data/2021_SensorClarinet/intermediate/"

    csv_file = args.csv_runsheet
    desc_file = args.channel_desc
    melody_file = args.melody_file
    if args.output is None:
        output_file = "notes_w_transients.pandas.pickle"
    else:
        output_file = args.output
    original_root_dir = args.root
    if args.intermediate is None:
        intermediate_root_dir = os.path.join(os.path.realpath(original_root_dir+'/..'),"intermediate")
        logging.info(f"Intermediate folder set to {intermediate_root_dir}")
    else:
        intermediate_root_dir = args.intermediate
    
    note_file = args.filename

    tp = init_processor(csv_file, desc_file, note_file, melody_file, 
                        original_root_dir=original_root_dir,
                        ts_root_dir=intermediate_root_dir)

    tdf =  tp.process_csv(csv_file, original_root_dir=original_root_dir,ts_root_dir=intermediate_root_dir)

    tdf.to_pickle(output_file)