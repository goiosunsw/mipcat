import argparse
import os
import sys
import re
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm


#notedf = pd.read_csv("played_notes.csv",index_col=0)
notedf = pd.read_pickle(sys.argv[1]) #pd.read_pickle('transients.pandas.pickle')

meldf = pd.read_csv("melody_scores.csv",index_col=0)
verdf = pd.read_csv("versions_played.csv",index_col=0)
wavdf = pd.read_csv('reed_lims.csv')
wavdf = wavdf[['ext_wavfile','reed_high_max','reed_low_max']]

# Check for duplicate versions
dups = set(verdf.index).difference(verdf.drop_duplicates().index)
print(f'Duplicate version numbers: {str(dups)}')


# for col in notedf.columns[notedf.columns.str.contains('f0')]:
#    notedf[col] = notedf[col]

# Correct pathnames
verdf.wavfile = verdf.wavfile.str.replace('\\','/',regex=False)
notedf.wavfile = notedf.wavfile.str.replace('\\','/',regex=False)

# Fetch f0 from whatever source is best
base_chan = 'barrel'
altcols = (notedf.columns.str.startswith(base_chan+'_f0') | 
             notedf.columns.str.startswith(base_chan+'_attack') |
             notedf.columns.str.startswith(base_chan+'_note_max') | 
           notedf.columns.str.startswith(base_chan+'_release') | 
          notedf.columns.str.startswith(base_chan+'_prev') | 
          notedf.columns.str.startswith(base_chan+'_follow'))

newcols = set([])
for col in notedf.columns[altcols]:
    new_label = col.replace(base_chan+'_','')
    notedf[new_label] = notedf[col]
    newcols.add(new_label)

f0cols = notedf.columns[notedf.columns.str.contains('f0')]
def fsuff(x):
    idx = x.find('_f0')
    if idx > -1:
        return x[:idx]
    else:
        return ''
basechans =  set(fsuff(x) for x in f0cols)
basechans = ['farfield','external']
for ch in basechans:
    idx = notedf.f0_avg.isna()
    for col in newcols:
        notedf.loc[idx,col] = notedf.loc[idx,ch+'_'+col]

# get a unique external amplitude average
# preferentially from far-field, otherwise from external
for suff in ['ampl_avg', 'ampl_var']:
    notedf[f'ext_{suff}'] = notedf[f'farfield_{suff}']
    notedf.loc[notedf.ext_ampl_avg.isna(),f'ext_{suff}'] = notedf.loc[notedf.ext_ampl_avg.isna(),f'external_{suff}']

# Correct eg names 
verdf['eg']=verdf.ver_name.apply(lambda x: re.findall(r'\w+',x)[0])

rename_eg = {'hapy':'happy'}
keep_eg = ['angry', 'expressive', 'happy', 'fearful', 'deadpan', 'sad', 'hapy']

def ren_fun(x):
    x = x.lower()
    try: 
        return rename_eg[x]
    except KeyError:
        return x
    
# Add a global version column
verdf.index.name = 'gl_ver_idx'

verdf['eg']=verdf['eg'].map(ren_fun)
verdf=verdf[verdf['eg'].isin(keep_eg)]

# Link version info to note table
link_keys = ['wavfile','ver_nbr']
needed_keys = ['gl_ver_idx','eg','instrument']
notedf=notedf.merge(verdf.reset_index()[link_keys+needed_keys],on=link_keys)

################
# Correct tables for reed position
notedf['reed_dc_corr_um'] = (notedf['reed_dc_avg']-notedf['reed_dc_avg'].groupby(notedf['subject_id']).transform('mean'))*1e6
notedf['attack_reed_corr'] = notedf['attack_reed_avg']-notedf['reed_dc_avg'].groupby(notedf['subject_id']).transform('mean')
notedf['reed_dc_var_um'] = (notedf['reed_dc_var'])*1e6

with open('/Users/goios/Devel/sensor_clarinet_processing/runsheets/melodies.yaml') as f:
     meljs = yaml.safe_load(f)
meldefs=pd.DataFrame(meljs).drop('notes').T

# Find previous deep from the best source available
notedf['prev_dip']=notedf['prev_min_before_crn_sec']
notedf.loc[notedf.prev_dip.isna(),'prev_dip'] = notedf.loc[notedf.prev_dip.isna(),'prev_local_min_sec']
notedf.loc[notedf.prev_dip.isna(),'prev_dip'] = notedf.loc[notedf.prev_dip.isna(),'prev_min_sec']
notedf['next_dip']=notedf['following_local_min_sec']
notedf.loc[notedf.next_dip.isna(),'next_dip'] = notedf.loc[notedf.prev_dip.isna(),'following_min_sec']

# dB ranges in attack
notedf['attack_val_range'] = notedf['attack_corner_val'] - notedf['attack_start_val']



##################
# Calculate averaged tempo and expected durations based on these

# Time columns that need to be offset
time_cols = ['start', 'end', 'sound_start', 'sound_end'] + notedf.columns[notedf.columns.str.endswith('_t')].tolist()
add_cols = [ 'prev_end', 'next_start', 'strict_duration', 'strong_beat', 'dur_sec']

for smpl_ver_nbr, enotedf in tqdm(notedf.groupby('gl_ver_idx')):
    #smpl_ver_nbr = notedf.gl_ver_idx.sample().iloc[0]

    #enotedf = notedf[notedf.gl_ver_idx==smpl_ver_nbr]
    smpl_notedf = enotedf.copy()
    smpl_notedf['global_index'] = enotedf.index
    smpl_notedf=smpl_notedf.loc[:smpl_notedf.note_id.idxmax()]
    smpl_notedf.set_index('note_id',inplace=True)
    
    u,c=np.unique(smpl_notedf.index, return_counts=True)
    dups = u[c>1]
    if len(dups):
        print(f'{smpl_ver_nbr}: duplicate indices {dups}. Reindexing')
        for col in ['wavfile','tune','ver_name','ver_nbr']:
            print(verdf.loc[smpl_ver_nbr,col])
        smpl_notedf=smpl_notedf.reset_index()
    
    tune = smpl_notedf.iloc[0].tune
    pmdf = meldf[meldf.tune==tune].copy()
    pmdf.iloc[-1,pmdf.columns.get_loc('strict_duration')]=False
    pd.concat([pmdf,pmdf.beats.cumsum().rename('cum_beats')],axis=1)


    base_time = smpl_notedf.start.min()
    smpl_notedf[time_cols] -= base_time
    prev_ends = smpl_notedf.set_index(smpl_notedf.index+1)[['end']].rename(columns={'end': 'prev_end'})
    next_starts = smpl_notedf.set_index(smpl_notedf.index-1)[['start']].rename(columns={'start': 'next_start'})

    prev_ends.loc[0] = 0.0
    next_starts.loc[next_starts.iloc[-1].name+1,'next_start'] = smpl_notedf.end.iloc[-1]
    #reed_offset = wavdf.loc[wavdf.ext_wavfile==verdf.loc[smpl_ver_nbr,'wavfile'],'reed_high_max'].iloc[0]
    smpl_notedf = pd.concat((smpl_notedf[['global_index','start','end','sound_start','sound_end','external_ampl_avg','external_cent_avg','f0_avg','mouth_dc_avg','reed_dc_avg']],prev_ends,next_starts),axis=1)
    #smpl_notedf['dur_sec'] = smpl_notedf.end-smpl_notedf.prev_end
    smpl_notedf['dur_sec'] = smpl_notedf.next_start-smpl_notedf.start

    pdf = pd.concat((smpl_notedf,pmdf.set_index('note_id')),axis=1)
    pdf['cum_beats'] = pdf.beats.cumsum()

    beat_durs = pdf.end/pdf.cum_beats
    #pdf['dur_beats'] = pdf.dur_sec/beat_dur
    note_st = 12*np.log2(pdf.f0_avg/440)+2
    pdf['semitones'] = note_st
    wind_beats=int(meldefs.loc[tune,'bar_duration'])*2
    rad_beats = wind_beats/2
    pdf = pdf[~pdf.global_index.isna()]

    for nid, note in pdf.iterrows():
        if not np.isnan(nid):
            nbeat = note.cum_beats-note.beats
            idx = (pdf.cum_beats>nbeat-rad_beats) & (pdf.cum_beats-pdf.beats<nbeat+rad_beats) & pdf.strict_duration
            ltempo = np.sum(pdf.loc[idx,'beats'])/np.sum(pdf.loc[idx,'dur_sec'])*60
            itempo = note.beats/note.dur_sec * 60
            exp_dur = note.beats/ltempo*60
            gid = note.global_index
            notedf.loc[gid,'ltempo'] = ltempo
            notedf.loc[gid,'itempo'] = itempo
            notedf.loc[gid,'exp_duration'] = exp_dur
            for col in add_cols:
                notedf.loc[gid,col] = note[col]
            #print(f"nbeat {nbeat:6.2f}. dur {note.dur_sec:6.2f}, exp {exp_dur:6.2f}. Local Tempo: {ltempo:8.2f}. Inst. Tempo: {itempo:8.2f} n: {np.count_nonzero(idx)}")

# FIXME: some values are not set
notedf.strict_duration.fillna(True,inplace=True)

# Duration of sound 
notedf['sound_duration'] = notedf['sound_end']-notedf['sound_start']
# Pitch difference from ref in cent
notedf['int_cent'] = 1200*np.log2(notedf['f0_avg']/notedf['exp_freq'])+200

# IOI
notedf['inter_note_time'] = -notedf.start.diff(-1)
not_strict = ~(notedf.strict_duration)
notedf.loc[not_strict,'inter_note_time'] = notedf.loc[not_strict,'end'] - notedf.loc[not_strict,'start']
# Legato fraction
notedf['staccato_frac'] = notedf['sound_duration']/notedf['inter_note_time']

# Correct detune (given that different players may use different A4 refs)
notedf['detune_corr'] = notedf['int_cent'] - notedf.groupby(['subject_id','instrument']).int_cent.transform('median')
notedf['st_var'] = np.log2(1+notedf['f0_var']/notedf['f0_avg'])
notedf['st_trend'] = 12*np.log2(1+notedf['f0_trend']/notedf['f0_avg'])
# Inter-onset interval siparity
notedf['ioi_disp'] = notedf['inter_note_time']-notedf['exp_duration']
# position of envelope max
notedf['max_rel_pos']=(notedf['note_max_sec']-notedf['prev_dip'])/(notedf['next_dip']-notedf['prev_dip'])

# Vocal tract to mouthpiece ratios
for ii in [1,2,3,4,5]:
    notedf[f'vt_h{ii}'] = 20*np.log10(notedf[f'mouth_h0{ii}_abs_avg']/notedf[f'mouthpiece_h0{ii}_abs_avg'])
    notedf[f'vt_h{ii}_var'] = 20*np.log10(notedf[f'mouth_h0{ii}_abs_var']/notedf[f'mouthpiece_h0{ii}_abs_avg'] + 
                                          notedf[f'mouthpiece_h0{ii}_abs_var']*notedf[f'mouth_h0{ii}_abs_avg']/notedf[f'mouthpiece_h0{ii}_abs_avg'])

# Minimum dB variation (avoid -inf)
min_var = -120

notedf['ext_db_avg'] = 20*np.log10(notedf['ext_ampl_avg']/2e-5)
notedf['ext_db_var'] = 20*np.log10(1+notedf['ext_ampl_var']/notedf['ext_ampl_avg'])
notedf.loc[notedf['ext_db_var']<min_var,'ext_db_var']=min_var
notedf['mouth_db_avg'] = 20*np.log10(notedf['mouth_dc_avg']/2e-5)
notedf['mouth_db_var'] = 20*np.log10(1+notedf['mouth_dc_var']/notedf['mouth_dc_avg'])
notedf['mouth_dc_var_frac'] = notedf['mouth_dc_var']/notedf['mouth_dc_avg']
notedf.to_pickle('played_notes_derived.pickle')

musical_par_cols = ['ext_db_avg','ext_db_var','ltempo','detune_corr','st_var','st_trend',
                    'staccato_frac','ioi_disp','farfield_cent_avg','attack_med_slope','max_rel_pos']
player_par_cols = ['mouth_dc_avg','mouth_dc_var_frac','reed_dc_corr_um', 'reed_dc_var_um'] + [f'vt_h{ii}' for ii in range(1,2)]
random_eff_cols = ['subject_id','instrument','tune','pitch','note_id']


################################################
################################################
####                                        ####
####     Per-version tables                 ####
####                                        ####
################################################
################################################

def most_freq(grp):
    try:
        return grp.value_counts().index[0]
    except IndexError:
        return None

ver_means=notedf[musical_par_cols+player_par_cols+['gl_ver_idx']].groupby('gl_ver_idx').mean()
ver_vars=notedf[musical_par_cols+player_par_cols+['gl_ver_idx']].groupby('gl_ver_idx').std()
ver_eg=notedf.groupby('gl_ver_idx').eg.apply(most_freq)

vdf=verdf[['wavfile', 'tune', 'instrument', 'ver_name', 'subject_id', 
       'vers_start', 'vers_end']].join(ver_means).join(ver_vars,rsuffix='_var')
vdf['eg'] = ver_eg

vdf.to_csv('clarinet_excerpts_dataset.csv')

