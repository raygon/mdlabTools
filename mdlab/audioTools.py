""" Assorted general tools for working with audio.
"""
import os
import subprocess
import numpy as np
from scipy.io import wavfile, loadmat
import scipy.signal
import warnings
import time
import glob
from scipy.io.wavfile import read 
from scipy.io.wavfile import write 
from os.path import basename
try:
  import pyaudio
except:
  import IPython.display

import mat2py
import mdlab.utilfx as utilfx

import pdb as ipdb


def play(s, sr, pause_s=0, interact=True, **kwargs):
  """Jupyter notebook based audio playback.

  Args:
    s (array-like): Signal waveform array to playback.
    sr (int): Sampling rate for audio playback.
    pause_s (number, default=0): Amount of time to wait after playback
      before completing; useful for adding silence between consecutive
      playback calls.
    interact (bool, default=True): Determines if audio element will be
      displayed. If ``True``, playback element will be created and
      displayed. If ``False``, the playback element will only be
      created and not displayed.
    **kwargs: Optional keyword arguments to pass to the underlying
      IPython.display.Audio call.

      * **autoplay** (*bool*): If ``True``, will initiate playback once
        the element is displayed.

  Returns:
    `IPython.display.Audio`:
      **audio_elem**-- The playback element for the provided signal at
        the provided sampling rate. Can be displayed in a notebook.
  """
  pause_s = pause_s + len(s) / sr if pause_s is True else pause_s
  audio_elem = IPython.display.Audio(s, rate=sr, **kwargs)
  if interact:
    audio_elem = IPython.display.display(audio_elem)
    time.sleep(pause_s)
  return audio_elem


#############################################
###---------- SIGNAL ATTRIBUTES ----------###
#############################################
def rms(a, strict=True):
  """Compute root mean squared of array.
  WARNING: THIS BREAKS WITH AXIS, only works on vector input.

  Args:
    a (array): Input array.

  Returns:
    array:
      **rms_a**: Root-mean-squared of array.
  """
  out = np.sqrt(np.mean(a * a))
  if strict and np.isnan(out):
    raise ValueError('rms calculation resulted in a nan: this will affect ' +
                     'later computation. Ignore with `strict`=False')
  return out


def get_channels(signal):
  n_channels = 1
  if signal.ndim > 1:
    n_channels = signal.shape[1]
  return n_channels


def pwelch_db(x, fs, **kwargs):
  fxx, pxx = scipy.signal.welch(x, fs=fs, **kwargs)
  pxx = 10 * np.log10(pxx)
  return fxx, pxx


def get_length_after_polyphase_resample(signal_length, from_sr, to_sr, invert=False):
  """Compute the approximate length of the signal after polyphase
  resampling (i.e., scipy.signal.resample_poly).

  Args:
    signal_length (TYPE): Description
    from_sr (TYPE): Description
    to_sr (TYPE): Description
    invert (bool, optional): Description

  Returns:
    TYPE: Description
  """
  from_sr, to_sr = (to_sr, from_sr) if invert else (from_sr, to_sr)
  return np.ceil(signal_length * (int(to_sr) / from_sr))


#############################################
###---------- SIGNAL ALTERATION ----------###
#############################################
def combine_signal_and_noise(signal, noise, snr, rms_mask=None):
  """Combine the signal and noise at the provided snr.

  Args:
    signal (array-like): Signal waveform data.
    noise (array-like): Noise waveform data; same shape as `signal`.
    snr (number): SNR level in dB.
    rms_mask (None, optional): Optional binary mask with the same shape
      as `signal` to use for the snr calculation. Mask values of 1 will
      be included in the calculation, values of 0 will be ignored. This
      mask will be applied to both `signal` and `noise`.

  Returns:
    `ndarray`:
    **signal_and_noise**-- Combined signal and noise waveform.
  """
  # normalize the signal
  # signal = signal / rms(signal)
  # sf = np.power(10, snr / 10)
  # signal_rms = rms(signal)
  # noise = noise * ((signal_rms / rms(noise)) / sf)
  # signal_and_noise = signal + noise
  # return signal_and_noise

  # ipdb.set_trace()
  rms_mask = np.full(len(signal), True, dtype=bool) if rms_mask is None else rms_mask
  signal = signal / rms(signal[rms_mask])
  # sf = np.power(10, snr / 10)
  sf = np.power(10, snr / 20) # Mar 15, 2018
  signal_rms = rms(signal[rms_mask])
  noise = noise * ((signal_rms / rms(noise[rms_mask])) / sf)
  signal_and_noise = signal + noise
  return signal_and_noise


def strip_silence(signal, db_threshold=-20):
  """Strip silence from the beginning and end of the signal.

  "Silent regions" are defined as any place where the signal power is
  lower than `db_thershold` below the signal's peak amplitude. The
  longest silent regions at the start and end of the signal will be
  removed

  Args:
    signal (array-like): Signal waveform data.
    db_threshold (TYPE, default=-60): Set silence threshold in terms
      of dB below the peak amplitude. Any sample below this threshold
      will be considered silence.

  Returns:
    TYPE: Description
  """
  amp_threshold = np.abs(signal).max() * 10 ** (db_threshold / 10)
  above_threshold_mask = np.abs(signal) >= amp_threshold

  above_threshold_inds = np.where(above_threshold_mask)[0]
  ind_first = min(above_threshold_inds)
  ind_last = max(above_threshold_inds)
  # print('[%s, %s]' % (ind_first, ind_last))

  # # custom function to find first inde
  # for i, x in enumerate(disp_signal_mask):
  #   print(x)
  #   ipdb.set_trace()
  #   if x is True:
  #     ipdb.set_trace()
  #     ind_first = i
  #     break
  # for i, x in enumerate(disp_signal_mask[::-1]):
  #   if x is True:
  #     ind_last = i
  #     break
  return signal[ind_first:ind_last], (ind_first, ind_last)



def pure_tone(f0_hz, dur_ms=50, sr=16000):
  dur_samples = dur_ms / 1000 * sr
  t = np.linspace(0, dur_ms / 1000, dur_samples)
  out = np.sin(2 * np.pi * f0_hz * t)
  return out


def dfGenerateSynthSignal(t, sr, f0, lowH, uppH, phaseMode, slope=None, phaseOffset=None, useHannWindow=True, stimStatMode='falloff', useStrict=True, timeOffset=0, per_harm_fx=None):
  ### Construct a synthetic tone consisting of a harmonic stack for use in
  ### deepFerret synthStim stimuli
  # get a random phase offset if one isn't provided:
  phaseOffset = np.random.uniform(0, 2 * np.pi) if phaseOffset is None else phaseOffset
  if useStrict:
    nyq = sr / 2
    nyqHarm = int(nyq // f0)
    if nyqHarm < uppH:
      raise ValueError('useStrict error in dfGenerateSynthSignal: uppH is higher than nyquist limit: %s' % nyqHarm)

  # shift time, if requested
  _t = t.copy()
  if timeOffset:
    _t = t + timeOffset

  ct = np.zeros_like(_t, dtype=float)
  for phase_ctr, h in enumerate(range(lowH, uppH + 1)): # +1 for mat2py inclusive range
    ### <generate the phase shift if any>
    if phaseMode.lower() == 'rand':
      phase = np.random.uniform(0, 2 * np.pi)
    elif phaseMode.lower() == 'sine':
      phase = 0
    elif phaseMode.lower() == 'sineoffset': # shift starting phase
      phase = phaseOffset
    elif phaseMode.lower() == 'randoffset': # randomize starting phase
      phase = np.random.uniform(0, 2 * np.pi) + phaseOffset
    elif phaseMode.lower() == 'alt':
      phase = np.pi / 2 * (phase_ctr % 2)
      # print('====> %s phase: %s' % (h, phase))
    else:
      raise NotImplementedError('unsupported `phaseMode`: %s' % phaseMode)
    ### </generate the phase shift, if any>

    ### <generate tone of this harmonic>
    temp = np.sin(2 * np.pi * h * f0 * _t + phase)  # include f0
    if 'falloff' in stimStatMode.lower():
        if 'reverse' in stimStatMode.lower():
            slope = -slope
        # apply harmonic-dependent attenuation, if any
        if slope:
          atten = slope * np.log2(h)
          temp = temp * 10**(atten / 20)
        else:
          raise ValueError('slope must be defined to impose frequency falloff')
    ct += temp
  if useHannWindow:
    ct = mat2py.hann(ct, 10, sr)
  ### <generate tone of this harmonic>
  return ct


###########################################
###---------- FREQUENCY TOOLS ----------###
###########################################
def get_semitone_difference(x, y):
  """Compute the difference between x and y (x - y) in semitones.

  Args:
    x (array-like):
    y (array-like):
  """
  return 12 * np.log2(x / y)


def add_semitone_offset(center_freq, n_semitones):
  """Add

  Args:
    center_freq (array-like):
    n_semitones (array-like):
  """
  return center_freq * 2.0**(n_semitones / 12)


def get_allowable_f0_range(center_frequency, range_offsets, offset_mode='semitone'):
  # parse range offsets
  range_offset_min, range_offset_max = range_offsets if np.iterable(range) else [range_offsets, range_offsets]

  # convert allowable offset range to frequency range
  if offset_mode.lower() == 'frequency' or offset_mode.lower() == 'hz':
    range_min_hz = center_frequency - range_offset_min
    range_max_hz = center_frequency + range_offset_max
  elif offset_mode.lower() == 'semitone':
    range_min_hz = add_semitone_offset(center_frequency, -range_offset_min)
    range_max_hz = add_semitone_offset(center_frequency, range_offset_max)
  else:
    raise ValueError('unrecognized `offset_mode`: %s' % offset_mode)

  # apply offset
  out = [range_min_hz, range_max_hz]
  return out


################################
###---------- MIDI ----------###
################################
def midi_to_freq(midi_number, semitone_only=True):
  """Convert the midi note number to frequency in Hz.

  Note: Fractional midi note number will give you 'cents' from standard
  pitch: e.g. 66.33 would be 33 cents above F#4.

  Equation from http://glassarmonica.com/science/frequency_midi.php

  Args:
    midi_number (number): The midi number representation of the note.
    semitone_only (bool, optional): If True, will restrict output to
      integer only inputs (semitone/half-tones/integer midi_numbers).

  Return:
    float: The frequency representation of the note in Hz.
  """
  # if semitone_only and int(midi_number) != midi_number:  # TODO: vectorize this
  if semitone_only and np.any(np.vectorize(np.int)(midi_number) != midi_number): # HACK
    raise ValueError('midi_number must be an integer with semitone_only=True: %s' % midi_number)
  freq = 27.5 * np.power(2.0, ((midi_number - 21.0) / 12.0))
  return freq


def freq_to_midi(freq, semitone_only=True):
  """Convert the note from frequency in Hz to the midi number.

  Note: Fractional midi note number will give you 'cents' from standard
  pitch: e.g. 66.33 would be 33 cents above F#4.

  Equation from http://glassarmonica.com/science/frequency_midi.php

  Args:
    freq (number): The frequency representation of the note in Hz.
    semitone_only (bool, optional): If True, will restrict output to
      integer only inputs (semitone/half-tones/integer midi_numbers).

  Return:
    float: The midi number representation of the note.
  """
  midi_number = (12.0 / np.log(2.0)) * np.log(freq / 27.5) + 21.0
  # if semitone_only and int(midi_number) != midi_number:  # TODO: vectorize this
  if semitone_only and np.any(np.vectorize(np.int)(midi_number) != midi_number): # HACK
    warnings.warn('casting midi_number to int, avoid with semitone_only=False')
  midi_number = int(midi_number) if semitone_only else midi_number
  return midi_number


######################################
###---------- DEPRECATED ----------###
######################################
def parse_rescale_arg(rescale):
  """DEPRECATED: try librosa.load; preference for no rescaling, or only
  rescaling before playback.
  Parse the rescaling argument to a standard form. Throws an error if rescale
  value is unrecognized.

  Args:
    rescale (str): String that determines type of rescaling to perform;
      see :func:`~mdlab.audioTools.wav_to_array` for specifics.

  Returns:
    `out_rescaled`:
    * **str** The standardized version of the ``rescale`` argument.

  Raises:
    ValueError
  """
  _rescale = rescale.lower()
  if _rescale == 'normalize':
    out_rescale = 'normalize'
  elif _rescale == 'standardize':
    out_rescale = 'standardize'
  elif _rescale is None or _rescale == '':
    out_rescale = None
  else:
    raise ValueError('Unrecognized rescale value: %s' % rescale)
  return out_rescale


def wav_to_array(fn, rescale='standardize'):
  """DEPRECATED: try librosa.load
  Reads wav file data into a numpy ndarray and rescales values
  according to ``rescale`` argument.

  Args:
    fn (str): path to .wav file
    normalize ({'standardize', 'normalize', None}; default='standardize'): Determines
      type of rescaling to perform. 'standardize' will divide by the
      max value allowed by the numerical precision of the input.
      'normalize' will rescale to the interval [-1, 1]. ``None`` will
      not perform rescaling.

  Returns:
    snd, samp_freq (int, np.array): Sampling frequency of the input sound, followed
      by the sound as a numpy-array.

  Returns:
    `tuple`:
    A tuple containing:

      * **snd** (*ndarray*)-- The sound waveform, from the file.
      * **samp_freq** (*int*)-- The sampling rate in Hz.

  Raises:
    warnings.DeprecationWarning
  """
  raise warnings.DeprecationWarning('General loading is being deprecated; you should probably ' +
                                    'use something that is more documented and feature rich, like librosa.load.')
  _rescale = parse_rescale_arg(rescale)
  samp_freq, snd = wavfile.read(fn)
  if _rescale == 'standardize':
    snd = snd / float(np.iinfo(snd.dtype).max)  # rescale so max value allowed by precision has value 1
  elif _rescale == 'normalize':
    snd = snd / float(snd.max())  # rescale to [-1, 1]
  # do nothing if rescale is None or ''
  return snd, samp_freq


def load_audio(rfn, signal_key='signal', rate_key='sr'):
  """DEPRECATED: try librosa.load and stop using MATLAB.
  Intended to be a general loader for signal waveform array data.

  Args:
    rfn (str): Path to file from which to load signal waveform data.
    signal_key (str, default='signal'): Specify the key-name to lookup
      the signal data  associated with the file; useful if the input file
      is formatted as key-value pairs.
    rate_key (str, default='sr'): Specify the key-name to lookup the
      sampling rate associated with the file; useful if the input file
      is formatted as key-value pairs.

  Returns:
    `tuple`:
    A tuple containing:

      * **s** (*ndarray*)-- The sound waveform, from the file.
      * **sr** (*int*)-- The sampling rate in Hz.
  Raises:
    NotImplementedError
    warnings.DeprecationWarning
  """
  # raise warnings.DeprecationWarning('General loading is being deprecated; you should probably ' +
  #                                   'use something that is more documented and feature rich, like librosa.load.')
  dep_msg = 'General loading is being deprecated; you should probably use something that is more documented and feature rich, like librosa.load.'
  raise warnings.warn(+DeprecationWarning)
  rfn_lower = rfn.lower()
  if rfn_lower.endswith('.wav'):
    sr, s = wavfile.read(rfn)
  elif rfn_lower.endswith('.mat'):
    out_data = loadmat(rfn)
    s = out_data[signal_key]
    sr = out_data[rate_key]
  else:
    raise NotImplementedError('load_audio is not defined for file type: %s' % rfn)
  return s, sr


def play_array(signal, pyaudio_params={}):
  """DEPRECATED: try ``play`` or Jupyter Notebook features.
  Play the provided signal via PyAudio; be careful with rescaling
  values, as this can get dangerously loud.

  Args:
    signal (array-like): Signal waveform array, e.g., as returned from
      ``wav_to_array``.
    pyaudio_params (dict, optional): Dictionary of keyword arguments
      to pass to pyaudio backend.

  Raises:
    warnings.DeprecationWarning
  """
  raise warnings.DeprecationWarning('PyAudio playback is being deprecated for IPython Notebook playback, see: ``play``')
  _pyaudio_params = {'format': pyaudio.paFloat32,
                     'channels': 1,
                     'rate': 44100,
                     'frames_per_buffer': 1024,
                     'output': True,
                     'output_device_index': 1}

  for k, v in pyaudio_params.items():
    _pyaudio_params[k] = v

  print(_pyaudio_params)
  p = pyaudio.PyAudio()
  # stream = p.open(format=pyaudio.paFloat32,
  #                 channels=1,
  #                 rate=44100,
  #                 frames_per_buffer=1024,
  #                 output=True,
  #                 output_device_index=1)
  stream = p.open(**_pyaudio_params)
  data = signal.astype(np.float32).tostring()

  # stream = p.open(format=pyaudio.paInt16, channels=1, rate=samp_freq, output=True, frames_per_buffer=CHUNKSIZE)
  # data = snd.astype(snd.dtype).tostring()
  stream.write(data)


#############################################
###---------- FORMAT CONVERSION ----------###
#############################################
def sphere_to_wav(fn_sphere, out_fn=None, dry=False):
  """Convert audio file from SPHERE format to WAV  files (e.g., for
  corpora from the Linguistic Data Consortium like Wall Street Journal).

  NOTE: This requires that sph2pipe is installed, see
  https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools.

  Args:
    fn_sphere (str): Path to SPHERE file to convert.
    out_fn (str, default=None): Path to write the results of the
      conversion. If ``None``, will attempt to change the file extension
      to .wav and write in the same location as ``fn_sphere``.
    dry (bool; default=False): If ``False``, conversion will occur and
      the output will be written to disk. If ``True``, only a summary
      of the work will be printed.

  Returns: TODO
    `MatlabEngine`:
    * **matlab_engine** A MATLAB engine for performing computation.

  """
  if out_fn is None:
    fn_wav = ''.join([os.path.splitext(fn_sphere)[0], '.wav'])
  else:
    fn_wav = out_fn
  print('%s --> %s' % (fn_sphere, fn_wav))
  if not dry:
    utilfx.touch_dir(os.path.dirname(fn_wav))
    return subprocess.call(['sph2pipe', fn_sphere, fn_wav])


def dir_convert_sphere_to_wav(path, out_dir=True, fx_filter=None, dry=False):
  """Convert all of the SPHERE formatted files in a given directory
  to WAV files (e.g., for corpora from the Linguistic Data Consortium).

  By default, only the files with "*.sph" extension will be converted.

  Args:
    path (str): Convert to SPHERE to WAV files within this directory.
    out_dir (bool, str; default=True): Determines the directory where
      converted files will be written. If ``True``, converted files
      will be written in the same directory as the unconverted files.
      This can be a str containing the path to the output directory
      where output will be written.
    fx_filter (callable, default=None): Function that will be used
      to select files for conversion from the parent directory in
      ``path`` by returning True. By default, this will only keep
      files that end in ".sph" for conversion.
    dry (bool, default=False): If ``False``, conversion will occur and
      the output will be written to disk. If ``True``, only a summary
      of the work will be printed.

  """
  fx_filter = lambda x: x.endswith('.sph') and os.path.isfile(x) if fx_filter is None else fx_filter
  fntp = [os.path.join(path, f) for f in os.listdir(path) if fx_filter(f)]

  out_dir = os.path.join(path, 'wav') if out_dir is True else out_dir

  for fn in fntp:
    out_fn = os.path.join(out_dir, os.path.basename(os.path.splitext(fn)[0] + '.wav'))
    sphere_to_wav(fn, out_fn=out_fn, dry=dry)



def make_random_phase_spectrogram(mu,sigma,shape):
    '''
    Makes spectrogram with randomized phases by:
        1. Drawing a waveform with `shape` smaples from a gaussian distribution.
        2. Taking the FFT of that wavefrom to get its spectrogram.
        3. Normalizing each sample in the spectrogram by its complex magnitude.
        4. Taking the real component of the IFFT of the normalized
        spectrogram.

    Parameters:
        Mu (int) : Mean of gaussian distribution samples are drawn from.
        Sigma (int): Standard devidation of gaussian distribution samples are
            drawn from.
        Shape (int) : length of the wavefrom to be sampled
    
    Returns:
        Spectrogram (ndarray) : Spectrogram containing randomized phases

    '''
    white_noise = np.random.normal(mu,sigma,shape)
    wnoise_fft = np.fft.fft(white_noise,shape)
    wnoise_amplitude = np.absolute(wnoise_fft)
    wnoise_phases = np.divide(wnoise_fft,wnoise_amplitude)

    return wnoise_phases


def randomize_waveform_phases(target_waveform):
    '''
    Makes waveform with random phases that mathces the average spectra
    of the target waveform.
    
    Parameters:
        Target_waveform (ndarray): 1xN array containing the waveform which will
        have it's pahse randomized

    Returns:
        target_waveform_rand_phase: Target wavefrom with randomized phases
    '''

    pdb.set_trace()
    target_spectrum = np.fft.fft(target_waveform,target_waveform.shape[-1]*2)
    wnoise_phases = \
        make_random_phase_spectrogram(0,1,target_waveform.shape[-1]*2)
    target_spectrogram_rand_phase = np.multiply(target_spectrum,wnoise_phases)
    target_waveform_rand_phase_padded = \
        np.fft.ifft(target_spectrogram_rand_phase).real
    halfway_idx = round(target_waveform_rand_phase_padded.shape[0]/2)
    target_waveform_rand_phase = \
        target_waveform_rand_phase_padded[:halfway_idx]
    return target_waveform_rand_phase


def make_speech_shaped_noise(target_folder_regex,output_dir):
    '''
    Takes `target_directory_regex` regular expression and creates speech shaped noise
    for all .wav files in that directory. The new wav files are placed in
    `output_dir`.

    Parameters:
        target_directory_regex (string) : Regular expression specifying which
        files to use as templates for shaped noise genration.

        output_dir (string) : Location to save shaped noise. If the location
        does not exist, the folder will be created.

    Returns:
        None

    Raises:
        UserWarning: Warns user if provided audio file has more than one
        channel. 


    '''


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fnames = glob.glob(target_folder_regex)
    for fname in fnames:
        sr, waveform = read(fname)
        if len(waveform.shape) > 1:
            warnings.warn("More than one audio channel in {}. Using first channel only.".format(fname))
            waveform = waveform[:,0]
        fname_base = basename(fname)
        ss_noise = randomize_waveform_phases(waveform)
        ss_noise = ss_noise.astype(np.int16)
        write_string = output_dir + '/' + fname_base.split('.')[0] + "_speechShapedNoise.wav"
        write(write_string, sr, ss_noise)


###---------- MAINS ----------###
def main_tedlium_convert_sph_to_wav():
  # path = '/mindhive/mcdermott/shared/Sounds/Speech/TED-LIUM_Release_2/TED-LIUM_release2/dev/sph'
  # path = '/mindhive/mcdermott/shared/Sounds/Speech/TED-LIUM_Release_2/TED-LIUM_release2/test/sph'
  path = '/mindhive/mcdermott/shared/Sounds/Speech/TED-LIUM_Release_2/TED-LIUM_release2/train/sph'
  dir_convert_sphere_to_wav(path, out_dir=True, dry=False)
