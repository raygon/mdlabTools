""" Assorted general tools for working with audio.
"""
import numpy as np
from scipy.io import wavfile
import warnings
try:
  import pyaudio
except:
  import IPython.display


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
  if semitone_only and int(midi_number) != midi_number:
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
  if semitone_only and int(midi_number) != midi_number:
    warnings.warn('casting midi_number to int, avoid with semitone_only=False')
  midi_number = int(midi_number) if semitone_only else midi_number
  return midi_number


def parse_rescale_arg(rescale):
  """ Parse the rescaling argument to a standard form. Throws an error if rescale
    value is unrecognized.
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


def get_channels(signal):
  n_channels = 1
  if signal.ndim > 1:
    n_channels = signal.shape[1]
  return n_channels


def wav_to_array(fn, rescale='standardize'):
  """ Reads wav file data into a numpy array.

    Args:
      fn (str): path to .wav file
      normalize (str): Determines type of rescaling to perform. 'standardize' will
        divide by the max value allowed by the numerical precision of the input.
        'normalize' will rescale to the interval [-1, 1]. None or '' will not
        perform rescaling.

    Returns:
      snd, samp_freq (int, np.array): Sampling frequency of the input sound, followed
        by the sound as a numpy-array .
  """
  _rescale = parse_rescale_arg(rescale)
  samp_freq, snd = wavfile.read(fn)
  if _rescale == 'standardize':
    snd = snd / float(np.iinfo(snd.dtype).max)  # rescale so max value allowed by precision has value 1
  elif _rescale == 'normalize':
    snd = snd / float(snd.max())  # rescale to [-1, 1]
  # do nothing if rescale is None or ''
  return snd, samp_freq


def play_array(signal, pyaudio_params={}):
  _pyaudio_params = {'format': pyaudio.paFloat32,
                     'channels': 1,
                     'rate': 44100,
                     'frames_per_buffer': 1024,
                     'output': True,
                     'output_device_index': 1}

  for k, v in pyaudio_params.items():
    _pyaudio_params[k] = v

  print _pyaudio_params
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


def plot_waveform(signal, samp_freq, n_channels=None):
  import matplotlib.pyplot as plt
  if n_channels is None:
    n_channels = get_channels(signal)

  print n_channels
  if signal.size % n_channels:  # if not 0
    raise ValueError('Odd amount of data in sound array')

  time_axis = np.arange(0, signal.size / n_channels, 1)
  time_axis = time_axis / float(samp_freq)
  time_axis = time_axis * 1000  # scale to milliseconds
  print time_axis[-1]
  plt.plot(time_axis, signal)
  return time_axis, signal


def plot_spectra():
  pass


def freq_to_semitones(f1, f2):
  return 12 * np.log2(f2 / f1)

