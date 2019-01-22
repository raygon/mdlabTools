"""Useful functions for working in jupyter-notebooks
"""
import sys
import IPython
import matplotlib.pyplot as plt

import os
import h5py
from mdlab.audioTools import *
from mdlab.h5Tools import *
import pandas as pd


def plot_sanity_check_diagnostic_training(rfn):
  ###---------- Sanity check diagnostic training dataset ----------###
  hd = h5py.File(rfn, 'r')
  h5_info(hd)

  df = pd.DataFrame({
                      'f0': hd['f0'].value,
                      'labels': hd['labels'].value,

                      'f0_offset_percentage': hd['/diagnostic/f0_offset_percentage'].value,
                      'low_harm': hd['/diagnostic/low_harm'].value,
                      'mistune_harmonic_number': hd['/diagnostic/mistune_harmonic_by_percentage'].value[:,0],
                      'mistune_harmonic_percentage': hd['/diagnostic/mistune_harmonic_by_percentage'].value[:, 1],
                      'nharm': hd['/diagnostic/nharm'].value,
                      'noise_mode': hd['/diagnostic/noise_mode'].value,
                      'phase_mode': hd['/diagnostic/phase_mode'].value,
                      'snr': hd['/diagnostic/snr'].value,
                      'upp_harm': hd['/diagnostic/upp_harm'].value,
                    })

  df.describe()

  df.hist(figsize=(15,15))
  plt.suptitle(os.path.basename(rfn))
  plt.show()

  plot_class_histogram(df['labels'], interact=False)
  plt.title('labels')
  plt.show()
  plot_class_histogram(df['low_harm'], interact=False)
  plt.title('low_harm')
  plt.show()
  plot_class_histogram(df['upp_harm'], interact=False)
  plt.title('upp_harm')
  plt.show()
  plot_class_histogram(df['nharm'], interact=False)
  plt.title('nharm')
  plt.show()


def plot_f0_and_labels_histogram(f0_arr, labels_arr, interact=True, figsize=(15, 5), **kwargs):
  if figsize is not None:
    plt.figure(figsize=figsize)

  plt.subplot(121)
  plt.hist(f0_arr, bins=100)
  plt.xlabel('F0 frequency (Hz)')
  plt.ylabel('counts')
  plt.title('F0 values histogram')

  plt.subplot(122)
  l_bins = plot_class_histogram(labels_arr, interact=False, figsize=None, **kwargs)

  if interact:
    plt.show()

  return l_bins


def plot_class_histogram(arr, interact=True, figsize=(15, 5), **kwargs):
  if figsize is not None:
    plt.figure(figsize=figsize)
  l_bins = range(int(arr.min()),  int(arr.max()) + 1 + 1)
  plt.title('bin range: %s' % l_bins)
  plt.hist(arr, bins=l_bins, **kwargs)
  plt.xlabel('integer class')
  plt.ylabel('count')

  if interact:
    plt.show()

  return l_bins


def plot_hist_and_scatter(df, key_x, key_y, label_x=None, label_y=None, title=None, figsize=(10, 10), interact=True):
  import seaborn as sns

  label_x = key_x if label_x is None else label_x
  label_y = key_y if label_y is None else label_y

  x = df[key_x]
  y = df[key_y]
  sns.jointplot(x, y, kind='hex', size=10, stat_func=None, bins=100)
  plt.title(title)
  plt.show()

  plt.figure(figsize=figsize);
  plt.plot(x, y, marker='.', linestyle='')
  plt.xlabel(label_x)
  plt.ylabel(label_y)
  plt.title(title)

  if interact:
    plt.show()


def printlink(link_title, interact=True, **kwargs):
  """Create an html link with the specified text.

  Args:
    link_title (str): Identifying text used to title the link. A related
      Markdown header will be rendered, as well.
    interact (bool, default=True): Toggle display of outputs and
      visualizations. If ``False``, you are responsible for displaying
      any returned elements.
    **kwargs: Any valid keyword arguments that can be passed to
      `IPython.display.Markdown`.

  """
  link_id = link_title.lower()
  html_link = '<a id="%s"></a>' % link_id
  printmd(html_link, **kwargs)
  printmd(link_title, **kwargs)


def printmd(md_str, interact=True, **kwargs):
  """Display the input as Markdown-formatted text in a Jupyter notebook.

  Args:
    md_str (str): Markdown text to render.
    interact (bool, default=True): Toggle display of outputs and
      visualizations. If ``False``, you are responsible for displaying
      any returned elements.
    **kwargs: Any valid keyword arguments that can be passed to
      `IPython.display.Markdown`.

  Returns:
    Ipython.display.display: The IPython.display element used to render
      this Markdown text.

  """
  elem = IPython.display.Markdown(md_str, **kwargs);
  if interact:
    IPython.display.display(elem);
  return elem;


def in_notebook():
  """Check if this function is being run in an IPython/Jupyter notebook.

  Returns:
    bool: ``True`` if the module is running in IPython kernel; ``False``
    if in IPython shell or other Python shell.

  """
  return 'ipykernel' in sys.modules


def ipython_info():
  """Return a string description of the current environment.

  Returns:
    str: Description of the current environment.
  """
  ip = False
  if 'ipykernel' in sys.modules:
    ip = 'notebook'
  elif 'Ipython' in sys.modules:
    ip = 'terminal'
  return ip


if __name__ == '__main__':
  print('In Ipython notebook: %s' % in_notebook())
