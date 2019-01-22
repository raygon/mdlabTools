import os
import sys
import numpy as np
import cv2
import scipy.io as sio
import h5py
import utilfx


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from pylab import tight_layout


def load_from_mat(filename, cvt_color=True, rotate=True):
  """ Load the data from a .mat file. Will convert color from BGR to RGB and will
    rotate 90 degrees, if requested.

  """
  data = sio.loadmat(filename)
  vid_data = data['StartToHitMx']
  if rotate:
    data['StartToHitMx'] = np.rot90(data['StartToHitMx'])
  if cvt_color:
    for i in range(vid_data.shape[-1]):
      data['StartToHitMx'][..., i] = cv2.cvtColor(data['StartToHitMx'][..., i], cv2.cv.CV_RGB2BGR)
    # for i in range(vid_data.shape[-1]):
    #   data['StartToHitMx'][..., i] = cv2.cvtColor(data['StartToHitMx'][..., i], cv2.cv.CV_BGR2RGB)
  return data


def load_from_hdf(fn):
  # with h5py.File(fn, 'r') as hdf_obj:
  #   da
  pass

def write_to_mat(data_dict, wfn, read_mode='w'):
  directory = os.path.dirname(wfn)
  if read_mode is not 'w-' and read_mode is not 'r':
    utilfx.touch_dir(directory)
  sio.savemat(wfn, data_dict)


def write_to_hdf(data_dict, wfn, read_mode='w'):
  logger = utilfx.get_logger(__name__)
  if 'vid' in data_dict and 'StartToHitMx' in data_dict:
    logger.error('data_dict cannot contain both keys: "vid" and "StartToHitMx"', exc_info=True)
    raise KeyError('data_dict cannot contain both keys: "vid" and "StartToHitMx"')
  if 'vid' in data_dict:
    data_dict['StartToHitMx'] = None
  if 'lor' not in data_dict or 'frameLaunchEnd' not in data_dict or 'StartToHitMx' not in data_dict or 'file_id' not in data_dict:
    logger.error('data_dict is incomplete', exc_info=True)
    raise KeyError('data_dict is incomplete')
  with h5py.File(wfn, read_mode) as hdf_obj:
    base_group = data_dict['file_id'].split('.hdf5')[0]
    for k in data_dict:
      if k == 'file_id':
        continue
      hdf_key = os.path.join(base_group, k)
      if k == 'StartToHitMx':
        if data_dict[k] is None:
          continue
        hdf_key = os.path.join(base_group, 'vid')
      if hdf_key in hdf_obj:
        logger.info('skipping writing to hdf: %s' % hdf_key)
      else:
        hdf_obj[hdf_key] = data_dict[k]


def matplotlib_writer(write_filename, shape, fps=30, dpi=100, scl=1):
  h, w, c, f = shape
  FFMpegWriter = manimation.writers['ffmpeg']
  writer = FFMpegWriter(fps=fps, codec="libx264")

  fig = plt.figure(figsize=(w / 100.0, h / 100.0))
  ax = fig.add_subplot(111)
  ax.set_aspect('equal')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  img_disp = ax.imshow(np.zeros((h, w, c)))
  img_disp.set_clim([0, 1])
  fig.set_size_inches([scl * w / 100.0, scl * h / 100.0])
  tight_layout()

  wf = writer.saving(fig, write_filename, dpi)

  def update_frame(frame):
    img_disp.set_data(frame)
    writer.grab_frame()

  return (wf, update_frame)
