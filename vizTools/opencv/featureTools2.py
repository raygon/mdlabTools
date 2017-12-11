""" Tools and operations for dealing with input features in observation x features
representation (scikit-learn representation). Usefule on prepped data. """
import cv2
import numpy as np


def n_pyrDown(img, n=1):
  """ Recursively apply pyrDown to downsample image n-times.

    Args:
      img (np.array): Frame data to be downsampled, should have shape (h, w, [1 or 3]).
      n (optional[int]): Number of times to apply downsampling. Note, each
      downsampling operation downsamples h and w by a factor of 2 (total
      downsampling by a factor of 4).

    Returns:
      out_frame (np.array): Downsampled output frame, with size (h/2, w/2, [1 or 3]).
  """
  if n < 0:
    raise ValueError('n_pyrDown arg n cannot be negative')
  elif n == 0:
    return img
  else:
    # return n_pyrDown(cv2.pyrDown(img.astype(np.float)), n-1)
    return n_pyrDown(cv2.pyrDown(img), n-1)


def downsample_vid(vid_array, n=1):
  """ Downsample each frame in the vid_array, using n_pyrDown.

    Args:
      img (np.array): Frame data to be downsampled, should have shape (h, w, [1 or 3]).
      n (optional[int]): Number of times to apply downsampling. Note, each
      downsampling operation downsamples h and w by a factor of 2 (total
      downsampling by a factor of 4).

    Returns:
      out_frame (np.array): Downsampled output frame, with size (h/2, w/2, [1 or 3]).
  """
  print vid_array.shape
  # print vid_array[:, :, 0, 0].shape
  # vid array is either [h, w, fr] for grayscale, or [h, w, 3, fr] for rgb
  if vid_array.ndim == 3:
    return np.array([n_pyrDown(vid_array[:, :, i], n=n) for i in range(vid_array.shape[-1])]).transpose((1, 2, 0))
  if vid_array.ndim == 4 and vid_array.shape[2] == 2:
    return np.array([n_pyrDown(vid_array[:, :, :, i], n=n) for i in range(vid_array.shape[-1])]).transpose((1, 2, 3, 0))


def feature_to_frame(array, frame_shape=(1280, 720, -1)):
  """ Convert data from a feature representation to an image frame representation.

    Args:
      array (np.array): Numpy array data in feature representation, i.e.,
      (observations, features).
      frame_shape (tuple): Specify the output frame shape, as (h, w, c, fr).
      -1 can be used to specific one uknown dimension that will be filled accordingly.

    Returns:
      frame (np.array): A numpy array containing the input array in an image
      frame representation, with the shape of frame_shape, i.e., (h, w, c, fr).
  """
  if len(frame_shape) == 2:
    _fr_shape = [-1, frame_shape[0], frame_shape[1]]
  elif len(frame_shape) == 3:
    _fr_shape = [frame_shape[2], frame_shape[0], frame_shape[1]]
  elif len(frame_shape) == 4:
    _fr_shape = [frame_shape[3], frame_shape[0], frame_shape[1], frame_shape[2]]
    return array.reshape(_fr_shape).transpose((1, 2, 3, 0))
  elif len(frame_shape) == 5:
    _fr_shape = [frame_shape[4], frame_shape[0], frame_shape[1], frame_shape[2], frame_shape[3]]
    return array.reshape(_fr_shape).transpose((1, 2, 3, 4, 0))
  else:
    raise NotImplementedError()
  return array.reshape(_fr_shape).transpose((1, 2, 0))


def frame_to_feature(array):
  """ Convert data from an image frame representation to a feature representation.

    Args:
      array (np.array): Numpy array data in image frame representation, i.e.,
      (h, w, c, fr). Note, currently only supports ndim=4 (h, w, c, fr).

    Returns:
      frame (np.array): A numpy array containing the input array in scikit-learn's
      feature representation, i.e., (observations, features).
  """
  if array.ndim == 3:
    return array.reshape((-1, array.shape[-1])).T
  # grayscale or rgb
  elif array.ndim == 4:
    return array.reshape((-1, array.shape[-1])).T
  elif array.ndim == 5:
    # raise NotImplementedError()
    return array.reshape((-1, array.shape[-1])).T
    # return array.reshape((-1, array.shape[-1]))
  # grayscale
  else:
    raise NotImplementedError()
