import numpy as np
import scipy as sp
from operator import mul


def flatten_vid_array(vid_array):
  """
    Flatten the vid_array into a single observation (row) x features (columns) representation.

    Args:
      vid_array (ndarray): An ndarray containing the video data to reshape. Should
      have the shape (height, width, channels, frames).

    Returns:
      out (ndarray): A flattened array for use in scikit-learn algorithms. Will
      be the original array, flattened to (1, frames * height * width * channels).
  """
  return reshape_vid_array(vid_array).reshape((1, -1))


def unflatten_vid_array(input_array, frame_shape=[1280, 720, 2]):
  frame_size = reduce(mul, frame_shape, 1)
  out = input_array.copy()
  out = out.reshape([-1, frame_size]).T
  return out.reshape([1280, 720, 2, -1])


def get_image_from_flattened(input_array, img_ind, frame_shape=[1280, 720, 2]):
  """
    Index into a flattened array to access a single image.

     Args:
      input_array (ndarray): An ndarray containing the flattened video data.

    Returns:
      out (ndarray): A image in the shape of frame_shape.
  """
  pass


def reshape_vid_array(vid_array):
  """
    Reshape the vid_array into a observation (rows) x features  (columns) representation.

    Args:
      vid_array (ndarray): An ndarray containing the video data to reshape. Should
      have the shape (height, width, channels, frames).

    Returns:
      out (ndarray): A reshaped array for use in scikit-learn algorithms. Will
      be the original array, reshaped to (frames, height * width * channels).
  """
  return vid_array.reshape([-1, vid_array.shape[-1]]).T


def get_image_from_input_representation(input_array, img_ind, frame_shape=[1280, 720, 2]):
  out = input_array[img_ind, :].copy()
  return out.reshape(frame_shape)


def to_sparse(input_array):
  """
    Convenience function to convert the input array into a sparse representation.
  """
  return sp.sparse.csr_matrix(input_array)


def stack_list(sparse_list):
  """
    Stack the sparse matrices in sparse_list into a single sparse matrix.
  """
  sparse_out = sparse_list[0].copy()
  for i in range(1, len(sparse_list)):
    mat = sparse_list[i]
    sparse_out = sp.sparse.vstack((sparse_out, mat), format='csr')
  return sparse_out


def to_sparse_with_threshold(input_array, thresh=0):
  pass
