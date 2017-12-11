"""Functions for working with hdf5 files.
"""
import os
import h5py
import numpy as np


# import ipdb

def h5_apply_lut(hdf_file, lut_inds, ref_batch_size, batch_dim=0, out_path=None, wfn_prefix=None, infer_groups=True, load_to_memory=True):
  if out_path is None:
    head, tail = os.path.split(hdf_file.filename)
    out_path = os.path.join(head, '%s__%s' % (wfn_prefix, tail))
  print('writing h5_apply_lut output to --> ', out_path)
  groups_to_process = h5_info(hdf_file)

  with h5py.File(out_path, 'w') as h5wf:
    for gn in groups_to_process:
      # load the unshuffled data into memory
      if load_to_memory:
        temp_data = hdf_file[gn][()]
      else:
        raise NotImplementedError()
      if infer_groups:
        # apply the lut to the data
        print('lut --> ', gn, ' ', temp_data.shape)
        if temp_data.ndim > 0 and temp_data.shape[batch_dim] == ref_batch_size:
          temp_data = temp_data[lut_inds]
      else:
        raise NotImplementedError()

      h5wf.create_dataset(gn, data=temp_data)
    return out_path


def h5_split_data(hdf_file, reference_groupname, split_lut, randomize=True, batch_dim=0, out_path=None, infer_groups=True, load_to_memory=True):
  split_sum = np.sum([x for x in split_lut.values()])
  if not np.isclose(split_sum, 1):
    raise ValueError('values in split_lut should add up to 1, found: %s' % split_sum)

  head, tail = os.path.split(hdf_file.filename)
  if out_path is None:
    wfn_base = 'autoSplit_'

  ref_batch_size = hdf_file[reference_groupname].shape[batch_dim]
  out_fn_list = []
  ref_mask = np.random.permutation(ref_batch_size) if randomize else np.arange(ref_batch_size)
  end = 0
  for n_split, split_name in enumerate(sorted(split_lut.keys())):
    split_value = split_lut[split_name]
    wfn_prefix = os.path.join(head, '%s%s%02d' % (wfn_base, split_name, split_value * 100))
    start = end
    end = ref_batch_size if n_split + 1 == len(split_lut) else int(split_value * ref_batch_size) + start
    mask = ref_mask[start:end]
    print('[%s, %s] -->  %s' % (start, end, mask.shape))
    out_fn = h5_apply_lut(hdf_file, mask, ref_batch_size, batch_dim=batch_dim, out_path=out_path, wfn_prefix=wfn_prefix, infer_groups=infer_groups, load_to_memory=load_to_memory)
    out_fn_list.append(out_fn)
  return out_fn_list


def h5_naive_balance_data(hdf_file, reference_groupname, num_per_class, batch_dim=0, out_path=None, infer_groups=True, load_to_memory=True):
  if out_path is None:
    head, tail = os.path.split(hdf_file.filename)
    wfn_prefix = os.path.join(head, 'autoBalanced%s' % num_per_class)

  ref_batch_size = hdf_file[reference_groupname].shape[batch_dim]

  if load_to_memory:
    classes = hdf_file[reference_groupname].value.flatten()
    mask = np.hstack([np.where(classes == c)[0][:num_per_class] for c in np.unique(classes)])
    mask.sort()
  else:
    raise NotImplementedError()

  out_fn = h5_apply_lut(hdf_file, mask, ref_batch_size, batch_dim=batch_dim, out_path=out_path, wfn_prefix=wfn_prefix, infer_groups=infer_groups, load_to_memory=load_to_memory)
  return out_fn


def h5_filter_data(hdf_file, reference_groupname, mask, batch_dim=0, out_path=None, filter_description=None, infer_groups=True, load_to_memory=True):
  filter_description = '' if filter_description is None else '_' + filter_description

  if out_path is None:
    head, tail = os.path.split(hdf_file.filename)
    wfn_prefix = os.path.join(head, 'autoFiltered%s' % filter_description)

  ref_batch_size = hdf_file[reference_groupname].shape[batch_dim]
  # if load_to_memory:
  #   ref_data = hdf_file[reference_groupname][()]
  #   mask = np.vectorize(filter_fx)(ref_data)
  # else:
  #   raise NotImplementedError()

  out_fn = h5_apply_lut(hdf_file, mask, ref_batch_size, batch_dim=batch_dim, out_path=out_path, wfn_prefix=wfn_prefix, infer_groups=infer_groups, load_to_memory=load_to_memory)
  return out_fn


def h5_shuffle_data(hdf_file, reference_groupname, batch_dim=0, out_path=None, infer_shuffle_groups=True, load_to_memory=True):
  if out_path is None:
    head, tail = os.path.split(hdf_file.filename)
    out_path = os.path.join(head, 'autoShuffled__' + tail)
  print('writing output to --> ', out_path)
  groups_to_process = h5_info(hdf_file)

  # Generate the shuffled indices
  ref_batch_size = hdf_file[reference_groupname].shape[batch_dim]
  shuffled_inds = np.random.permutation(ref_batch_size)
  with h5py.File(out_path, 'w') as h5wf:
    for gn in groups_to_process:
      print('shuffling --> ', gn)
      # load the unshuffled data into memory
      if load_to_memory:
        temp_data = hdf_file[gn][()]
      else:
        raise NotImplementedError()
      if infer_shuffle_groups:
        # shuffle the data
        print(temp_data.shape)
        if temp_data.ndim > 0 and temp_data.shape[batch_dim] == ref_batch_size:
          temp_data = temp_data[shuffled_inds]
      else:
        raise NotImplementedError()

      h5wf.create_dataset(gn, data=temp_data)


def h5_info(hdf_file, start_path='/'):
  """Print out summary information (name, shape, and dtype) for all
  datasets contained in the start_path group.

  Args:
    hdf_file (h5py.File): An open h5py.File object.
    start_path (str, optional): Start recursing at this groupname, i.e.,
      treat this as the root directory.

  Returns:
    list: A list of groupname paths to the datasets.
  """
  return h5_walk(hdf_file, start_path, 1)


def h5_walk(hdf_file, start_path='/', verbose=0):
  """Recursively visit all groups and subgroups in an hdf5 file object
  until datasets are encountered. This mimics (but does exactly mirror)
  functionality of os.walk.

  Args:
    hdf_file (h5py.File): An open h5py.File object.
    start_path (str, optional): Start recursing at this groupname, i.e.,
      treat this as the root directory.
    verbose (int, bool, optional): If truthy, this will print the
      groupname path, dataset shape, and dtype for any encountered
      datasets.

  Returns:
    list: A list of groupname paths to the datasets.

  Raises:
    ValueError: Throws an error if the hdf5 file contains duplicate keys.
  """
  out = []  # I can't think of a better way to make a flat list
  def _filter_fx(name, hd):
    if isinstance(hd, h5py.Dataset):
      if verbose:
        print('%s: %s %s' % (hd.name, hd.shape, hd.dtype))
      out.append(hd.name)
  hdf_file[start_path].visititems(_filter_fx)
  if len(out) != len(set(out)):
    raise ValueError('h5_walk output list has duplicate keys')
  return out


def h5_filtered_walk(hdf_obj, start_path='/', filter_fx=None, cull_hidden=True):
  """Perform a walk on the provided HDF file object, starting from the
  given directory, filtering the results.

    Args:
      start_path (str): Path to top-level directory from which to start
        the walk.
      filter_fx (function, optional): A function that returns False for
        files to discard and True for files to keep.

    Returns:
      fn_to_process (list): A list of strings containing the paths that
        passed the filter_fx test.
  """
  if filter_fx is None:
    _filter_fx = lambda x: True
  else:
    _filter_fx = filter_fx
  fn_to_process = h5_walk(hdf_obj, start_path=start_path)
  return [f for f in fn_to_process if _filter_fx(f)]


def h5_iFile(name, mode=None, driver=None, libver=None, userblock_size, **kwargs):
    """Open or create a new h5py file, checking to see if the

      Note that in addition to the File-specific methods and properties
      listed below, File objects inherit the full interface of Group.

      Args:
        name – Name of file (str or unicode), or an instance of h5f.FileID to bind to an existing file identifier.
        mode – Mode in which to open file; one of (“w”, “r”, “r+”, “a”, “w-”). See Opening & creating files.
        driver – File driver to use; see File drivers.
        libver – Compatibility bounds; see Version Bounding.
        userblock_size – Size (in bytes) of the user block. If nonzero, must be a power of 2 and at least 512. See User block.
        kwds – Driver-specific keywords; see File drivers.

      Returns:

    """
    raise NotImplementedError()


if __name__ == '__main__':

  rfn = '/Users/raygon/Desktop/mdLab/sounds/timit_all_flat/trimmed/postSTRAIGHT/voiced_inds_50ms/timitNoPichShift.h5'
  hd = h5py.File(rfn, 'r')
  # h5_shuffle_data(hd, 'labels')

  # mask = (hd['f0'].value.flatten() >= 100) & (hd['labels'].value.flatten() <= 30)
  # print(mask.sum())
  # h5_filter_data(hd, 'labels', mask, filter_description='GTE100Hz_LTEc30')

  # h5_naive_balance_data(hd, 'labels', 3)

  print(hd['labels'].shape)
  h5_split_data(hd, 'labels', {'train': 0.8, 'validation': 0.2})
