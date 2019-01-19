"""Functions for working with hdf5 files.
"""
import json
import os
import h5py
import numpy as np
import pandas as pd

# import ipdb


def _h5_format(arr, mode='auto', **kwargs):
  """

  Args:
    arr (pd.Series, pd.DataFrame, array of array-like): Description
    mode (str, optional): Description
    **kwargs: Description

  Returns:
    TYPE: Description

  Raises:
    NotImplementedError: Description
  """
  if isinstance(arr, pd.Series) or isinstance(arr, pd.DataFrame):
    arr = arr.values

  if mode == 'auto':
    if isinstance(arr[0], np.ndarray):
      mode = 'array'
    elif isinstance(arr[0], str):
      mode = 'string'
    else:
      return arr

  out = arr.tolist()
  if mode == 'array':
    out = np.array(out, **kwargs)
  elif mode == 'string':
    out = np.string_(out, **kwargs)
  else:
    raise NotImplementedError(mode)
  return out


###---------- HDF5 Inspection and Summary Functions ----------###
def h5_info(hdf_obj, start_path=None, print_values=False):
  """Print out summary information (name, shape, dtype [,value]) for all
  datasets contained in the `start_path` group.

  Args:
    hdf_file (h5py.File): An open h5py.File object; can also be an h5py
      group.
    start_path (str, default='/'): Path to top-level hdf5 group
      (directory) from which to start the walk.
    print_values (bool, default=False): Determines if hdf5 values will
      be printed for each dataset. **CAUTION**: this will load the full
      dataset into memory; this is only recommended for config data.

  Returns:
    list: A list of groupname paths to the datasets.

  """
  verbose = 2 if print_values else 1
  return h5_walk(hdf_obj, start_path, verbose=verbose)


def h5_filtered_walk(hdf_obj, start_path=None, filter_fx=None):
  """Perform a walk on the provided HDF file object, starting from the
  given directory, filtering the results.

  Args:
    hdf_obj (h5py.File): An open h5py file or group object.
    start_path (str, default='/'): Path to top-level hdf5 group
      (directory) from which to start the walk.
    filter_fx (function, optional): A function that returns ``False``
      for files to discard and ``True`` for files to keep. Defaults
      to returning all children items.
    verbose (int, bool, default=False): If truthy, this will print the
      groupname path, dataset shape, and dtype for any encountered
      datasets.

  Returns:
    fn_to_process (list): A list of strings containing the paths that
      passed the filter_fx test.

  """
  start_path = hdf_obj.name if start_path is None else start_path
  _filter_fx = lambda x: True if filter_fx is None else filter_fx
  fn_to_process = h5_walk(hdf_obj, start_path=start_path)
  return [f for f in fn_to_process if _filter_fx(f)]


def h5_walk(hdf_file, start_path=None, verbose=0):
  """Recursively visit all groups and subgroups in an hdf5 file object
  until datasets are encountered. This mimics (but does not exactly
  mirror functionality of ``os.walk``.

  Args:
    hdf_file (h5py.File): An open h5py.File object.
    start_path (str, default='/'): Start recursing at this groupname;
      i.e., treat this as the root directory.
    verbose (int, default=0): If truthy, this will print the
      groupname path, dataset shape, and dtype for any encountered
      datasets.

  Returns:
    list: A list of groupname paths to the datasets.

  Raises:
    ValueError: Throws an error if the hdf5 file contains duplicate keys.
  """
  out = []  # I can't think of a better way to make a flat list
  start_path = hdf_file.name if start_path is None else start_path

  def _filter_fx(name, hd):
    if isinstance(hd, h5py.Dataset):
      if verbose:
        val_str = '' if verbose <= 1 else hd[()]
        print('%s: %s %s %s' % (hd.name, hd.shape, hd.dtype, val_str))
      out.append(hd.name)
  hdf_file[start_path].visititems(_filter_fx)
  if len(out) != len(set(out)):
    raise ValueError('h5_walk output list has duplicate keys')
  return out


###---------- HDF5 Manipulation Functions ----------###
def h5_naive_add_data(hdf_file, dset_name, data, axis=0, single_write=False, write_none=False, verbose=False, **kwargs):
  """Summary

  Args:
    hdf_file (h5py.File): An open h5py.File object.
    dset_name (str): The path to the dataset to be written to. If this
      dataset does not exist, it will be created.
    data (h5py compatible data object, ndarray): The data to write to
      disk.
    axis (int, default=0): Defines the batch axis to use when writing
      data to an existing dataset.
    single_write (bool, default=False): Controls writing behavior so
      that the dataset is only created/written to once. If
      `single_write` is ``True`` and the dataset exists, an error will
      be raised. This is useful for writing config data.
    write_none (bool, default=False): Determines behavior for when the
      data is ``None``. If ``False``, the dataset will not be created
      or written to; If ``True``, writing will proceed as normal --
      NOTE: this can lead to unexpected behavior with hdf5 files.
    verbose (int, default=0): If truthy, this will print the
      groupname path, dataset shape, and dtype for any encountered
      datasets.
    **kwargs: Any optional keyword arguments to pass to
      `h5_naive_concatenate`.

  Raises:
    NotImplementedError: Description
  """
  if not write_none and data is None:
    if verbose:
      print('`data` is None for dset_name: %s; not writing' % dset_name)
    return

  dtype = kwargs.get('dtype')

  if axis != 0:
    raise NotImplementedError('function not implemented for axis: %s' % axis)

  # convert_dict_to_json = True
  # # elif kwargs.get('convert_dict_to_json') and isinstance(data, dict):
  # if convert_dict_to_json and isinstance(data, dict):
  #   data = json.dumps(data, sort_keys=True)

  convert_dict_to_json = True
  if isinstance(data, str):
    data = np.string_(data)
  # else:
  #   try:
  #     data = np.atleast_1d(data)
  #   except TypeError as e:
  #     print('TYYPEERROR, not making into array')
  # elif kwargs.get('convert_dict_to_json') and isinstance(data, dict):
  elif convert_dict_to_json and not isinstance(data, np.ndarray):
    data = json.dumps(data, sort_keys=True)
  # else:
    # raise ValueError('h5_naive_add_data failed: %s' % dset_name)

  if single_write is True:
    hdf_file.create_dataset(dset_name, data=data, dtype=dtype)
  else:
    if dset_name not in hdf_file:
      try:
        maxshape = (None, *data.shape[1:])
      except AttributeError as e:
        maxshape = (None,)

      hdf_file.create_dataset(dset_name, data=data, maxshape=maxshape, dtype=dtype)
    else:
      h5_naive_concatenate(hdf_file[dset_name], data, axis=0, verbose=verbose, **kwargs)


def h5_naive_concatenate(hdf_dset, data, axis=0, verbose=1, **kwargs):
  """Append data to an existing hdf5 dataset, extending along `axis`.

  Args:
    hdf_dset (h5py.Dataset): An open h5py dataset object.
    data (h5py compatible data object, ndarray): The data to add to the
      existing dataset.
    axis (int, default=0): Defines the batch axis to use when writing
      data to an existing dataset.
    verbose (int, default=1): If truthy, this will print the
      groupname path, dataset shape, and dtype for any encountered
      datasets.
    **kwargs: Description

  Raises:
    NotImplementedError: Description
  """
  if axis != 0:
    raise NotImplementedError('function not implemented for axis: %s' % axis)

  # dtype = kwargs.get('dtype')

  old_shape = hdf_dset.shape
  new_shape = list(old_shape)
  new_shape[axis] = new_shape[axis] + data.shape[axis]
  if verbose:
    print('resizing hdf5_dset %s --> %s' % (hdf_dset.shape, new_shape))
  hdf_dset.resize(new_shape)
  hdf_dset[old_shape[0]:] = data  # TODO: generalize this

  # hdf_slice = [slice(None) for x in new_shape]
  # hdf_slice[axis] = slice(-1, None)
  # print(hdf_slice)
  # hdf_dset[hdf_slice] = data # i don't know how to generalize this


def h5_apply_lut(hdf_file, lut_inds, ref_batch_size, batch_dim=0, out_path=None, wfn_prefix=None, infer_groups=True, load_to_memory=True):
  """Summary

  Args:
    hdf_file (h5py.File): An open h5py.File object.
    lut_inds (TYPE): Description
    ref_batch_size (TYPE): Description
    batch_dim (int, optional): Description
    out_path (None, optional): Description
    wfn_prefix (None, optional): Description
    infer_groups (bool, optional): Description
    load_to_memory (bool, default=True): Determines if each dataset will
      be loaded into memory for faster processing. TODO: implement with
      dask methods

  Returns:
    TYPE: Description

  Raises:
    NotImplementedError: Description
  """
  if out_path is None:
    head, tail = os.path.split(hdf_file.filename)
    out_path = os.path.join(head, '%s__%s' % (wfn_prefix, tail))
  print('writing h5_apply_lut output to --> ', out_path)
  groups_to_process = h5_info(hdf_file)

  with h5py.File(out_path, 'w') as h5wf:
    for gn in groups_to_process:
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
  return out_path  # post-Feb 20, 2018


def h5_split_data(hdf_file, reference_groupname, split_lut, randomize=True, batch_dim=0, out_path=None, infer_groups=True, load_to_memory=True):
  """Split an hdf5 dataset into several, mutually exclusive splits.

  Args:
    hdf_file (h5py.File): An open h5py.File object.
    reference_groupname (str): The path to the hdf5 dataset used to
      define the batch size
    split_lut (dict): A dictionary containing the names of the splits to
      be created along with the associated fraction of the data used to
      construct the split; i.e. a dict with
      'split_name': split_percentage pairs. NOTE: the values in
       `split_lut` should add up to 1, will throw an error if otherwise.
    randomize (bool, default=True): Specify if the data should be
      shuffled before creating the splits.
    batch_dim (int, optional): Description
    out_path (None, optional): Description
    infer_groups (bool, default=True): If ``True``, then this function
      will attempt to split every dataset that matches the batch size
      of the `reference_groupname`.
    load_to_memory (bool, default=True): Determines if each dataset will
      be loaded into memory for faster processing. TODO: implement with
      dask methods

  Returns:
    TYPE: Description

  Raises:
    ValueError: Description
  """
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
  """Create a shuffled

  Args:
      hdf_file (TYPE): Description
      reference_groupname (TYPE): Description
      batch_dim (int, optional): Description
      out_path (None, optional): Description
      infer_shuffle_groups (bool, optional): Description
      load_to_memory (bool, optional): Description

  Raises:
      NotImplementedError: Description
  """
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
      print('h5_shuffle_data --> ', gn)
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
