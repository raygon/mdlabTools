"""Assorted tools and helpful functions for general work in Python.
This module is intended to contain general purpose routines; any
domain-specific tools (e.g., working with audio) should be grouped into
a separate module.
"""
import os
import sys
import numpy as np
import pickle
from itertools import zip_longest
import datetime
from time import strftime, gmtime
from operator import itemgetter as i
from functools import cmp_to_key
from warnings import warn
from termcolor import colored, cprint
from functools import reduce
from pandas import notnull as pd_notnull
from hashlib import sha1
from base64 import b32encode
import json

# import ipdb
import pdb as ipdb
import pprint
ppr = pprint.PrettyPrinter()


def reduce_to_unique(array, axis=0, remove_nans=True):
  """A naive way to reduce the array values to the set of unique
  values, similar to np.unique but attempts to preserve the shape of
  the array and handle nans.

  Args:
    array (array-like): Description
    remove_nans (bool):
  """
  if axis != 0:
    raise NotImplementedError()

  try:
    # look for a completely valid entry
    valid_str = 'non-nan'
    reduced_list = np.unique([x for x in array if np.all(pd_notnull(x))], axis=axis)
  except Exception as e:
    # look for a partially valid entry (some nans, but not all)
    valid_str = 'some-nan'
    reduced_list = [x for x in array if np.any(pd_notnull(x))]
    if reduced_list:
      ref_val = reduced_list[0]
      reduced_list = [x for x in reduced_list if not np.allclose(x, ref_val, equal_nan=True)]
    else:
      valid_str = 'all-nan'
      reduced_list = [array[0],]

  # print(valid_str, reduced_list)

  return reduced_list


def compute_max_segments(signal_length, duration, padding=0):
  """Compute the maximum number of continuous segments
  that can be made from the inputs with specified params.
  """
  total_seg_len = 2 * padding + duration
  return signal_length - total_seg_len + 1


def compute_max_chunks(signal_length, duration, padding=0):
  """Maximum number of non-overlapping chunks that can be
  snipped from the signal of a given length.
  """
  total_seg_len = 2 * padding + duration
  return signal_length // total_seg_len


def extract_center_window(x, window_len, strict=True):
  if strict and window_len > len(x):
    raise ValueError('window_len is greater than x; ignore with strict=False')
  # start = int(len(x) / 2 - window_len // 2)  # prefer right edge
  start = int(len(x) / 2 - window_len / 2)  # prefer left edge
  stop = start + window_len
  x = x[start:stop]
  # print('[%s, %s] --> %s' % (start, stop, len(x)))
  return x


def extract_random_window(a, width, start=None, stop=None, seed=None, verbose=1, return_inds=False):
  if len(a) == width:
    out = a
    start_ind, stop_ind = 0, width - 1
  else:
    start = 0 if start is None else start
    stop = len(a) - width if stop is None else stop
    np.random.seed(seed)

    try:
      start_ind = np.random.randint(start, stop)
    except ValueError as e:
      ipdb.set_trace()

    stop_ind = start_ind + width
    out = a[start_ind:stop_ind]

    if verbose > 0:
      print('[%s, %s] %s --> [%s, %s]' % (start, stop, out.shape, start_ind, stop_ind))

  if return_inds:
    out = (out, [start_ind, stop_ind])
  return out


def extract_windows_uneven(a, inner, pad_left=0, pad_right=None, auto_cast=True, ret_inds=True):
  """Excise windows from the array.

  Args:
    a (array-like): Array to slice up.
    width (int): Width of outer window without padding.
    stride (int, None, optional): Width of inner window steps.
    padding (int, optional): Amount of padding to apply to each side of the
      inner window, this is probably to allow skipping samples.
    auto_cast (bool, optional): If True, will coerce all arguments to the
      appropriate types.

  Returns:
    np.array: extracted windows
  """
  pad_right = pad_left if pad_right is None else pad_right

  if auto_cast:
    inner = int(inner)
    pad_left = int(pad_left)
    pad_right = int(pad_right)

  if not isinstance(inner, int):
    raise ValueError('inner must be integer')
  if not isinstance(pad_left, int):
    raise ValueError('pad_left must be integer')
  if not isinstance(pad_right, int):
    raise ValueError('pad_right must be integer')

  n = a.shape[0]
  # return np.vstack(a[i:1+n+i-width:stride] for i in range(width)).T
  start_ind = 0
  stop_ind = pad_left + inner + pad_right
  ctr = 0
  window_list = []
  inds_list = []
  while stop_ind <= n:
    temp_window = a[start_ind:stop_ind]
    window_list.append(temp_window)
    inds_list.append((start_ind, stop_ind))

    start_ind += inner
    stop_ind += inner
    ctr += 1
  return window_list, inds_list


def extract_windows(a, width, stride=None, padding=0, auto_cast=True):
  """Excise windows from the array.

  Args:
    a (array-like): Array to slice up.
    width (int): Width of outer window without padding.
    stride (int, None, optional): Width of inner window steps.
    padding (int, optional): Amount of padding to apply to each side of the
      inner window, this is probably to allow skipping samples.
    auto_cast (bool, optional): If True, will coerce all arguments to the
      appropriate types.

  Returns:
    np.array: extracted windows
  """
  width = 2 * padding + width
  stride = width if stride is None else stride

  if auto_cast:
    width = int(width)
    stride = int(stride)

  if not isinstance(width, int):
    raise ValueError('width must be integer')
  if not isinstance(stride, int):
    raise ValueError('stride must be integer')

  n = a.shape[0]
  return np.vstack(a[i:1+n+i-width:stride] for i in range(width)).T


def extract_chunks_vectorized(a, width, auto_cast=True):
  width = int(width) if auto_cast else width

  if not isinstance(width, int):
    raise ValueError('width must be integer')

  max_samples = int((len(a) // width) * width)
  out = a[:max_samples].reshape((-1, width))
  return out


def conjunction(*conditions):
  return reduce(np.logical_and, conditions)


def split_list_by_percentages(list_to_split, split_dict, randomize=False, strict=True):
  split_keys = sorted(split_dict.keys())
  split_percentages = [split_dict[k] for k in split_keys]
  list_len = len(list_to_split)

  # verify the percentages add to 1
  if sum(split_percentages) != 1:
    raise ValueError('percentages do not add to 1: %s' % sum(split_percentages))

  list_sizes = [int(round(list_len * p)) for p in split_percentages]

  # handle the discrepancy with the last key
  if sum(list_sizes) != list_len and not strict:
    diff = list_len - sum(list_sizes)
    list_sizes[-1] += diff

  if sum(list_sizes) != list_len:
    raise ValueError('sublist sizes do not sum to original length: %s/%s' % (sum(list_sizes), list_len))

  if randomize:
    _list_to_split = np.random.permutation(list_to_split)
  else:
    _list_to_split = list_to_split

  out_dict = {}
  start_ind = 0
  for (i, k) in enumerate(split_keys):
    end_ind = start_ind + list_sizes[i]
    out_dict[k] = _list_to_split[start_ind:end_ind]
    start_ind = end_ind
  return out_dict


def ndirname(path, n=1):
  n -= 1
  out_path = path
  for i in range(n):
    out_path = os.path.dirname(out_path)
  return out_path


def print_pkl(rfn):
  return pickle.load(open(rfn, 'rb' ))


def _cmp(a, b):
  """cmp doesn't exist in Python 3
  """
  return (a > b) - (a < b)


def sort_multikey(items, columns):
  # columns.reverse()  # make first key most important
  comparers = [
    ((i(col[1:].strip()), -1) if col.startswith('-') else (i(col.strip()), 1))
    for col in columns
  ]
  def comparer(left, right):
    comparer_iter = (
      _cmp(fn(left), fn(right)) * mult
      for fn, mult in comparers
    )
    return next((result for result in comparer_iter if result), 0)
  return sorted(items, key=cmp_to_key(comparer))


def num_to_kstr(num):
  """Convert a number to a string in Nk represenation (eg, '48k') if the
  number is at least 1000k
  """
  return str(num) if num < 1000 else '%sk' % (num // 1000)


def indicate_debug(s, debug_mode=False, prefix='debug_', suffix=''):
  """if debug_mode is True, add the prefix and suffix to the s.
  """
  if debug_mode:
    debug_str = prefix + s + suffix
  else:
    debug_str = s
  return debug_str





def make_iterable(x):
  """HACK WARNING: This might not work
  Ensure that the input is iterable, and if not, put it in a list so
  that it is iterable.

  Args:
    x (object): The object to make iterable. Will do nothing if the
      object is already iterable.

  Returns:
    iterable: An iterable version of the input
  """
  # try:
  #   if isinstance(x, str):
  #     raise TypeError()
  #   for i in x:
  #     break # failes wtih str
  #   # if len(list(x)) ==
  #   x_list = list(x)
  # except TypeError:
  #   x_list = [x, ]
  # return x_list
  x_list = [x, ] if np.isscalar(x) else x
  return x_list


def grouper(iterable, n_groups=None, chunk_size=None):
  """Group contents of iterable into groups of (at most) size n. If
  the iterable doesn't divide evenly into groups of size n, the last
  group will have fewer items.

  Args:
    iterable (iterable): An iterable containing the objects to be
      grouped.
    n (int): Maximum group size; last group can be smaller.

  Returns:
    list: A list of lists containing the groups of size n.
  """
  if n_groups is not None and chunk_size is not None:
    raise ValueError('can only provide one of `n_groups` or `chunk_size`')
  elif n_groups is not None:
    n = int(np.ceil(len(iterable) / n_groups))
  elif chunk_size is not None:
    n = chunk_size
  else:
    raise ValueError('grouper arguments are malformatted')

  args = [iter(iterable)] * n
  # return ([e for e in t if e != None] for t in zip_longest(*args))
  return ([e for e in t if e is not None] for t in zip_longest(*args))


def join_iter(iterable, delimiter='_', strict=False):
  if strict and iterable is None:
    return None
  else:
    out = delimiter.join(map(str, iterable))
    return out if out else None


def format_time(t, format='%H:%M:%S'):
  """Convert time (in seconds) to a reasonable format.

  Args:
    t (int): Time, in seconds.
    format (str, optional): A python time format string to use to format the
      results. Defaults to '%H:%M:%S'

  Returns:
    str: String description of the time.
  """
  return strftime(format, gmtime(t))


def get_timestamp(filename_safe=False, **kwargs):
  """Get a current timestamp with month, day, year, hour, and minute
  information.

  Args:
    filename_safe (bool, optional): If False (default), will return a
      human-readable timestamp; if True, will return a timestamp
      suitable for using in filenames.
    **kwargs: Additional keyword arguments. `str_format` can be used to
      set the format of the output.

  Returns:
    str: A string containing the timestamp information.
  """
  str_format = '%Y-%m-%d-%H%M' if filename_safe else '%b-%d-%Y-%H:%M'
  str_format = kwargs['str_format'] if 'str_format' in kwargs else str_format
  return datetime.datetime.now().strftime(str_format)


##########################################
###---------- Print Functions----------###
##########################################
# def print_flush(*args, **kwargs):
#   print(*args, **kwargs, flush=True)


def print_if_verbose(to_print, verbose, verbose_threshold=1):
  """Print the message if verbosity level is above threshold.

  Args:
    to_print (object): Object to print, if type is `callable`, will
      call the callable (just call it)
    verbose (int, bool): The verbosity of this message.
    verbose_threshold (int, optional): The threshold that must be
      met or exceeded to print.
  """
  if verbose is True:
    flex_print(to_print)
  if verbose >= verbose_threshold:
    flex_print(to_print)


def printiv(to_print, verbose, verbose_threshold=1):
  """Short name for `print_if_verbose` function, which will print the
  message if verbosity level is above threshold.

  Args:
    to_print (object): Object to print, if type is `callable`, will
      call the callable (just call it)
    verbose (int, bool): The verbosity of this message.
    verbose_threshold (int, optional): The threshold that must be
      met or exceeded to print.
  """
  print_if_verbose(to_print, verbose, verbose_threshold)


def print_replace(to_print):
  """Replace the current line in the terminal with the new message.

  Args:
    to_print (str): Message to print.
  """
  sys.stdout.write('%s\r' % to_print)
  sys.stdout.flush()


def cpprint(to_print, *args, flush=True):
  """PrettyPrinter with support for a callable.

  Args:
    to_print (object): Object to print with pprint, if type is
    `callable`, will call the callable (just call it)
  """
  print(colored(ppr.pformat(to_print), *args))
  if flush:
    sys.stdout.flush()


def flex_print(to_print, flush=True):
  """PrettyPrinter with support for a callable.

  Args:
    to_print (object): Object to print with pprint, if type is
    `callable`, will call the callable (just call it)
  """
  pp = pprint.PrettyPrinter()
  if to_print is callable:
    to_print()
  else:
    pp.pprint(to_print)
  if flush:
    sys.stdout.flush()


################################################################
###---------- File and Path Manipulation Functions ----------###
################################################################
def touch_dir(path, mode=0o755):
  """Make the directory if it doesn't exist

  Args:
    path (str): Path to directory to create if necessary.
    mode (int, optional): Octal mode to specific directory permissions.
  """
  if not os.path.isdir(path):
    try:
      os.makedirs(path, mode=mode)
    except FileExistsError as e:
      print('dir exists: ', e)


def filtered_walk(in_path, filter_fx=None, cull_hidden=True, sort=False):
  """Perform a walk on the provided directory, filtering the results.

  Args:
      in_path (str): Path to top-level directory from which to start the
        walk.
      filter_fx (function, optional): A function that returns False for
        files to discard and True for files to keep.
      cull_hidden (bool, default=True): Determines if hidden files
        (those that start with '.') should be included in the output.
      sort (bool, default=False): Determines if the output list should
        be sorted alphabetically; NOTE: this can cause out of order
        directory structure, not recommended for nested directories.

  Returns:
      fn_to_process (list): A list of strings containing the paths that
      passed the filter_fx test.
  """
  if filter_fx is None:
    _filter_fx = lambda x: True
  else:
    _filter_fx = filter_fx
  if cull_hidden:
    _hf = lambda x: not x.startswith('.')
  else:
    _hf = lambda x: True

  fn_to_process = []
  for root, dirs, files in os.walk(in_path):
    trimmed_files = [os.path.join(root, f) for f in files if _filter_fx(os.path.join(root, f)) and _hf(f)]
    if len(trimmed_files) != 0:
      fn_to_process.extend(trimmed_files)
  return fn_to_process


def strip_extension(fn):
  """Remove the filename extension by removing text to the right of the
  last '.'

  Args:
    fn (str): Filename with extension.

  Returns:
    str: Filename without extension.
  """
  return os.path.splitext(fn)[0]


def is_hidden_file(fn):
  """Check if the provided file is a hidden file. This is just to make
  code more explicit.

  Args:
    fn (str): Filename to check hidden status.

  Returns:
    bool: Indicates if file is hidden or not.
  """
  return fn.startswith('.')


def hash_json_dict(d, exclude_keys={}, **kwargs):
  _d = {k: v for k, v in d.items() if k not in exclude_keys}
  # ppr.pprint(_d)
  _kwargs = {'sort_keys': True, 'ensure_ascii': True, 'indent': 2, 'separators': (',', ':')}
  _kwargs.update(kwargs)
  d_json = json.dumps(_d, **_kwargs)
  return hash_str(d_json)


def hash_str(x):
  bin_digest = sha1(repr(x).encode()).digest()
  b32_digest = urlsafe_b32encode(bin_digest)
  return b32_digest


def hash_if_too_long(s, max_len=255):
  out_s = hash_str(s) if len(s) > max_len else s
  return out_s


def urlsafe_b32encode(s):
  b32_str = b32encode(s)
  return b32_str.decode().replace('=', '-')
  # return b32_str.replace(b'=', b'-')


def wrapped_write(write_fx, verbose=1):
  def _fx(wfn, *args, **kwargs):

    dest_dir = os.path.dirname(wfn)
    if dest_dir:
      touch_dir(dest_dir)
    out = write_fx(wfn, *args, **kwargs)
    if verbose:
      print('wrapped_write to --> %s' % wfn)
    return wfn, out
  return _fx


###---------- MISC ----------###
def convert_precision(array, dtype, round_vals=True, error_on_clamp=False):
  """Reduce numerical precision of input by rescaling depending on the
    dtype.

  Args:
      array (array-like): Description
      dtype (np.dtype): Description
      round_vals (bool, optional): Description
      error_on_clamp (bool, optional): Will throw an error if values are
        clamped, instead of just a warning.

  Returns:
      TYPE: Description

  Raises:
      ValueError: Description
  """
  out_array = np.copy(array)
  if round_vals:
    out_array = np.round(out_array)
  info = np.iinfo(dtype)
  if np.any(out_array > info.max) or np.any(out_array < info.min):
    warn('Clamping array min: %s, max: %s, to [%s, %s]' % (out_array.min(), out_array.max(), info.min, info.max))
    if error_on_clamp:
      raise ValueError('Clamped array min: %s, max: %s, to [%s, %s]' % (out_array.min(), out_array.max(), info.min, info.max))
    out_array[out_array < info.min] = info.min
    out_array[out_array > info.max] = info.max
  out_array = out_array.astype(dtype)
  return out_array
