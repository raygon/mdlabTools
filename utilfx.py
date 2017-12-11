import pprint
import os
import sys
import numpy as np
import pickle
from itertools import zip_longest
from time import strftime, gmtime
from operator import itemgetter as i
from functools import cmp_to_key
from warnings import warn
from termcolor import colored, cprint
from functools import reduce

# import ipdb
import pprint
ppr = pprint.PrettyPrinter()


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

def strip_extension(fn):
  """Remove the filename extension by removing text to the right of the
  last '.'

  Args:
    fn (str): Filename with extension.

  Returns:
    str: Filename without extension.
  """
  return os.path.splitext(fn)[0]


def grouper(iterable, n):
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
  args = [iter(iterable)] * n
  # return ([e for e in t if e != None] for t in zip_longest(*args))
  return ([e for e in t if e is not None] for t in zip_longest(*args))


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


def cpprint(to_print, *args, flush=True):
  """PrettyPrinter with support for a callable.

  Args:
    to_print (object): Object to print with pprint, if type is
    `callable`, will call the callable (just call it)
  """
  print(colored(ppr.pformat(to_print), *args))
  if flush:
    sys.stdout.flush()

def print_replace(to_print):
  """Replace the current line in the terminal with the new message.

  Args:
    to_print (str): Message to print.
  """
  sys.stdout.write('%s\r' % to_print)
  sys.stdout.flush()


def touch_dir(path, mode=0o755):
  """Make the directory if it doesn't exist

  Args:
    path (str): Path to directory to create if necessary.
    mode (int, optional): Octal mode to specific directory permissions.
  """
  # ipdb.set_trace()
  if not os.path.isdir(path):
    os.makedirs(path, mode=mode)


def filtered_walk(in_path, filter_fx=None, cull_hidden=True):
  """Perform a walk on the provided directory, filtering the results.

    Args:
      in_path (str): Path to top-level directory from which to start the
      walk.
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


def is_hidden_file(fn):
  """Check if the provided file is a hidden file. This is just to make
  code more explicit.

  Args:
    fn (str): Filename to check hidden status.

  Returns:
    bool: Indicates if file is hidden or not.
  """
  return fn.startswith('.')


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
