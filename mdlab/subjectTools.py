"""Assorted tools for running psychophysics with in-lab subjects.
"""
import os
import re
from datetime import date
import fire

import utilfx

# import pdb


def get_age(year, month=1, day=1):
  """Get the age from the provided day to the reference day (defaults
  to age today).

  Args:
    year (int): Birthday year.
    month (int, default=1): Birthday month, defaults to January 1.
    day (int, default=1): Birthday day, defaults to January 1.

  Returns:
    str: The current age, formatted as y
  """
  birthday = date(year, month, day)
  now = date.today()
  age = (now - birthday) / 365
  print('-------------\n%s years old\n-------------\n' % age.days)
  return age


def make_metadata_files(data_dir, strip_suffix, filter_fx=None, metadata_stem='meta.txt', dry=False):
  """For a given subject run, create the associated metadata file.

  Args:
    data_dir (str): Path to the directory containing the data files from
      which to create templated metadata files.
    strip_suffix (str): String to replace with `metadata_stem`.
    filter_fx (None, optional): Function to filter the input data files
      to process. Returns ``True`` if the file will kept.
    metadata_stem (str, default='meta.txt'): The suffix used to write
       the metadata file to. Should probably include a filename
      extension.
    dry (bool, default=False): If ``True``, no output will be written
      or created.
  """
  filter_fx = lambda x: True if filter_fx is None else filter_fx

  metadata_template_str = 'Age: \nSex: \nMusical Experience: '

  fntp = [f for f in os.listdir(data_dir) if filter_fx(f)]

  # keep track of what metadata files exist
  metadata_fn_list = [f for f in fntp if metadata_stem in f]
  ctr = 0
  for fn in fntp:
    metadata_fn = re.sub(strip_suffix, metadata_stem, fn)
    if metadata_fn not in metadata_fn_list:
      ctr += 1
      metadata_fn_list.append(metadata_fn)
      print('%s ---> ' % ctr, metadata_fn)
      if not dry:
        # with open(os.path.join(data_dir, metadata_fn), 'w') as wfn:
          # wfn.write(metadata_template_str)
        wfn = os.path.join(data_dir, metadata_fn)
        utilfx.wrapped_write(lambda x: open(x, 'w').write(metadata_template_str))(wfn)


if __name__ == '__main__':
  fire.Fire()
