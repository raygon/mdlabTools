import os
import re
from datetime import date
import fire
import pdb

def get_age(year, month=1, day=1):
  """Get the age from the provided day to the reference day (defaults to age
  today).

  Args:
    year (int): Birthday year.
    month (int, optional): Birthday month, defaults to January 1.
    day (int, optional): Birthday day, defaults to January 1.

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
  """
  if filter_fx is None:  # don't filter
    filter_fx = lambda: True

  metadata_template_str = 'Age: \nSex: \nMusical Experience: '

  fntp = [f for f in os.listdir(data_dir) if filter_fx()]

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
        # Path(os.path.join(data_dir, metadata_fn)).touch(mode=0o644)
        with open(os.path.join(data_dir, metadata_fn), 'w') as wfn:
          wfn.write(metadata_template_str)


if __name__ == '__main__':
  # main()
  fire.Fire()
  # make_metadata_files('/Users/raygon/Desktop/forJosh/fusion/Fusion_Data','fusion_v1\..*')
