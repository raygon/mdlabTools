from colorama import Fore, Back, Style
import getpass
import os
import socket
from sys import platform as _platform


def freeze_config(config, verbose=True):
  if len(config.keys()) > 1:
    raise ValueError('can only freeze single config dicts, found %s' % len(config))
  frozen_config = config
  for k in frozen_config:
    # print(frozen_config[k])
    for kk, vv in frozen_config[k].items():
      if hasattr(vv, '__call__'):
        frozen_config[k][kk] = vv()
  if verbose:
      print('froze config: %s%s%s' % (Fore.BLUE, next (iter (frozen_config.keys())), Style.RESET_ALL))
  return frozen_config


def get_config(config_data, freeze=True, strict=True, verbose=True):
  """
    Provides constants and evironment variables based on the outcome of associated
    tests.

    Args:
      config_data (dict): A dictionary containing a set of possible configurations.
        The the name of the configuration are keys in this dictionary, while
        the values are dictionaries holding configuration attributes and settings.
        Each configuration dictionary must contain the key 'testfx', a function
        which returns a boolean that determines if the configuration setting will
        be used. Note, each configuration test must be mutually exclusive so that
        there is, at most, one possible valid configuration. If multiple valid
        configurations will throw a ValueError (see "strict" argument).
      freeze (bool): If True, will iterate over all dictionary items and replace
        the value of any functions with their returned output.
      strict (bool): If True, will throw an error if more than one valid configuration
        setting is found. If False, will return all valid configurations.
      verbose (bool): If True, display verbose output.

    Returns:
      (dict): A dictionary containing the name of the valid configuration settings as a
      key, and the dictionary of configuration settings as the value.
  """
  valid_config = {}
  for k, v in config_data.items():
    if 'testfx' not in v.keys():
      raise KeyError('Every config settings dict must contain key "testfx".')
    else:
      if v['testfx']():
        valid_config[k] = v

  if len(valid_config.keys()) > 1 and strict:
    raise ValueError('Multiple valid configurations found. Use "strict" argument to ignore.')
  elif verbose:
    print('using config settings: %s%s%s' % (Fore.BLUE, next (iter (valid_config.keys())), Style.RESET_ALL))

  if freeze:
    valid_config = freeze_config(valid_config, verbose=verbose)

  return valid_config


def get_om_config(local_config, om_config, slurm_config=None):
  """
    Provides constants based on the openmind vs localhost environment.
  """
  if slurm_config is None:
    slurm_config = om_config

  if os.environ.get('HOSTNAME') == 'openmind7':
    if os.environ.get('SLURM_JOB_NAME') is not None or os.environ.get('SLURM_JOB_ACCOUNT') is not None:
      return slurm_config
    else:
      return om_config
  else:
    return local_config


def is_openmind():
  """ Check to see if process is being run on openmind. CAUTION: processes running
  through a SLURM scheduler are considered different and require a different check.

    Args:
      None

    Returns:
      (bool): True or False, indicates if process is being run on openmind.
  """
  # return socket.gethostname() == 'openmind7.mit.edu'
  return socket.gethostname() == 'openmind7'


def is_slurm_openmind():
  """ Check to see if process is being run as a SLURM job on openmind.

    Args:
      None

    Returns:
      (bool): True or False, indicates if process is being run through SLURM on
      openmind.
  """
  return os.getenv('SLURM_CLUSTER_NAME') is not None and os.getenv('SLURM_CLUSTER_NAME') == 'openmind7'


def is_mindhive():
  """ Check to see if process is being run on mindhive, specifically,
  mizell.mit.edu or rushen.mit.edu.

    Args:
      None

    Returns:
      (bool): True or False, indicates if process is being run on mindhive.
  """
  return _platform.lower() == 'linux' and socket.gethostname() in ['mizell.mit.edu', 'rushen.mit.edu']


def is_localhost():
  """ Check to see if process is being run on the localhost, specifically,
  a Mac platform with username "raygon"

    Args:
      None

    Returns:
      (bool): True or False, indicates if process is being run on localhost.
  """
  return _platform.lower() == 'darwin' and getpass.getuser() == 'raygon'
