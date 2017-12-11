import sys


def in_notebook():
  """
  Returns ``True`` if the module is running in IPython kernel,
  ``False`` if in IPython shell or other Python shell.
  """
  return 'ipykernel' in sys.modules


def ipython_info():
  ip = False
  if 'ipykernel' in sys.modules:
    ip = 'notebook'
  elif 'Ipython' in sys.modules:
    ip = 'terminal'
  return ip

if __name__ == '__main__':
  print('In Ipython notebook: %s' % in_notebook())
