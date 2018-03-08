"""Useful functions for working in jupyter-notebooks
"""
import sys
import IPython


def printlink(link_title, interact=True, **kwargs):
  """Create an html link with the specified text.

  Args:
    link_title (str): Identifying text used to title the link. A related
      Markdown header will be rendered, as well.
    interact (bool, default=True): Toggle display of outputs and
      visualizations. If ``False``, you are responsible for displaying
      any returned elements.
    **kwargs: Any valid keyword arguments that can be passed to
      `IPython.display.Markdown`.

  """
  link_id = link_title.lower()
  html_link = '<a id="%s"></a>' % link_id
  printmd(html_link, **kwargs)
  printmd(link_title, **kwargs)


def printmd(md_str, interact=True, **kwargs):
  """Display the input as Markdown-formatted text in a Jupyter notebook.

  Args:
    md_str (str): Markdown text to render.
    interact (bool, default=True): Toggle display of outputs and
      visualizations. If ``False``, you are responsible for displaying
      any returned elements.
    **kwargs: Any valid keyword arguments that can be passed to
      `IPython.display.Markdown`.

  Returns:
    Ipython.display.display: The IPython.display element used to render
      this Markdown text.

  """
  elem = IPython.display.Markdown(md_str, **kwargs);
  if interact:
    IPython.display.display(elem);
  return elem;


def in_notebook():
  """Check if this function is being run in an IPython/Jupyter notebook.

  Returns:
    bool: ``True`` if the module is running in IPython kernel; ``False``
    if in IPython shell or other Python shell.

  """
  return 'ipykernel' in sys.modules


def ipython_info():
  """Return a string description of the current environment.

  Returns:
    str: Description of the current environment.
  """
  ip = False
  if 'ipykernel' in sys.modules:
    ip = 'notebook'
  elif 'Ipython' in sys.modules:
    ip = 'terminal'
  return ip


if __name__ == '__main__':
  print('In Ipython notebook: %s' % in_notebook())
