"""
Simple matrix intensity plot, similar to MATLAB imagesc()
David Andrzejewski (david.andrzej@gmail.com)
"""
import numpy as NP
import matplotlib.pyplot as P
import matplotlib.ticker as MT
import matplotlib.cm as CM


def scaled_image(W, pixwidth=1, ax=None, grayscale=True, equal_aspect=True):
  """
  Do intensity plot, similar to MATLAB imagesc()
  W = intensity matrix to visualize
  pixwidth = size of each W element
  ax = matplotlib Axes to draw on
  grayscale = use grayscale color map
  Rely on caller to .show()
  """

  # N = rows, M = column
  (N, M) = W.shape
  # Need to create a new Axes?
  if(ax is None):
    ax = P.figure().gca()
  # extents = Left Right Bottom Top
  if equal_aspect:
    max_dim = max(pixwidth * M, pixwidth * N)
    exts = (0, max_dim, 0, max_dim)
  else:
    exts = (0, pixwidth * M, 0, pixwidth * N)
  if(grayscale):
    ax.imshow(W, interpolation='nearest', cmap=CM.gray, extent=exts)
  else:
    ax.imshow(W, interpolation='nearest', extent=exts)

  ax.xaxis.set_major_locator(MT.NullLocator())
  ax.yaxis.set_major_locator(MT.NullLocator())
  return ax

if __name__ == '__main__':
  # Define a synthetic test dataset
  testweights = NP.array([[0.25, 0.50, 0.25, 0.00],
                          [0.00, 0.50, 0.00, 0.00],
                          [0.00, 0.10, 0.10, 0.00],
                          [0.00, 0.00, 0.25, 0.75]])
  # Display it
  ax = scaled_image(testweights)
  P.show()

  # import torchfile
  # d = torchfile.load('/Users/raygon/Desktop/mdLab/projects/deepFerret/old/data/train_50ms_classes104_nharm_10/data_for_python.t7')

  # for i in range(100):
  #   img = d.trainData.data[i, 0, :, :]
  #   P.subplot(2, 1, 1)
  #   # scaled_image(img.reshape([img.shape[1], img.shape[0]], order='C').T)
  #   P.imshow(img.reshape([img.shape[1], img.shape[0]], order='C').T)

  #   arr = d.trainData.data
  #   arr = arr.reshape((166, 1, 301, 79), order='C')
  #   arr = arr.transpose((0, 1, 3, 2))
  #   print arr.shape
  #   P.subplot(2, 1, 2)
  #   # scaled_image(arr[i, 0, :, :])
  #   P.imshow(arr[i, 0, :, :])

  #   P.draw()
  #   P.waitforbuttonpress(2)

  #   print arr.shape
