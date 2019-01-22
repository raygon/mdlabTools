import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm as cm


def tile_image_batch(img_array, h_tiles, w_tiles=None, pad=2, batch_rep='tf'):
  if w_tiles is None:
    w_tiles = h_tiles

  # print('img shape:', img_array.shape)
  pad = 2
  # disp = out_layer2[:, :, :, :w_tiles * h_tiles]
  # print(disp.shape)
  # plt.imshow(disp[0,:,:,-1])
  # plt.show()
  if batch_rep.lower() == 'tf':
    ofr, oh, ow, oc = img_array.shape
  elif batch_rep.lower() == 'np' or batch_rep.lower() == 'numpy':
    oh, ow, oc, ofr = img_array.shape
  elif batch_rep.lower() == 'torch':
    oc, oh, ow, ofr = img_array.shape
  else:
    raise ValueError('unrecognized batch_rep argument: %s' % batch_rep)

  # ofr = w_tiles * h_tiles
  disp = np.zeros(((oh+pad)*h_tiles-pad, (ow+pad)*w_tiles-pad), dtype=img_array.dtype)
  print('disp', disp.shape)
  im_ctr = 0
  for iw in range(w_tiles):
    for ih in range(h_tiles):
      if im_ctr >= ofr:
        return disp
      # print(im_ctr, ofr)
      # print(disp[:, :, im_ctr].shape)
      # print(im_ctr, iw*ow, (iw+1)*ow)
      # print(im_ctr, iw*(ow+pad), (iw+1)*ow)
      # disp[ih*(oh+pad):(ih+1)*(oh+pad), iw*(ow+pad):(iw+1)*(ow+pad)] = img_array[0, :, :, im_ctr]
      if batch_rep.lower() == 'tf':
        disp[ih*(oh+pad):(ih+1)*oh+ih*pad, iw*(ow+pad):(iw+1)*ow+iw*pad] = img_array[im_ctr, :, :, 0]
      if batch_rep.lower() == 'np' or batch_rep.lower() == 'numpy':
        disp[ih*(oh+pad):(ih+1)*oh+ih*pad, iw*(ow+pad):(iw+1)*ow+iw*pad] = img_array[:, :, 0, im_ctr]
      if batch_rep.lower() == 'torch':
        disp[ih*(oh+pad):(ih+1)*oh+ih*pad, iw*(ow+pad):(iw+1)*ow+iw*pad] = img_array[0, :, :, im_ctr]
      # disp[ih*oh:(ih+1)*oh, iw*ow+(iw+1)*pad:(iw+1)*ow+(iw+1)*pad] = img_array[0, :, :, im_ctr]
      # disp[ih*oh:(ih+1)*oh, iw*ow:(iw+1)*ow] = img_array[0, :, :, im_ctr]
      im_ctr += 1
  return disp


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
    ax = plt.figure().gca()
  # extents = Left Right Bottom Top
  if equal_aspect:
    max_dim = max(pixwidth * M, pixwidth * N)
    exts = (0, max_dim, 0, max_dim)
  else:
    exts = (0, pixwidth * M, 0, pixwidth * N)
  if(grayscale):
    ax.imshow(W, interpolation='nearest', cmap=cm.gray, extent=exts)
  else:
    ax.imshow(W, interpolation='nearest', extent=exts)

  ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
  ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
  return ax

if __name__ == '__main__':
  CIFAR_FN = '/Users/raygon/Desktop/mdLab/projects/public/mdlab/data/cifar-10-batches-py/data_batch_1'
  def unpickle(file):
      import pickle
      fo = open(file, 'rb')
      d = pickle.load(fo, encoding='latin1')
      print(d.keys())
      fo.close()
      return d

  cifar_data = unpickle(CIFAR_FN)['data']
  img_array = cifar_data.reshape((-1, 32,32,3), order='f').transpose((2, 1, 3, 0))
  # plt.imshow(b[..., 10])
  # plt.show()
  print(cifar_data.shape)

  img_array = img_array[:, :, :1, :]
  print(img_array.shape, ...)
  # plt.imshow(img_array[:,:,0,11], cmap='gray')
  # plt.show()
  img_array = img_array.transpose((2, 0, 1, 3))
  print(img_array.shape)

  tile_img = tile_image_batch(img_array, 4, 5)
  print(tile_img.shape)
  plt.imshow(tile_img, cmap='gray')
  plt.show()

  # # Define a synthetic test dataset
  # testweights = np.array([[0.25, 0.50, 0.25, 0.00],
  #                         [0.00, 0.50, 0.00, 0.00],
  #                         [0.00, 0.10, 0.10, 0.00],
  #                         [0.00, 0.00, 0.25, 0.75]])
  # # Display it
  # ax = scaled_image(testweights)
  # plt.show()

  # import torchfile
  # d = torchfile.load('/Users/raygon/Desktop/mdLab/projects/deepFerret/old/data/train_50ms_classes104_nharm_10/data_for_python.t7')

  # for i in range(100):
  #   img = d.trainData.data[i, 0, :, :]
  #   plt.subplot(2, 1, 1)
  #   # scaled_image(img.reshape([img.shape[1], img.shape[0]], order='C').T)
  #   plt.imshow(img.reshape([img.shape[1], img.shape[0]], order='C').T)

  #   arr = d.trainData.data
  #   arr = arr.reshape((166, 1, 301, 79), order='C')
  #   arr = arr.transpose((0, 1, 3, 2))
  #   print arr.shape
  #   plt.subplot(2, 1, 2)
  #   # scaled_image(arr[i, 0, :, :])
  #   plt.imshow(arr[i, 0, :, :])

  #   plt.draw()
  #   plt.waitforbuttonpress(2)

  #   print arr.shape
