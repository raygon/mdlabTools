import os
import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import goalieTrial as gt
import flowTrial as ft
from utilfx import *
from frameSlicer import *

from joblib import Parallel, delayed
import multiprocessing
import itertools
import time

from vizCommon import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from pylab import tight_layout
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA

import cvVideo
import scipy.misc
import flowViz


def plot_img_vx(img_vx, title_params, color='blue', y_flip=True, use_pca=False):
  disp_img_vx = img_vx.copy()
  # if use_pca:
  #   pca = PCA(n_components=2, whiten=True)
  #   disp_img_vx = pca.fit_transform(disp_img_vx)
  #   print disp_img_vx.shape

  # color = [1, 1, 0]
  plt.scatter(disp_img_vx[:, 0], disp_img_vx[:, 1], c=color)

  if use_pca:
    plt.title('Plot of PCA Flow Values%s' % title_params['title'])
    plt.xlabel('PC1 Component')
    plt.ylabel('PC2 Component')
  else:
    plt.title('Plot of XY Flow Values%s' % title_params['title'])
    plt.xlabel('X Component')
    plt.ylabel('Y Component')
  # Convert to screen-space coordinates
  if y_flip:
    plt.gca().invert_yaxis()
  return plt.gcf()


def plot_flow_vx(flow, use_pca=True, n_sample=None, title_params=None, write_path=None, display=False):
  if title_params is None:
    title_params = {'title': ''}
  if len(flow.shape) != 2 and len(flow.shape) != 3 and len(flow.shape) != 4:
    raise ValueError('Argument "flow" must have 2 or 3 or 4 dimensions.')
  if len(flow.shape) == 4:
    flow_vx = flow.reshape((-1, flow.shape[-2], flow.shape[-1]))
  else:
    flow_vx = flow.reshape((-1, flow.shape[-1], 1))
  print '# of flow frames: %s :: %s' % (flow_vx.shape[-1], flow_vx.shape.__str__())

  if flow_vx.shape[-1] > 1:
    title_params['title'] = ' Stacked %s' % title_params['title']

  color = iter(cm.rainbow(np.linspace(0, 1, flow_vx.shape[-1])))
  # c = next(color)
  # plt.plot(x, y, c=c)

  legend_patches = []
  for i in range(flow_vx.shape[-1]):
    c = next(color)
    legend_patches.append(mpatches.Patch(color=c, label=i))
    temp_vx = flow_vx[..., i]
    print "=="
    print temp_vx.shape

    if use_pca:
      pca = PCA(n_components=2, whiten=False)
      # temp_vx = pca.fit_transform(temp_vx)
      pca.fit(temp_vx)

      print pca.components_
      print pca.components_[:, 0]
      print pca.components_[0]
      print "$$$$"
      print pca.mean_
      print pca.explained_variance_ratio_
      print pca.get_covariance()
      print temp_vx.shape

    # sample array (for large amounts of data)
    if n_sample is not None:
      sample_inds = np.random.choice(temp_vx.shape[0], n_sample, replace=False)
      temp_vx = temp_vx[sample_inds, :]
      print temp_vx.shape

    if write_path is not None or display:
      c = temp_vx.copy() / temp_vx.max()
      c = draw_hsv(temp_vx.reshape([-1, 1, 2]), gamma=1/5.0, k=10)
      print c.min()
      print c.max()
      print c.dtype
      c = c.reshape([-1, 3])
      # c = c / float(255)
      # c[:] = [0.5, 0, 0]
      print
      print c.min()
      print c.max()
      print c.shape
      print temp_vx.shape
      fig = plot_img_vx(temp_vx, title_params, color=c, y_flip=False, use_pca=False)
      if use_pca:
        ax = fig.gca()
        # Which one to use???
        # vx_avg = temp_vx.mean(axis=0)
        vx_avg = pca.mean_
        pc_comp = pca.components_

        # Some form of scaling
        pc_covar = pca.get_covariance()
        pc_eig_vals = np.linalg.eigvalsh(pc_covar)
        pc1_k = np.sqrt(pc_eig_vals[1])
        pc2_k = np.sqrt(pc_eig_vals[0])

        pc1 = np.vstack((vx_avg, pc1_k * pc_comp[0, :] + vx_avg)).T
        pc2 = np.vstack((vx_avg, pc2_k * pc_comp[1, :] + vx_avg)).T
        ax.plot(pc1[0, :], pc1[1, :], c='r', linewidth=3.0)
        ax.plot(pc2[0, :], pc2[1, :], c='k', linewidth=3.0)
        plt.axis('equal')

  plt.legend(handles=legend_patches)
  if write_path is not None or display:
    if display:
      plt.gca().invert_yaxis()
      plt.show()
    if write_path is not None:
      wfn = os.path.join(write_path, 'xy_flow_%s.png' % title_params['title'])
      plt.savefig(wfn)



def plot_flow_vx_from_fn(flow_fn, offset=flowViz.OFFSET_DFLT, n_sample=1000, write_path=None, display=False):
  # load flow
  flow_obj = ft.FlowTrial.from_file(flow_fn)

  # slice frames
  # flow_frames0 = easy_slice(flow_obj, offset)
  fx = slicer_fx(easy_slice, {'offset': offset})
  flow_frames = fx(flow_obj)
  # print flow_frames.shape
  # print np.all(flow_frames == flow_frames0)

  # avg frame
  avg_frame = flow_frames.mean(axis=-1)
  # avg_frame = flow_frames
  # print avg_frame.shape
  print write_path

  # plot
  if write_path is not None or display:
    title_params = {'title': ' for Avg_Frame: %s' % os.path.basename(flow_fn).split('_Dense')[0]}
    plot_flow_vx(avg_frame, n_sample=n_sample, title_params=title_params, write_path=write_path, display=display)



def main():
  # WRITE_PATH = '/Users/raygon/Desktop/nkLab/goalie/out/flow/dense/block_avg'
  WRITE_PATH = '/Users/raygon/Desktop/nkLab/goalie/test_out/flow/dense/block_avg'
  # READ_PATH = '/Users/raygon/Desktop/nkLab/goalie/out/flow/dense/subj1_2_block1'
  READ_PATH = '/Volumes/Untitled 4/goalie/out/flow/dense/subj11_12_block2'
  RAW_VID_READ_PATH = '/Volumes/Untitled 4/goalie/pertrialFiles'
  # READ_PATH = '/Volumes/Untitled 4/goalie/out/flow/dense'
  # test_fn = '/Users/raygon/Desktop/nkLab/goalie/out/flow/dense/subj1_2_block1/left_trial10/subj1_2_block1/left_trial10_LK_Flow.mat'
  test_fn = '/Volumes/Untitled 4/goalie/out/flow/dense/subj11_12_block2/left_trial30/left_trial30_Dense_Flow.mat'
  test_fn2 = '/Volumes/Untitled 4/goalie/out/flow/dense/subj11_12_block2/right_trial1/right_trial1_Dense_Flow.mat'
  path_to_blocks = '/Volumes/Untitled 4/goalie/out/flow/dense/'

  plot_flow_vx_from_fn(test_fn, offset=10, n_sample=50000, display=True)
  plot_flow_vx_from_fn(test_fn2, offset=10, n_sample=50000, display=True)

  # mean = (0, 0)
  # cov = [[2, 1], [1, 3]]
  # cov = [[5, 1], [1, 2]]
  # gauss_flow = np.random.multivariate_normal(mean, cov, (720, 1280))
  # print gauss_flow.shape
  # plot_flow_vx(gauss_flow, n_sample=10000, display=True)


if __name__ == '__main__':
  main()
