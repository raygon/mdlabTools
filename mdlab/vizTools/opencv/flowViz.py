import os
import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import goalieTrial as gt
import projectConfig as config
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

import cvVideo
import scipy.misc

OFFSET_DFLT = -10
MX_SCALE_DFLT = 5
STEP_DFLT = 30

######################
####  @S UTILITY  ####
######################


def load_cache(cache_fn):
  """
    Load data from cache_fn .mat file. If file doesn't exist, returns None.
  """
  out = None
  if os.path.exists(cache_fn):
    out = sp.io.loadmat(cache_fn)
  return out


def gen_avg_flow_cache_name(path, offset=None):
  if offset is None:
    offset = ''
  root, fn = os.path.split(path)
  return os.path.join(root, config.AVG_FLOW_SUBDIR, config.AVG_FLOW_STEM + 'offset(%s)_%s' % (offset, fn))


def partition_list(l, n):
  """
    Try to group the l into n groups with roughly equal size.
  """
  groups = [l]
  while len(groups) < n:
    group_lens = map(len, groups)
    arg_min = group_lens.index(max(group_lens))
    temp_group = groups[arg_min]
    # split list
    del groups[arg_min]
    groups.extend(split_list(temp_group))
  return groups


def split_list(l):
  """
    Split the provided list into two sublists.
  """
  mid = len(l) / 2
  return [l[:mid], l[mid:]]


###################
####  @S MAIN  ####
###################


def avg_flow_per_trial(read_filename, write_path=None, offset=OFFSET_DFLT, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, force=False, display=False, verbose=False):
  if verbose:
    print
    print "Viz Average Flow Per Trial"
    print "=========================="
    print 'basename: %s, offset: %s' % (read_filename, offset)

  avg_frame = None
  fn_split = os.path.split(read_filename)
  cache_fn = os.path.join(fn_split[0], 'avg_flow_cache', 'avg_flow_cache_offset(%s)_%s' % (offset, fn_split[1]))
  print cache_fn
  if os.path.exists(cache_fn) and not force:
    if verbose:
      print "Reading trail average flow from cached file"
      print '@ %s' % cache_fn

    # avg_frame = sp.io.loadmat(cache_fn)
    # if avg_frame['offset'] != offset:
    #   raise ValueError("Frame offset in loaded doesn't match")
    # avg_frame = avg_frame['avg_frame']

    cache_data = load_cache(cache_fn)
    if cache_data is not None:
      # unpack cache data
      avg_frame = cache_data['avg_frame']
      if offset != cache_data['offset']:
        raise ValueError("Cached data 'offset' does not match 'offset' argument.")
      print "Using cached values from %s" % cache_fn
  if avg_frame is None:
    # Load flow
    flow_obj = ft.FlowTrial.from_file(read_filename)
    if verbose:
      print 'Motion: %s, Target: %s, Vid Shape: %s' % (flow_obj.frameLaunchEnd[0][0], flow_obj.lor, flow_obj.vid.shape.__str__())

    # Select subset of frames for feature vector
    start = flow_obj.frameLaunchEnd[0][0]
    stop = start + offset
    d_slice = make_frame_slice(start, offset)
    vx = extract_frames(flow_obj.vid.copy(), d_slice)
    print 'Keeping Frames: %s to %s' % (min(start, stop), max(start, stop))
    end = start + offset
    if end < start:
      start, end = end, start
    vx2 = flow_obj.vid.copy()[..., start:end]
    assert np.all(vx == vx2)

    # Get average values of feature vector
    avg_frame = vx.mean(axis=-1)

    # Write to cache
    touch_dir(os.path.dirname(cache_fn))
    sp.io.savemat(cache_fn, {'avg_frame': avg_frame, 'offset': offset})

  # Handle Figures
  if write_path is not None or display:
    print avg_frame.shape
    title_params = {'read_filename': read_filename, 'offset': offset}
    viz_avg_summary_trial(avg_frame, write_path=write_path, display=display, title_params=title_params)

  # Handle data writing
  if write_path is not None:
    wfn = os.path.join(write_path, 'offset(%s)_scale(%s)_step(%s)_trial_avg' % (offset, mx_scale, step))
    touch_dir(os.path.dirname(wfn))
    sp.io.savemat(wfn, {'avg_frame': avg_frame, 'offset': offset})

  return avg_frame


def avg_flow_per_block(block_path, write_path=None, offset=OFFSET_DFLT, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, display=False, force=False):
  """
    Visualize the average flow per block from average flows per trial.
  """
  first_frame_fn = '/Users/raygon/Desktop/nkLab/goalie/data/pertrialFiles/subj1_2_block1/left_trial6_StartToHit.mat'
  first_frame = gt.GoalieTrial.from_file(first_frame_fn).vid[..., 0]
  first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

  print "VIZING "
  print write_path
  print offset

  left_avg, right_avg = None, None
  left_ctr, right_ctr = 0, 0
  # Check cache
  p_str = 'offset(%s)_%s' % (offset, os.path.basename(block_path))
  cache_fn = os.path.join(block_path, 'block_avg_cache', 'block_avg_cache_%s.mat' % (p_str))
  if not force:
    print "Checking for cached file: %s" % cache_fn
    cache_data = load_cache(cache_fn)
    if cache_data is not None:
      # unpack cache data
      left_avg = cache_data['left_avg']
      right_avg = cache_data['right_avg']
      if offset != cache_data['offset']:
        raise ValueError("Cached data 'offset' does not match 'offset' argument.")
      print "Using cached values from %s" % cache_fn
  if left_avg is None or right_avg is None:
    for root, dirs, files in os.walk(block_path):
      for name in [f for f in files if f.endswith('.mat')]:
        if 'cache' in root or 'cache' in name:
          break
        if 'avg_flow' in root or 'avg_flow' in name:
          break
        fn = os.path.join(root, name)
        avg_frame = avg_flow_per_trial(fn, offset=offset, mx_scale=mx_scale, step=step, display=False)
        if right_avg is None and left_avg is None:
          right_avg = np.zeros_like(avg_frame)
          left_avg = np.zeros_like(avg_frame)

        if 'right' in fn:
          right_avg += avg_frame
          right_ctr += 1
        elif 'left' in fn:
          left_avg += avg_frame
          left_ctr += 1
        else:
          raise ValueError("Trial must be either left or right, found: %s" % fn)

        print "r_max: %s, l_max: %s" % (right_avg.max(), left_avg.max())
        print "r#: %s, l#: %s" % (right_ctr, left_ctr)

    right_avg /= float(right_ctr)
    left_avg /= float(left_ctr)

    # Write cache file
    print "writing cache file"
    touch_dir(os.path.dirname(cache_fn))
    sp.io.savemat(cache_fn, {'right_avg': right_avg,
                             'left_avg': left_avg,
                             'mx_scale': mx_scale,
                             'step': step,
                             'offset': offset,
                             'info:': 'total_avg'})

  print "MAXES: %s, %s" % (right_avg.max(), left_avg.max())

  # Handle figures
  if write_path is not None or display:
    wfn = os.path.join(write_path, 'offset(%s)_scale(%s)_step(%s)' % (offset, mx_scale, step), 'block_avgXX')
    title_params = {'block_path': block_path, 'offset': offset, 'mx_scale': mx_scale, 'step': step}
    viz_avg_summary_lr(left_avg, right_avg, write_path=wfn, display=display, title_params=title_params)

  # Data writing
  if write_path is not None:
    touch_dir(os.path.dirname(wfn))
    # wfn = os.path.join(write_path, 'offset(%s)_scale(%s)_tot_avg' % (offset, mx_scale))
    sp.io.savemat(wfn, {'right_avg': right_avg,
                        'left_avg': left_avg,
                        'mx_scale': mx_scale,
                        'step': step,
                        'offset': offset,
                        'info:': 'total_avg'})

  return {'right': right_avg, 'left': left_avg}


def avg_flow_all(block_path, write_path=None, offset=OFFSET_DFLT, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, display=False, force=False):
  """
    Visualize the average flows across all blocks from average flows per trial.
  """
  left_avg, right_avg = None, None
  left_ctr, right_ctr = 0, 0
  # Check cache
  p_str = 'offset(%s)' % (offset)
  cache_fn = os.path.join(block_path, 'tot_avg_cache', 'tot_avg_cache_%s.mat' % p_str)
  if not force:
    print "Checking for cached file: %s" % cache_fn
    cache_data = load_cache(cache_fn)
    if cache_data is not None:
      # unpack cache data
      left_avg = cache_data['left_avg']
      right_avg = cache_data['right_avg']
      if offset != cache_data['offset']:
        raise ValueError("Cached data 'offset' does not match 'offset' argument.")
      print "Using cached values from %s" % cache_fn
  if left_avg is None or right_avg is None:
    # Get list of files to compute averages
    to_compute = []
    all_fn = []
    for root, dirs, files in os.walk(block_path):
      for name in [f for f in files if f.endswith('.mat') and 'cache' not in f]:
        fn = os.path.join(root, name)
        pregen_fn = os.path.join(root, config.AVG_FLOW_SUBDIR, config.AVG_FLOW_STEM + 'offset(%s)_%s' % (offset, name))
        if not os.path.exists(pregen_fn):
          to_compute.append(fn)
        all_fn.append(fn)
    print 'Need to compute averages for %s trials' % len(to_compute)

    # exit()
    # Compute averages for required files
    if len(to_compute) > 0:
      pregen_fn = os.path.join(config.AVG_FLOW_SUBDIR, config.AVG_FLOW_STEM + 'offset(%s)' % (offset))
      params = {'write_stem': pregen_fn, 'offset': offset, 'mx_scale': mx_scale, 'step': step}
      par_avg_flow_per_trial(to_compute, params)

    # Collect averages
    print 'Collecting averages'
    # all_fn = [gen_avg_flow_cache_name(f, offset=offset) for f in all_fn]
    # d = par_sum_and_count_lr(all_fn, num_cores=4)
    # left_avg = d['left_avg']
    # right_avg = d['right_avg']

    for root, dirs, files in os.walk(block_path):
      for name in [f for f in files if f.endswith('.mat') and 'avg' not in f]:
        fn = os.path.join(root, name)
        avg_frame = avg_flow_per_trial(fn, offset=offset, mx_scale=mx_scale, step=step)

        if right_avg is None and left_avg is None:
          right_avg = np.zeros_like(avg_frame)
          left_avg = np.zeros_like(avg_frame)

        if 'right' in fn:
          right_avg += avg_frame
          right_ctr += 1
        elif 'left' in fn:
          left_avg += avg_frame
          left_ctr += 1
        else:
          raise ValueError("Trial must be either left or right, found: %s" % fn)
        print "Right #:%s, Left #:%s" % (right_ctr, left_ctr)
    right_avg /= float(right_ctr)
    left_avg /= float(left_ctr)

    # Write cache file
    touch_dir(os.path.dirname(cache_fn))
    sp.io.savemat(cache_fn, {'right_avg': right_avg,
                             'left_avg': left_avg,
                             'mx_scale': mx_scale,
                             'step': step,
                             'offset': offset,
                             'info:': 'total_avg'})

  # Handle figures
  if write_path is not None or display:
    wfn = os.path.join(write_path, 'offset(%s)_scale(%s)' % (offset, mx_scale), 'tot_avgXX')
    print wfn
    print '%s, %s' % (left_avg.min(), left_avg.max())
    print '%s, %s' % (right_avg.min(), right_avg.max())
    title_params = {'block_path': block_path, 'offset': offset, 'mx_scale': mx_scale, 'step': step}
    viz_avg_summary_lr(left_avg, right_avg, write_path=wfn, display=display, title_params=title_params)

  # Data writing
  if write_path is not None:
    touch_dir(os.path.dirname(write_path))
    # wfn = os.path.join(write_path, 'offset(%s)_scale(%s)_tot_avg' % (offset, mx_scale))
    sp.io.savemat(wfn, {'right_avg': right_avg,
                        'left_avg': left_avg,
                        'mx_scale': mx_scale,
                        'step': step,
                        'offset': offset,
                        'info:': 'total_avg'})

  return {'right': right_avg, 'left': left_avg}


def par_make_figs(read_path, write_path):
  """
    Compute block averages for all blocks in 'read_path' using various offset
    parameters, storing results to 'write_path'
  """
  write_path = '/Users/raygon/Desktop/nkLab/goalie/out/flow/dense/block_avg'
  # read_path = '/Users/raygon/Desktop/nkLab/goalie/out/flow/dense/subj1_2_block1'
  read_path = '/Volumes/Untitled 4/goalie/out/flow/dense/'

  offsets = [-10, -2, 2, 10]

  trials = [os.path.join(read_path, d) for d in os.listdir(read_path) if 'block' in d]
  print trials
  trials = trials[:1]
  print len(trials)
  print trials
  # exit()

  num_cores = multiprocessing.cpu_count()
  num_cores -= 2
  print num_cores


  # print zip(trials, offsets)
  ctr = 0
  params = itertools.product(trials, offsets)
  # for t, o in params:
  #   print t
  #   print o
  #   ctr += 1
  # print ctr

  # exit()

  # avg_flow_per_block(read_path, write_path=write_path, offset=10, mx_scale=5, step=30)
  results = Parallel(n_jobs=num_cores)(delayed(avg_flow_per_block)(t, write_path=write_path, offset=o) for t, o in params)
  print results


def par_avg_flow_per_trial(fn_list, params, num_cores=None):
  """
    Compute average frame per trial using multiprocessing.
  """
  print "PAR AVG FLOW PER TRIAL"
  # fn_list = fn_list[:2]
  if num_cores is None:
    num_cores = multiprocessing.cpu_count()
    num_cores -= 2
  print 'Computing trial averages for %s trials using %s/%s cores' % (len(fn_list), num_cores, multiprocessing.cpu_count())

  if 'write_path' in params:
    raise KeyError("Invalid key 'write_path'. Use 'write_stem' for parallel methods.")
  # if 'write_stem' in params:
  #   wfn_list = [params['write_stem'] + '_%s' % os.path.basename(f)for f in fn_list]
  #   del params['write_stem']
  # else:
  #   wfn_list = [None for f in fn_list]

  write_stem = None
  if 'write_stem' in params:
    write_stem = params['write_stem']
    del params['write_stem']
  print write_stem
  # results = Parallel(n_jobs=num_cores)(delayed(avg_flow_per_trial)(fn, write_path=wfn, **params) for fn, wfn in zip(fn_list, wfn_list))
  results = Parallel(n_jobs=num_cores)(delayed(_par_avg_flow_per_trial_runner)(fn, write_stem, params) for fn in fn_list)

  print results


def _par_avg_flow_per_trial_runner(fn, write_stem, params):
  # require offset for writing purposes
  if 'offset' not in params:
    raise KeyError("Params must have key 'offset' when running parallel.")
  # compute avg frame
  avg_frame = avg_flow_per_trial(fn, write_path=None, **params)
  # write avg frame data
  if write_stem is not None:
    # wfn = os.path.join(os.path.dirname(fn), write_stem + '_offset(%s)_%s' % (params['offset'], os.path.basename(fn)))
    wfn = os.path.join(os.path.dirname(fn), write_stem + '_%s' % os.path.basename(fn))
    touch_dir(os.path.dirname(wfn))
    sp.io.savemat(wfn, {'avg_frame': avg_frame, 'offset': params['offset']})


def par_sum_and_count_lr(fn_list, num_cores=None):
  """
    Attempt to speed up repeated load-and-accumulate processes by distributing
    work to a worker pool.
    NOTE: Seems to work slower than using one process. This might have to do with
    using an external drive.
  """
  if num_cores is None:
    num_cores = multiprocessing.cpu_count()
    # num_cores -= 2
  print 'Accumulating trial averages for %s trials using %s/%s cores' % (len(fn_list), num_cores, multiprocessing.cpu_count())

  # Break list into groups for mutliprocessing
  fn_group_list = partition_list(fn_list, num_cores)
  print 'Partitioned list sizes: %s' % map(len, fn_group_list).__str__()
  # exit()
  print fn_group_list[0]
  # exit()
  start = time.time()
  results = Parallel(n_jobs=num_cores, verbose=5)(delayed(_par_sum_and_count_lr_runner)(fn) for fn in fn_group_list)
  print time.time() - start
  print results

  left_ctr, right_ctr = 0, 0
  left_avg, right_avg = None, None
  for r in results:
    r = results[0]
    if left_avg is None and right_avg is None:
      left_avg = np.zeros_like(r['left_sum'])
      right_avg = np.zeros_like(r['right_sum'])
    left_avg += r['left_sum']
    right_avg += r['right_sum']
    left_ctr += r['left_ctr']
    right_ctr += r['right_ctr']

  right_avg /= float(right_ctr)
  left_avg /= float(left_ctr)

  return {'right_avg': right_avg, 'left_avg': left_avg}


def _par_sum_and_count_lr_runner(fn_list):
  left_ctr, right_ctr = 0, 0
  temp_data = sp.io.loadmat(fn_list[0])
  left_sum = np.zeros_like(temp_data['avg_frame'])
  right_sum = np.zeros_like(temp_data['avg_frame'])
  ctr = 0
  for fn in fn_list:
    # if left_ctr > 1 and right_ctr > 1:
      # return {'left_sum': left_sum, 'left_ctr': left_ctr, 'right_sum': right_sum, 'right_ctr': right_ctr}
    ctr += 1
    print ctr
    print '%s, %s' % (left_ctr, right_ctr)
    temp_data = sp.io.loadmat(fn)
    if 'right' in fn:
      right_sum += temp_data['avg_frame']
      right_ctr += 1
    elif 'left' in fn:
      left_sum += temp_data['avg_frame']
      left_ctr += 1
    else:
      raise ValueError("Trial must be either left or right, found: %s" % fn)

  return {'left_sum': left_sum, 'left_ctr': left_ctr, 'right_sum': right_sum, 'right_ctr': right_ctr}


############################
####  @S VISUALIZATION  ####
############################

def _draw_avg_flow_circle(img, avg_flow_frame, mx_scale, c=(0, 0, 255)):
  h, w = img.shape[:2]
  x_avg = avg_flow_frame[..., 0]
  y_avg = avg_flow_frame[..., 1]
  mag_avg = np.sqrt(x_avg*x_avg + y_avg*y_avg)
  y_tot_avg = y_avg[mag_avg > 1].mean() * mx_scale
  x_tot_avg = x_avg[mag_avg > 1].mean() * mx_scale
  cv2.circle(img, (w/2, h/2), 50, c, 2)
  cv2.circle(img, (w/2, h/2), 3, c, -1)
  cv2.line(img, (w/2, h/2), (w/2 + int(x_tot_avg), h/2 + int(y_tot_avg)), c, 2)
  return img


def viz_avg_hist(avg_frame, read_filename=''):
  print avg_frame.shape
  avg_x = avg_frame[:, :, 0].flatten()
  avg_y = avg_frame[:, :, 1].flatten()

  # avg_x[:] = -3
  # avg_y[:] = -8

  print avg_x.shape
  print avg_y.shape
  print avg_x.dtype
  xedges = np.arange(-10, 10+1, 0.8)
  yedges = np.arange(-10, 10+1, 0.8)
  avg_hist, xedges, yedges = np.histogram2d(-avg_y, avg_x, bins=(xedges, yedges))
  print avg_hist.shape

  plt.subplot(1, 3, 3)
  # im = plt.imshow(avg_hist, interpolation='nearest', origin='low',
                # extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
  im = plt.imshow(avg_hist, interpolation='nearest', origin='low',
                extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]])
  plt.xlabel("Image X")
  plt.xlabel("Flow X (screen coords)")
  plt.ylabel("Flow Y (screen coords)")
  plt.colorbar()

  plt.subplot(1, 3, 1)
  plt.hist(avg_x.flat, bins=50)
  plt.title("Histogram of Average Flow for Channel 0\n%s" % os.path.basename(read_filename))
  plt.xlabel("Flow Value")
  plt.ylabel("Number of Occurrences")
  plt.subplot(1, 3, 2)
  plt.hist(avg_y.flat, bins=50)
  plt.title("Histogram of Average Flow for Channel 1\n%s" % os.path.basename(read_filename))
  plt.xlabel("Flow Value")
  plt.ylabel("Number of Occurrences")
  plt.show()


def viz_avg_hist_per_trial(read_filename):
  avg_frame = avg_flow_per_trial(read_filename)
  viz_avg_hist(avg_frame, read_filename=read_filename)


def viz_avg_hist_per_block(block_path):
  avg_frames = avg_flow_per_block(block_path)
  right_avg = avg_frames['right']
  left_avg = avg_frames['left']
  viz_avg_hist(right_avg, '/%s' % 'Right '+block_path)
  viz_avg_hist(left_avg, '/%s' % 'Left '+block_path)


def viz_avg_summary_trial(avg_frame, write_path=None, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, display=False, title_params=None):
  if title_params is None:
    title_params = {}
    title_params = {'read_filename': '/None', 'offset': None}
  print "VIZ AVG SUMMARY TRIAL"
  h, w = avg_frame.shape[:2]

  x_avg = avg_frame[..., 0]
  y_avg = avg_frame[..., 1]
  mag_avg = np.sqrt(x_avg*x_avg + y_avg*y_avg)

  # plt.hist(x_avg.flat, bins=100)
  # plt.hist(y_avg.flat, bins=100)
  # plt.hist(mag_avg.flat, bins=100)
  # plt.show()

  # # Get total average
  # y_tot_avg = avg_frame[..., 1].mean() * mx_scale
  # x_tot_avg = avg_frame[..., 0].mean() * mx_scale
  # Get total average of pixels with non-zero flow
  # print x_avg[np.isclose(mag_avg, 1)].shape
  y_tot_avg = y_avg[mag_avg > 1].mean() * mx_scale
  x_tot_avg = x_avg[mag_avg > 1].mean() * mx_scale
  print 'X Avg: %s, Y Avg: %s' % (x_tot_avg, y_tot_avg)

  avg_frame = avg_frame.astype(np.float32)
  avg_frame *= mx_scale
  hsv_img = draw_flow(avg_frame, np.ones((h, w, 1), dtype=np.float32), step=step).astype(np.float64)
  # hsv_img = draw_hsv(avg_frame)
  # print hsv_img.dtype
  # print hsv_img.min()
  # print hsv_img.max()
  cv2.circle(hsv_img, (w/2, h/2), 50, (0, 0, 255), 2)
  cv2.circle(hsv_img, (w/2, h/2), 3, (0, 0, 255), -1)
  cv2.line(hsv_img, (w/2, h/2), (w/2 + int(x_tot_avg), h/2 + int(y_tot_avg)), (0, 0, 255), 2)

  # plt.imshow(hsv_img / hsv_img.max())

  fig = plt.figure(figsize=(20, 10))
  plt.subplot2grid((3, 3), (0, 0), rowspan=3)
  plt.imshow(hsv_img / hsv_img.max())
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.title("Sampled Vector Flow for\n%s, offset: %s" % (os.path.basename(title_params['read_filename']), title_params['offset']))

  plt.subplot2grid((3, 3), (0, 1), rowspan=3)
  hsv_img2 = draw_hsv(avg_frame)
  plt.imshow(hsv_img2)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.title("Average HSV Flow for\n%s, offset: %s" % (os.path.basename(title_params['read_filename']), title_params['offset']))

  plt.subplot2grid((3, 3), (0, 2))
  circ = make_hsv_key()
  circ = draw_hsv(circ)
  cw, ch = circ.shape[:2]
  plt.imshow(circ, extent=[-cw/2, cw/2, -ch/2, ch/2])
  plt.title("Color Key")

  # plt.subplot2grid((3,3), (0, 2))
  avg_x = avg_frame[:, :, 0].flatten()
  avg_y = avg_frame[:, :, 1].flatten()
  # xedges = np.arange(-10, 10+1, 0.8)
  # yedges = np.arange(-10, 10+1, 0.8)
  # avg_hist, xedges, yedges = np.histogram2d(-avg_y, avg_x, bins=(xedges, yedges))
  # print avg_hist.shape
  # im = plt.imshow(avg_hist, interpolation='nearest', origin='low',
  #               extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]])
  # plt.xlabel("Image X")
  # plt.xlabel("Flow X (screen coords)")
  # plt.ylabel("Flow Y (screen coords)")
  # plt.colorbar()

  plt.subplot2grid((3, 3), (1, 2))
  plt.hist(avg_x.flat, bins=50)
  plt.title("Histogram of Average Flow for X (ch0)")
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  # plt.xlabel("Flow Value")
  plt.ylabel("Number of Occurrences")
  plt.subplot2grid((3, 3), (2, 2))
  plt.hist(avg_y.flat, bins=50)
  plt.title("Histogram of Average Flow for Y (ch1)")
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  # plt.xlabel("Flow Value")
  plt.ylabel("Number of Occurrences")

  if write_path is not None:
    offset = title_params['offset']
    wfn = os.path.join(write_path, 'offset(%s)_scale(%s)_step(%s)_trial_avg.png' % (offset, mx_scale, step))
    touch_dir(os.path.dirname(wfn))
    plt.savefig(wfn)
  if display:
    plt.show()


def viz_avg_summary_lr(left_avg, right_avg, write_path=None, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, display=False, title_params=None):
  if title_params is None:
    title_params = {}
    title_params = {'block_path': '/None', 'offset': None}
  print "VIZ AVG SUMMARY LEFT-RIGHT"

  left_avg *= mx_scale
  right_avg *= mx_scale

  # HSV Visualization
  right_hsv = draw_hsv(right_avg)
  right_hsv = _draw_avg_flow_circle(right_hsv, right_avg, 1, c=(255, 0, 0))
  left_hsv = draw_hsv(left_avg)
  left_hsv = _draw_avg_flow_circle(left_hsv, left_avg, 1, c=(0, 0, 255))

  # # Vector Visualization
  # h, w = right_avg.shape[:2]
  # step = 15
  # right_hsv = draw_flow(first_frame, right_avg, step=step).astype(np.float64)
  # right_hsv /= right_hsv.max()
  # left_hsv = draw_flow(first_frame, left_avg, step=step).astype(np.float64)
  # left_hsv /= left_hsv.max()

  fig = plt.figure(figsize=(21, 15))
  plt.subplot2grid((3,3), (0, 0), rowspan=3)
  plt.imshow(left_hsv)
  plt.title("Average Flow for Left Trials Block:\n%s, offset: %s, scale: %s" % (os.path.basename(title_params['block_path']), title_params['offset'], mx_scale))
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())

  plt.subplot2grid((3,3), (0, 1), rowspan=3)
  plt.imshow(right_hsv)
  plt.title("Average Flow for Right Trials Block:\n%s, offset: %s, scale: %s" % (os.path.basename(title_params['block_path']), title_params['offset'], mx_scale))
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())

  plt.subplot2grid((3,3), (0, 2))
  circ = make_hsv_key()
  circ = draw_hsv(circ)
  cw, ch = circ.shape[:2]
  plt.imshow(circ, extent=[-cw/2, cw/2, ch/2, -ch/2])
  plt.title("HSV Flow Color Key")
  plt.xlabel("X Component")
  plt.ylabel("Y Component")


  right_avg_x = right_avg[:, :, 0].flatten()
  right_avg_y = right_avg[:, :, 1].flatten()
  left_avg_x = left_avg[:, :, 0].flatten()
  left_avg_y = left_avg[:, :, 1].flatten()
  # xedges = np.arange(-10, 10+1, 0.8)
  # yedges = np.arange(-10, 10+1, 0.8)
  # avg_hist, xedges, yedges = np.histogram2d(-avg_y, avg_x, bins=(xedges, yedges))
  # im = plt.imshow(avg_hist, interpolation='nearest', origin='low',
  #               extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]])
  # plt.xlabel("Image X")
  # plt.xlabel("Flow X (screen coords)")
  # plt.ylabel("Flow Y (screen coords)")
  # plt.colorbar()

  plt.subplot2grid((3, 3), (1, 2))
  plt.hist(right_avg_x.flat, bins=50, color='r', alpha=0.5, label='right')
  plt.hist(left_avg_x.flat, bins=50, color='b', alpha=0.5, label='left')
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.title("Histogram of Average Flow for X (ch0)")
  # plt.xlabel("Flow Value")
  plt.ylabel("Number of Occurrences")
  plt.legend(loc='best')

  plt.subplot2grid((3, 3), (2, 2))
  plt.hist(right_avg_y.flat, bins=50, color='r', alpha=0.5, label='right')
  plt.hist(left_avg_y.flat, bins=50, color='b', alpha=0.5, label='left')
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.title("Histogram of Average Flow for Y (ch1) ")
  # plt.xlabel("Flow Value")
  plt.ylabel("Number of Occurrences")
  plt.legend(loc='best')

  if write_path is not None:
    touch_dir(os.path.dirname(write_path))
    plt.savefig(write_path + '.png')
  if display:
    plt.show()


def _sanity_check_histograms():
  test_grid = np.mgrid[-255:255, -255:255].transpose((2, 1, 0))
  test_grid = test_grid.astype(np.float64)
  test_x = test_grid[:, :, 0]
  test_y = test_grid[:, :, 1]

  test_x = np.sign(test_x) * 5
  test_y = np.sign(test_y) * 10
  test_grid[:, : ,0] = test_x
  test_grid[:, :, 1] = test_y
  test_grid *= 2
  test_hsv = draw_hsv(test_grid)
  z = np.zeros_like(test_hsv)
  z = z[:,:,0]
  gray = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
  test_vx = draw_flow(test_grid, z)

  plt.subplot(1, 3, 1)
  plt.imshow(test_grid[:, :, 0])
  plt.title("Test Grid X (ch0)")
  plt.subplot(1, 3, 2)
  plt.title("Test Grid Y (ch1)")
  plt.imshow(test_grid[:, :, 1])
  plt.subplot(1, 3, 3)
  plt.title("Test Grid")
  test_grid = np.dstack((test_grid, np.ones_like(test_grid[:, :, 0])))
  plt.imshow(test_grid / 2**0.5)
  plt.colorbar()
  plt.show()

  plt.figure(figsize=(14, 6))
  key = make_hsv_key()
  plt.subplot(1, 3, 1)
  plt.imshow(key[:, :, 0])
  plt.title("X Component of Flow (ch0)")
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.colorbar()
  plt.subplot(1, 3, 2)
  plt.imshow(key[:, :, 1])
  plt.title("Y Component of Flow (ch1)")
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.colorbar()
  plt.subplot(1, 3, 3)
  hsv_key = draw_hsv(key)
  cw, ch = hsv_key.shape[:2]
  plt.imshow(hsv_key, extent=[-cw/2, cw/2, ch/2, -ch/2])
  plt.title("HSV Flow Key")
  plt.xlabel("X Component")
  plt.ylabel("Y Component")
  plt.show()

  plt.figure(figsize=(22, 8))
  plt.subplot2grid((2, 4), (0, 0))
  plt.imshow(test_grid[:, :, 0])
  plt.title("Test Grid\nX Component of Flow (ch0)")
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.colorbar()
  plt.subplot2grid((2, 4), (1, 0))
  plt.title("Test Grid\n Y Component of Flow (ch1)")
  plt.imshow(test_grid[:, :, 1])
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.colorbar()
  plt.subplot2grid((2, 4), (0, 1), rowspan=4)
  plt.imshow(test_vx)
  plt.title("Test Grid\nVector Flow")
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.subplot2grid((2, 4), (0, 2), rowspan=4)
  plt.title("Test Grid\nHSV Flow")
  plt.imshow(test_hsv)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.subplot2grid((2, 4), (0, 3), rowspan=4)
  # plt.gca().tight_layout()
  plt.tight_layout(pad=3)
  plt.title("HSV Flow Key")
  plt.xlabel("X Component")
  plt.ylabel("Y Component")
  plt.imshow(hsv_key, extent=[-cw/2, cw/2, ch/2, -ch/2])
  # plt.tight_layout()
  # plt.show()
  plt.savefig('/Users/raygon/Desktop/test_grid_flow.png')

  # plt.subplot(1, 2, 1)
  # plt.hist(test_x.flat, bins=50)
  # plt.title("Histogram of Average Flow for Channel 0")
  # plt.subplot(1, 2, 2)
  # plt.hist(test_y.flat, bins=50)
  # plt.title("Histogram of Average Flow for Channel 1")
  # plt.show()


def _get_raw_vid_avg_from_fn_list(raw_vid_path, block_path, fn_list, offset):
  print "POP"
  raw_frames_avg = None
  block_dir = os.path.basename(block_path)
  for fn in fn_list:
    rfn = os.path.basename(fn).split('_Dense')[0] + '_StartToHit.mat'
    rfn = os.path.join(raw_vid_path, block_dir, rfn)
    print '-X: %s' % (rfn)
    # load raw vids
    raw_vid_obj = gt.GoalieTrial.from_file(rfn)
    # slice raw frames
    offset = -abs(offset)
    if offset < 0:
      # raw_frames = easy_slice(raw_vid_obj, offset - 1)[..., :-1]
      raw_frames = symmetric_easy_slice(raw_vid_obj, offset - 1)[..., :-2]
    elif offset > 0:
      print 'b:%s ' % raw_vid_obj.frameLaunchEnd
      raw_vid_obj.frameLaunchEnd -= 1
      print 'a:%s ' % raw_vid_obj.frameLaunchEnd
      raw_frames = easy_slice(raw_vid_obj, offset)
    else:
      raise ValueError('Argument "offset" should be nonzero.')
    # # convert to gray
    raw_frames_shape = raw_frames.shape
    raw_frames = np.mean(raw_frames, axis=2)
    raw_frames = np.dstack((raw_frames, raw_frames, raw_frames))
    raw_frames = raw_frames.reshape(raw_frames_shape)
    # intialize
    if raw_frames_avg is None:
      raw_frames_avg = np.zeros_like(raw_frames)
    # accumulate
    raw_frames_avg += raw_frames
  print len(fn_list)
  # assert np.all(raw_frames_avg / float(len(fn_list)) == raw_frames)
  return raw_frames_avg / float(len(fn_list))
  # # convert to gray
  # raw_frames_shape = raw_frames_avg.shape
  # raw_frames_gray = np.mean(raw_frames_avg, axis=2)
  # raw_frames_gray = np.dstack((raw_frames_gray, raw_frames_gray, raw_frames_gray))
  # raw_frames_gray = raw_frames.reshape(raw_frames_shape)
  # plt.imshow(raw_frames[..., 0])
  # plt.show()
  # exit()
  # print raw_frames_avg.shape
  # return raw_frames_avg

# def get_raw_vid(raw_vid_path, block_path, left_fn_list, right_fn_list):
#   print "GETTING RAW VID"
#   block_dir = os.path.basename(block_path)
#   left_rfn_list = [os.path.join(raw_vid_path, block_path, f) for f in left_fn_list]
#   right_rfn_list = [os.path.join(raw_vid_path, block_path, f) for f in right_fn_list]
#   print left_rfn_list
#   exit()
#   # for fn in left_fn_list:
#   left_rfn = os.path.basename(left_fn).split('_Dense')[0] + '_StartToHit.mat'
#   right_rfn = os.path.basename(right_fn).split('_Dense')[0] + '_StartToHit.mat'
#   left_rfn = os.path.join(raw_vid_path, block_dir, left_rfn)
#   right_rfn = os.path.join(raw_vid_path, block_dir, right_rfn)
#   # load raw vids
#   left_raw_obj = gt.GoalieTrial.from_file(left_rfn)
#   right_raw_obj = gt.GoalieTrial.from_file(right_rfn)
#   # slice raw frames
#   if offset < 0:
#     left_raw_frames = easy_slice(left_raw_obj, offset - 1)[..., :-1]
#     right_raw_frames = easy_slice(right_raw_obj, offset - 1)[..., :-1]
#   else:
#     raise NotImplementedError
#   # convert to gray
#   raw_frames_shape = left_raw_frames.shape
#   left_raw_frames = np.mean(left_raw_frames, axis=2)
#   right_raw_frames = np.mean(right_raw_frames, axis=2)
#   left_raw_frames = np.dstack((left_raw_frames, left_raw_frames, left_raw_frames))
#   right_raw_frames = np.dstack((right_raw_frames, right_raw_frames, right_raw_frames))
#   left_raw_frames = left_raw_frames.reshape(raw_frames_shape)
#   right_raw_frames = right_raw_frames.reshape(raw_frames_shape)
#   print left_raw_frames.shape
#   print right_raw_frames.shape


def viz_frame_compare_vids_trial(left_fn, right_fn, raw_vid_path=None, write_path=None, offset=OFFSET_DFLT, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, display=False, force=False):
  # load flows
  left_flow_obj = ft.FlowTrial.from_file(left_fn)
  right_flow_obj = ft.FlowTrial.from_file(right_fn)

  # slice flow frames
  # left_frames = easy_slice(left_flow_obj, offset)
  # right_frames = easy_slice(right_flow_obj, offset)
  offset = -abs(offset)
  left_frames = symmetric_easy_slice(left_flow_obj, offset)
  right_frames = symmetric_easy_slice(right_flow_obj, offset)

  print left_frames.shape
  print right_frames.shape
  # exit()

  # scale flow frames for viz
  mx_scale = 5
  left_frames *= mx_scale
  right_frames *= mx_scale

  if raw_vid_path is not None:
    block_dir = os.path.basename(os.path.dirname(os.path.dirname(left_fn)))
    # print block_dir
    # left_rfn = os.path.basename(left_fn).split('_Dense')[0] + '_StartToHit.mat'
    # right_rfn = os.path.basename(right_fn).split('_Dense')[0] + '_StartToHit.mat'
    # left_rfn = os.path.join(raw_vid_path, block_dir, left_rfn)
    # right_rfn = os.path.join(raw_vid_path, block_dir, right_rfn)
    # # load raw vids
    # left_raw_obj = gt.GoalieTrial.from_file(left_rfn)
    # right_raw_obj = gt.GoalieTrial.from_file(right_rfn)
    # # slice raw frames
    # if offset < 0:
    #   left_raw_frames = easy_slice(left_raw_obj, offset - 1)[..., :-1]
    #   right_raw_frames = easy_slice(right_raw_obj, offset - 1)[..., :-1]
    # else:
    #   raise NotImplementedError
    # # convert to gray
    # # raw_frames_shape = left_raw_frames.shape
    # # left_raw_frames = np.mean(left_raw_frames, axis=2)
    # # right_raw_frames = np.mean(right_raw_frames, axis=2)
    # # left_raw_frames = np.dstack((left_raw_frames, left_raw_frames, left_raw_frames))
    # # right_raw_frames = np.dstack((right_raw_frames, right_raw_frames, right_raw_frames))
    # # left_raw_frames = left_raw_frames.reshape(raw_frames_shape)
    # # right_raw_frames = right_raw_frames.reshape(raw_frames_shape)
    # print left_raw_frames.shape
    # print right_raw_frames.shape

    left_raw_frames = _get_raw_vid_avg_from_fn_list(os.path.join(raw_vid_path, block_dir), '/', [left_fn], offset)
    right_raw_frames = _get_raw_vid_avg_from_fn_list(os.path.join(raw_vid_path, block_dir), '/', [right_fn], offset)


  # make viz
  title_params = {
                  'left_title': os.path.basename(left_fn).split('_Dense')[0],
                  'right_title': os.path.basename(right_fn).split('_Dense')[0],
                  'offset': offset,
                  'mx_scale': mx_scale}
  overlay_frames = {'left_overlay_frames': left_raw_frames,
                    'right_overlay_frames': right_raw_frames,
                    'blend_k': .5}
  if raw_vid_path is not None:
    overlay_str = '_overlay'
  else:
    overlay_str = ''
  wfn = 'avg_frame_for_offset(%s)_%s_vs_%s%s.mp4' % (offset, os.path.basename(left_fn).split('_Dense')[0], os.path.basename(right_fn).split('_Dense')[0], overlay_str)
  wfn = os.path.join(write_path, wfn)
  touch_dir(wfn)
  print "writing viz to %s" % wfn
  viz_frame_compare_vids(left_frames, right_frames, overlay_frames, write_path=wfn, title_params=title_params, offset=offset, mx_scale=mx_scale, step=step, display=display)


def viz_frame_compare_vids_block_avg(block_path, raw_vid_path=None, write_path=None, offset=OFFSET_DFLT, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, display=False, force=False):
  left_frames_avg, right_frames_avg = None, None
  left_ctr, right_ctr = 0, 0
  left_fn_list, right_fn_list = [], []
  # build file lists
  for root, dirs, files in os.walk(block_path):
    for name in [f for f in files if f.endswith('.mat') and 'cache' not in f and 'avg' not in f]:
      fn = os.path.join(root, name)
      # determine membership
      if 'left' in name:
        left_fn_list.append(fn)
      elif 'right' in name:
        right_fn_list.append(fn)
      else:
        raise ValueError("Frame must be of type left or right.")
  # Check cache
  p_str = 'offset(%s)_%s' % (offset, os.path.basename(block_path))
  cache_fn = os.path.join(block_path, 'block_frames_avg_cache', 'block_frames_avg_cache_%s.mat' % (p_str))
  if not force:
    print "Checking for cached file: %s" % cache_fn
    cache_data = load_cache(cache_fn)
    if cache_data is not None:
      # unpack cache data
      left_frames_avg = cache_data['left_frames_avg']
      right_frames_avg = cache_data['right_frames_avg']
      if offset != cache_data['offset']:
        raise ValueError("Cached data 'offset' does not match 'offset' argument.")
      print "Using cached values from %s" % cache_fn
  # compute averages, if needed
  if left_frames_avg is None or right_frames_avg is None or force:
    left_frames_avg = _get_avg_frames_from_fn_list(left_fn_list, offset)
    right_frames_avg = _get_avg_frames_from_fn_list(right_fn_list, offset)

    # write cache file
    print "writing cache file"
    touch_dir(os.path.dirname(cache_fn))
    sp.io.savemat(cache_fn, {'right_frames_avg': right_frames_avg,
                             'left_frames_avg': left_frames_avg,
                             'mx_scale': mx_scale,
                             'step': step,
                             'offset': offset,
                             'info:': 'block_frames_avg'})

  # handle raw vid frames
  if raw_vid_path is not None:
    left_raw_frames_avg, right_raw_frames_avg = None, None
    # Check cache
    p_str = 'offset(%s)_%s' % (offset, os.path.basename(block_path))
    raw_cache_fn = os.path.join(raw_vid_path, os.path.basename(block_path), 'block_frames_avg_cache', 'block_frames_avg_cache_%s.mat' % (p_str))
    print raw_cache_fn
    if not force:
    # if False:
      print "Checking for raw video cached file: %s" % raw_cache_fn
      cache_data = load_cache(raw_cache_fn)
      if cache_data is not None:
        # unpack cache data
        left_raw_frames_avg = cache_data['left_raw_frames_avg']
        right_raw_frames_avg = cache_data['right_raw_frames_avg']
        if offset != cache_data['offset']:
          raise ValueError("Cached data 'offset' does not match 'offset' argument.")
        print "Using cached values from %s" % raw_cache_fn
    if left_raw_frames_avg is None or right_raw_frames_avg is None or force:
      left_raw_frames_avg = _get_raw_vid_avg_from_fn_list(raw_vid_path, block_path, left_fn_list, offset)
      right_raw_frames_avg = _get_raw_vid_avg_from_fn_list(raw_vid_path, block_path, right_fn_list, offset)
      print "writing raw cache file"
      touch_dir(os.path.dirname(raw_cache_fn))
      sp.io.savemat(raw_cache_fn, {'right_raw_frames_avg': right_raw_frames_avg,
                               'left_raw_frames_avg': left_raw_frames_avg,
                               'mx_scale': mx_scale,
                               'step': step,
                               'offset': offset,
                               'info:': 'block_frames_avg'})
    # left_raw_frames_avg *= 2
    # right_raw_frames_avg *= 2

  # scale frames for viz
  mx_scale = 5
  left_frames_avg *= mx_scale
  right_frames_avg *= mx_scale

  # print left_frames.shape
  # print right_frames.shape

  # set title params for viz
  if raw_vid_path is not None:
    overlay_frames = {'left_overlay_frames': left_raw_frames_avg,
                      'right_overlay_frames': right_raw_frames_avg,
                      'blend_k': 0.5}
  else:
    overlay_frames = None

  title_params = {'master_title': 'Avg Frame for Block: %s' % os.path.basename(block_path),
                  'offset': offset,
                  'mx_scale': mx_scale}

  # make viz
  if raw_vid_path is not None:
    overlay_str = '_overlay'
  else:
    overlay_str = ''
  wfn = 'avg_frame_for_block_offset(%s)_%s%s.mp4' % (offset, os.path.basename(block_path), overlay_str)
  wfn = os.path.join(write_path, wfn)
  touch_dir(wfn)
  print "writing viz to %s" % wfn
  viz_frame_compare_vids(left_frames_avg, right_frames_avg, overlay_frames=overlay_frames, write_path=wfn, title_params=title_params, offset=offset, mx_scale=mx_scale, step=step, display=display)


def _get_avg_frames_from_fn_list(fn_list, offset):
  frames_avg = None
  for fn in fn_list:
    # load flow
    flow_obj = ft.FlowTrial.from_file(fn)

    # slice frames
    # frames = easy_slice(flow_obj, offset)
    offset = -abs(offset)
    frames = symmetric_easy_slice(flow_obj, offset)

    # intialize
    if frames_avg is None:
      frames_avg = np.zeros_like(frames)

    # determine membership and accumulate
    frames_avg += frames
  return frames_avg / float(len(fn_list))


# def viz_frame_compare_vids_block_avg(block_path, raw_vid_path=None, write_path=None, offset=OFFSET_DFLT, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, display=False, force=False):
#   left_frames_avg, right_frames_avg = None, None
#   left_ctr, right_ctr = 0, 0
#   left_fn_list, right_fn_list = [], []
#   # Check cache
#   p_str = 'offset(%s)_%s' % (offset, os.path.basename(block_path))
#   cache_fn = os.path.join(block_path, 'block_frames_avg_cache', 'block_frames_avg_cache_%s.mat' % (p_str))
#   print "tttt"
#   if not force:
#     print "Checking for cached file: %s" % cache_fn
#     cache_data = load_cache(cache_fn)
#     if cache_data is not None:
#       # unpack cache data
#       left_frames_avg = cache_data['left_frames_avg']
#       right_frames_avg = cache_data['right_frames_avg']
#       if offset != cache_data['offset']:
#         raise ValueError("Cached data 'offset' does not match 'offset' argument.")
#       print "Using cached values from %s" % cache_fn
#   if left_frames_avg is None or right_frames_avg is None or force:
#     for root, dirs, files in os.walk(block_path):
#       for name in [f for f in files if f.endswith('.mat') and 'cache' not in f and 'avg' not in f]:
#         print name
#         fn = os.path.join(root, name)
#         # load flow
#         flow_obj = ft.FlowTrial.from_file(fn)
#         # exit()

#         # slice frames
#         frames = easy_slice(flow_obj, offset)

#         # intialize
#         if left_frames_avg is None and right_frames_avg is None:
#           left_frames_avg = np.zeros_like(frames)
#           right_frames_avg = np.zeros_like(frames)

#         # determine membership and accumulate
#         if 'left' in name:
#           left_ctr += 1
#           left_frames_avg += frames
#           left_fn_list.append(fn)
#           print "%s, %s" % (left_frames_avg.max(), left_frames_avg.shape.__str__())
#         elif 'right' in name:
#           right_ctr += 1
#           right_frames_avg += frames
#           right_fn_list.append(fn)
#         else:
#           raise ValueError("Frame must be of type left or right.")

#     print "l: %s, r: %s" % (left_ctr, right_ctr)

#     left_frames_avg = left_frames_avg / float(left_ctr)
#     right_frames_avg = right_frames_avg / float(right_ctr)

#     # write cache file
#     print "writing cache file"
#     touch_dir(os.path.dirname(cache_fn))
#     sp.io.savemat(cache_fn, {'right_frames_avg': right_frames_avg,
#                              'left_frames_avg': left_frames_avg,
#                              'mx_scale': mx_scale,
#                              'step': step,
#                              'offset': offset,
#                              'info:': 'block_frames_avg'})

#   # handle raw vid frames
#   if raw_vid_path is not None:
#     left_raw_frames_avg, right_raw_frames_avg = get_raw_vid(raw_vid_path, block_path, left_fn_list, right_fn_list)


#   # scale frames for viz
#   # mx_scale = 10
#   left_frames_avg *= mx_scale
#   right_frames_avg *= mx_scale

#   # print left_frames.shape
#   # print right_frames.shape

#   # set title params for viz
#   title_params = {'master_title': 'Avg Frame for Block: %s' % os.path.basename(block_path),
#                   'offset': offset,
#                   'mx_scale': mx_scale}

#   # make viz
#   viz_frame_compare_vids(left_frames_avg, right_frames_avg, write_path=write_path, title_params=title_params, offset=offset, mx_scale=mx_scale, step=step, display=display)



def viz_frame_compare_vids_all_avg(block_path, write_path=None, offset=OFFSET_DFLT, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, display=False, force=False):
  pass


def viz_frame_compare_vids(left_array, right_array, overlay_frames=None, write_path=None, title_params=None, offset=None, mx_scale=MX_SCALE_DFLT, step=STEP_DFLT, display=False, force=False):
  """
    Create a movie that compares the frames in the left and right arrays.
  """
  print "viz lr comp"
  print left_array.shape
  print right_array.shape

  if left_array.shape != right_array.shape:
    raise ValueError('Array dimensions must match.')

  # handle overlay frames
  if overlay_frames is not None:
    if 'left_overlay_frames' not in overlay_frames and 'right_overlay_frames' not in overlay_frames:
      raise KeyError('Argument overlay_frames should have keys left_overlay_frames and right_overlay_frames.')
    lovr_s = list(overlay_frames['left_overlay_frames'].shape)
    rovr_s = list(overlay_frames['right_overlay_frames'].shape)
    lovr_s[2] -= 1
    rovr_s[2] -= 1
    lovr_s = tuple(lovr_s)
    rovr_s = tuple(rovr_s)
    if left_array.shape != rovr_s and left_array.shape != lovr_s:
      print lovr_s
      print rovr_s
      print left_array.shape
      raise ValueError('Overlay array dimensions must match')
    if 'blend_k' not in overlay_frames:
      overlay_frames['blend_k'] = 0.5

  # handle title params defaults
  default_title_params = {'left_title': '',
                          'right_title': '',
                          'master_title': '',
                          'offset': ''}
  if title_params is None:
    title_params = default_title_params
  # else:
  #   for k in default_title_params:
  #     if k not in title_params:
  #       title_params[k] = default_title_params[k]

  fps=1
  dpi=100
  scl=1
  h, w, c, f = left_array.shape
  assert f == right_array.shape[-1]

  # convert flows to images
  left_hsv = map(draw_hsv, [left_array[..., i] for i in range(f)])
  right_hsv = map(draw_hsv, [right_array[..., i] for i in range(f)])
  # print len(left_hsv)
  # print left_hsv[0].shape
  # exit()

  pad = np.ones((h, 20, 3), dtype=left_hsv[0].dtype) * 255
  # v_pad = np.ones()
  # wfn = '/Users/raygon/Desktop/test_out_CV.mp4'
  wfn = write_path

  # Define the codec and create VideoWriter object
  fh, fw = h, w * 2 + pad.shape[1]
  fourcc = cv2.cv.CV_FOURCC(*'mp4v')
  # out = cv2.VideoWriter(wfn, fourcc, 20.0, (fw, fh))
  out = cvVideo.CVVideo(wfn, fourcc, fps, (fw, fh))

  # ctr = 0
  # for lfr, rfr in zip(left_hsv, right_hsv):
  #   ctr += 1

  #   font = cv2.FONT_HERSHEY_SIMPLEX
  #   # f_ctr = '%s_%s' % (f_ctr, offset + ctr)
  #   f_ctr = offset + ctr
  #   cv2.putText(lfr, 'Offset: %s' % f_ctr, (10, 50), font, 2, (255, 255, 255), 2, cv2.CV_AA)

  #   concat_frame = np.hstack((lfr, pad, rfr))
  #   # plt.imshow(concat_frame)
  #   # plt.show()
  #   # exit()
  #   print f_ctr
  #   # concat_frame = cv2.cvtColor(concat_frame, cv2.COLOR_RGB2BGR)
  #   print concat_frame.shape
  #   print concat_frame.dtype
  #   out.write(concat_frame)

  #   cv2.imshow('frame', concat_frame)
  #   if cv2.waitKey(1) & 0xFF == ord('q'):
  #     break
  # # Release everything if job is finished
  # out.release()
  # cv2.destroyAllWindows()

  with out:
    ctr = -1
    # for lfr, rfr in zip(left_hsv, right_hsv):
    print "######## %s" % len(left_hsv)
    for i in range(len(left_hsv)):
      lfr = left_hsv[i]
      rfr = right_hsv[i]
      if overlay_frames is not None:
        fr_dtype = lfr.dtype
        lovr = overlay_frames['left_overlay_frames'][..., i].astype(fr_dtype)
        rovr = overlay_frames['right_overlay_frames'][..., i].astype(fr_dtype)
        lovr = cv2.cvtColor(lovr, cv2.COLOR_RGB2BGR)
        rovr = cv2.cvtColor(rovr, cv2.COLOR_RGB2BGR)
        lfr = (1.0 - overlay_frames['blend_k']) * lfr + float(overlay_frames['blend_k']) * lovr
        rfr = (1.0 - overlay_frames['blend_k']) * rfr + float(overlay_frames['blend_k']) * rovr
        lfr = lfr.astype(fr_dtype)
        rfr = rfr.astype(fr_dtype)

      ctr += 1

      font = cv2.FONT_HERSHEY_SIMPLEX
      if offset < 0:
        f_ctr = offset + ctr
        if f_ctr == 0:
          ctr += 1
          f_ctr += 1
      else:
        if ctr == 0:
          ctr = 1
        f_ctr = '+%s' % ctr

      concat_frame = np.hstack((lfr, pad, rfr))
      concat_frame = cv2.cvtColor(concat_frame, cv2.COLOR_RGB2BGR)
      disp_step_y = 50
      disp_step_y_ctr = 0
      # optional display text
      if 'master_title' in title_params:
        disp_step_y_ctr += 1
        cv2.putText(concat_frame, '%s' % title_params['master_title'], (10, disp_step_y * disp_step_y_ctr), font, 1, (255, 255, 255), 2, cv2.CV_AA)
      if 'left_title' in title_params:
        disp_step_y_ctr += 1
        cv2.putText(concat_frame, 'Left Src: %s' % title_params['left_title'], (10, disp_step_y * disp_step_y_ctr), font, 1, (255, 255, 255), 2, cv2.CV_AA)
      if 'right_title' in title_params:
        disp_step_y_ctr += 1
        cv2.putText(concat_frame, 'Right Src: %s' % title_params['right_title'], (10, disp_step_y * disp_step_y_ctr), font, 1, (255, 255, 255), 2, cv2.CV_AA)
      if 'mx_scale' in title_params:
        disp_step_y_ctr += 1
        cv2.putText(concat_frame, 'Contrast Scaler: %s' % title_params['mx_scale'], (10, disp_step_y * disp_step_y_ctr), font, 1, (255, 255, 255), 2, cv2.CV_AA)
      if 'step' in title_params:
        disp_step_y_ctr += 1
        cv2.putText(concat_frame, 'Step: %s' % title_params['step'], (10, disp_step_y * disp_step_y_ctr), font, 1, (255, 255, 255), 2, cv2.CV_AA)
      # required display text
      disp_step_y_ctr += 1
      cv2.putText(concat_frame, 'Offset: %s' % f_ctr, (10, disp_step_y * disp_step_y_ctr), font, 1, (255, 255, 255), 2, cv2.CV_AA)
      hsv_key = draw_hsv(make_hsv_key())
      hsv_key = cv2.cvtColor(hsv_key, cv2.COLOR_RGB2BGR)
      hsv_key = scipy.misc.imresize(hsv_key, 0.5)
      concat_frame[10:10+hsv_key.shape[0], -hsv_key.shape[1]-10:-10, :] = hsv_key


      out.write(concat_frame)

      print concat_frame.shape
      print fw

      cv2.imshow('frame', concat_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # add black frame
    for i in range(3):
      out.write(np.zeros_like(concat_frame))
      cv2.imshow('frame', np.zeros_like(concat_frame))
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  # FFMpegWriter = manimation.writers['ffmpeg']
  # writer = FFMpegWriter(fps=fps, codec="libx264")

  # # compute figure sizes
  # pad = np.ones((h, 20, 3), dtype=left_hsv[0].dtype) * 255
  # fh, fw, fc = h, w * 2 + pad.shape[1], c

  # # fig = plt.figure(figsize=((w * 2 + 150) / 100.0, h / 100.0))
  # fig = plt.figure(figsize=(fw / 100.0, fh / 100.0))
  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # ax.set_aspect('equal')
  # ax.get_xaxis().set_visible(False)
  # ax.get_yaxis().set_visible(False)

  # # img_disp = ax.imshow(vid_array[..., 0], cmap='gray', interpolation='nearest')
  # img_disp = ax.imshow(np.zeros((fh, fw, 3)))
  # img_disp.set_clim([0, 1])
  # # fig.set_size_inches([scl * w / 100.0, scl * h / 100.0])
  # tight_layout()

  # wf = writer.saving(fig, wfn, dpi)
  # print fw

  # def update_frame(frame):
  #   # img_disp.set_data(cv2.cvtColor(left_frame, cv2.cv.CV_RGB2BGR))
  #   # img_disp.set_data(cv2.cvtColor(left_frame, cv2.cv.CV_BGR2RGB))
  #   img_disp.set_data(frame)
  #   writer.grab_frame()
  #   # cv2.waitKey(1000)

  # # return (wf, update_frame)
  # with wf:
  #   ctr = -1
  #   f_ctr = ''
  #   for lfr, rfr in zip(left_hsv, right_hsv):
  #     ctr += 1

  #     font = cv2.FONT_HERSHEY_SIMPLEX
  #     # f_ctr = '%s_%s' % (f_ctr, offset + ctr)
  #     f_ctr = offset + ctr
  #     cv2.putText(lfr, 'Offset: %s' % f_ctr, (10, 50), font, 2, (255, 255, 255), 2, cv2.CV_AA)

  #     concat_frame = np.hstack((lfr, pad, rfr))
  #     # plt.imshow(concat_frame)
  #     # plt.show()
  #     # exit()
  #     print f_ctr
  #     update_frame(concat_frame)

  #   # add black frame
  #   update_frame(np.zeros_like(concat_frame))


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

  # avg_flow_per_trial(test_fn, write_path=os.path.join(WRITE_PATH, 'testXXXX'), display=False)
  # avg_flow_per_block(READ_PATH, write_path=WRITE_PATH, offset=10, mx_scale=5, step=30, display=True, force=False)
  # avg_flow_all(path_to_blocks, write_path=WRITE_PATH, offset=-10, display=True, force=True)

  # _sanity_check_histograms()
  # viz_avg_hist_per_trial(test_fn)
  # viz_avg_hist_per_block(READ_PATH)

  # par_make_figs()

  # viz_frame_compare_vids_trial(test_fn, test_fn2, raw_vid_path=RAW_VID_READ_PATH,  write_path=WRITE_PATH, offset=-10, display=True, force=True)
  # viz_frame_compare_vids_trial(test_fn, test_fn2, raw_vid_path=RAW_VID_READ_PATH,  write_path=WRITE_PATH, offset=10, display=True, force=True)

  # viz_frame_compare_vids_block_avg(READ_PATH, raw_vid_path=RAW_VID_READ_PATH, write_path=WRITE_PATH, offset=-10, display=True, force=True)
  # viz_frame_compare_vids_block_avg(READ_PATH, raw_vid_path=RAW_VID_READ_PATH, write_path=WRITE_PATH, offset=10, display=True, force=True)

  viz_frame_compare_vids_block_avg(READ_PATH, write_path=WRITE_PATH, offset=-10, display=True, force=False)
  # viz_frame_compare_vids_block_avg(READ_PATH, write_path=WRITE_PATH, offset=10, display=True, force=True)



if __name__ == '__main__':
  main()
