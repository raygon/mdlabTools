import numpy as np
import scipy as sp
import cv2
from common import draw_str
import vizTools
import utilfx


def dense_track(vid_array, write_fx_list=[], display=False, out_dtype=None):
  """ Compute dense optical flow between frames in vid_array using Gunnar Farneback's
    algorithm.

    Args:
      vid_array (ndarray): Array containing image frames to use for calculating flow.
      Should have shape (height, width, channels, n_frames).
      wfn_dict (Optional [str]): Specifies the type of flow videos to write.
      Valid values are 'hsv', 'flow', and 'glitch'.
      display (Optional [bool]): If wfn_dict is not specified, and display
      is True, will display the flow results in hsv format.
      writer_type (optional(str)): Specifies the type of video writer to use. Must be 'cv' to
      use OpenCV's video writer, otherwise, it must be 'mplt' to use a matplotlib-based
      fallback. Defaults to 'mplt'. Note, OpenCV's writer doesn't work across
      multiple processes.
      fps (optional(int)): Specifies frame rate of the output videos. Defaults to 30 fps.
      mx_scale (optional(int)): Integer used to scale the hsv video.
    Returns:
      out (ndarray): Array containing the flow frames. This should have shape
      (height, width, 2, n_frames), where the zeroth-frame is all zeros and
      doesn't represent any flow.

  """
  logger = utilfx.get_logger(__name__)
  logger.info('Computing Dense flow for array with shape: %s, out vids: %s' % (vid_array.shape.__str__(), write_fx_list))
  h, w, c, f = vid_array.shape
  # if len(write_fx_list) == 0:
  #   write_fx_list = None
  if out_dtype is None:
    out = np.zeros((h, w, 2, f))
  else:
    out = np.zeros((h, w, 2, f), dtype=out_dtype)

  try:
    # open writers
    if len(write_fx_list) > 0:
      for wr in write_fx_list:
        print wr.wfn
    [x.__enter__() for x in write_fx_list]
    print write_fx_list

    for i in range(f):
      print i
      logger.debug("Computing Dense Flow on frame %s" % i)
      img = vid_array[..., i]

      # set things up if this is the first frame
      if i == 0:
        prev = img
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        continue

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
      prevgray = gray

      if display is not None or len(write_fx_list) > 0:
        [wfx.update_writer(flow, gray) for wfx in write_fx_list]
        if display:
          cv2.imshow('Dense Flow', vizTools.draw_hsv_with_key(flow))
          ch = 0xFF & cv2.waitKey(5)
          if ch == 27:
            break

      if out_dtype is not None:
          flow = utilfx.convert_precision(flow, out_dtype)

      out[..., i] = flow
  finally:
    # close writers
    [x.__exit__(None, None, None) for x in write_fx_list]
    if display:
      cv2.destroyAllWindows()

  if (not np.any(np.isclose(out[..., 0], 0))):
    logger.warn('Zeroth frame of dense flow is not zero, min:%s max: %s' % (out[..., 0].min(), out[..., 0].max()))
    raise ValueError('Zeroth frame of dense flow is not zero, min:%s max: %s' % (out[..., 0].min(), out[..., 0].max()))
  return out


def lk_track(vid_array, write_fx_list=[], display=False, out_dtype=None):
  """ Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
    for track initialization and back-tracking for match verification
    between frames.

    Args:
      vid_array (ndarray): Array containing image frames to use for calculating flow.
      Should have shape (height, width, channels, n_frames).
      wfn_dict (Optional [str]): Specifies the type of flow videos to write.
      Valid values are 'hsv', 'flow', and 'glitch'.
      display (Optional [bool]): If wfn_dict is not specified, and display
      is True, will display the flow results.
      writer_type (str): Specifies the type of video writer to use. Must be 'cv' to
      use OpenCV's video writer, otherwise, it must be 'mplt' to use a matplotlib-based
      fallback. Defaults to 'mplt'. Note, OpenCV's writer doesn't work across
      multiple processes.
      fps (int): Specifies frame rate of the output videos. Defaults to 30 fps.

    Returns:
      out (ndarray): Array containing the flow frames. This should have shape
      (height, width, 2, n_frames), where the zeroth-frame is all zeros and
      doesn't represent any flow.

  """
  logger = utilfx.get_logger(__name__)
  logger.info('Computing LK-flow on corners for array with shape: %s, out vids: %s' %
    (vid_array.shape.__str__(), write_fx_list))
  h, w, c, f = vid_array.shape
  out = np.zeros((h, w, 2, f))

  lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

  track_len = 2
  detect_interval = 1
  tracks = []
  frame_idx = 0
  is_init = False

  try:
    # open writers
    [x.__enter__() for x in write_fx_list]
    for i in range(vid_array.shape[-1]):
      print i
      logger.debug("Computing LK Sparse Flow on frame %s" % i)
      frame = vid_array[:, :, :, i]
      frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      vis = frame.copy()

      if not is_init:
        prev_gray = np.zeros_like(frame_gray)
        is_init = True
        continue

      if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        # filter the points to those with small amounts of motion between frames?
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
          if not good_flag:
            continue
          tr.append((x, y))
          if len(tr) > track_len:
            del tr[0]
          new_tracks.append(tr)
        tracks = new_tracks
        # flow is represented as (x_start, y_start, delta_x, delta_y)
        flow = [(t[0][0], t[0][1], t[1][0] - t[0][0], t[1][1] - t[0][1]) for t in tracks]

        x = np.array(flow)
        rows = np.floor(x[:, 0])
        cols = np.floor(x[:, 1])
        flow_x = sp.sparse.csr_matrix(((x[:, 2]), (cols, rows)), shape=(h, w))
        flow_y = sp.sparse.csr_matrix(((x[:, 3]), (cols, rows)), shape=(h, w))
        flow_frame = np.dstack((flow_x.toarray(), flow_y.toarray()))

        for tr in tracks:
          x0, y0 = tr[0][0], tr[0][1]
          cv2.circle(vis, (x0, y0), 2, (0, 255, 0), -1)
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (255, 0, 255))
        draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
        draw_str(vis, (20, 40), 'Frame: %d' % i)

        out[..., i] = flow_frame

      if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
          cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
          for x, y in np.float32(p).reshape(-1, 2):
            tracks.append([(x, y)])

      frame_idx += 1
      prev_gray = frame_gray
      if display or write_fx_list is not None:
        [wfx.update_writer(vis, None) for wfx in write_fx_list]
        if display:
          cv2.imshow('LK Flow', vis)
          ch = 0xFF & cv2.waitKey(300)
          if ch == 27:
            break

  finally:
    # close writers
    [x.__exit__(None, None, None) for x in write_fx_list]
    if display:
      cv2.destroyAllWindows()

  return out
