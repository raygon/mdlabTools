import time
import numpy as np
import scipy.misc
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
import utilfx
import cvVideo

VID_WRITER_TYPES = ['cv', 'mplt']


# def mplt_vid(fr=10):
#   plt.figure()
#   plt.show(block=False)
#   def imshow(img, clear_fx=None):
#     if clear_fx is None:
#       _clear_fx = plt.cla
#     _clear_fx()
#     plt.imshow(img)

# def mplt_vid(wait=1):
#   f = plt.figure()
#   ax = f.gca()
#   plt.show(block=False)
#   return (f, ax)


def circular_mask(r, n, (a, b)):
  """ Make a square circular mask.

    Args:
      r (int): Radius to use for mask. Any area outside the radius will be zero.
      n (int): Side length of output mask.
      (a, b) (int, int): Center of circular mask.

    Returns:
      circular_mask (np.array): A square circular mask with ones inside the
      radius, and zeros outside.
  """
  y, x = np.ogrid[-a:n-a, -b:n-b]
  mask = x*x + y*y <= r*r
  array = np.zeros((n, n))
  array[mask] = 1
  return array


def make_hsv_key():
  """ Make a key for interpreting flow images.

    Returns:
    circ: A [530, 530, 2] array that contains a horizontal gradient in the
    first channel and a vertical gradient in the second channel. Note, this
    needs to be processed by draw_hsv to be meaningful.
  """
  pad = 20
  r = 255
  n = r * 2 + pad
  s = n / 2
  circ = circular_mask(r, n, (s, s))
  grid = np.mgrid[-s:n-s, -s:n-s]
  circ = grid * circ
  circ = circ.transpose((2, 1, 0))
  return circ


def write_vid_hsv(vid_array, wfn, fourcc=cv2.cv.CV_FOURCC(*'mp4v'), mx_scale=3, overlay_key=True, overlay_offset=None, fr=15, overlay_text=None):
  """ Write the flow frames in the array, using OpenCV.

    Args:
      vid_array (np.array): A [w, h, 2, f] array containing f frames of flow data.
      mx_scale (optional(int)): Value used to scale hsv output.
      overlay_key (optional(bool)): Determines if displayed frames with contain
      an hsv in the top right corner. Default is True.
      fr (optional(int)): Frame rate to use for video display.
      overlay_text (optional(list)): A list of strings that will be drawn onto
      the displayed frame.
  """

  with cvVideo.CVVideo(wfn, fourcc, fr, (vid_array.shape[1], vid_array.shape[0])) as wf:
    for i in xrange(vid_array.shape[-1]):
      frame = vid_array[..., i]
      disp = draw_hsv(frame)
      disp *= mx_scale
      if overlay_key:
        hsv_key = draw_hsv(make_hsv_key())
        hsv_key = scipy.misc.imresize(hsv_key, 0.5)
        disp[:hsv_key.shape[0], -hsv_key.shape[1]:, :] = hsv_key
      disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
      # if overlay_text is not None or overlay_offset is not None:
      #   font = cv2.FONT_HERSHEY_SIMPLEX
      #   disp_step_y = 50
      #   disp_step_y_ctr = 0
      #   if overlay_offset is not None:
      #     offset_str = i + overlay_offset
      #     disp_step_y_ctr += 1
      #     cv2.putText(disp, 'Offset: %s' % offset_str, (10, disp_step_y * disp_step_y_ctr), font, 0.8, (255, 255, 255), 1, cv2.CV_AA)
      #   for s in overlay_text:
      #     disp_step_y_ctr += 1
      #     cv2.putText(disp, '%s' % s, (10, disp_step_y * disp_step_y_ctr), font, 0.8, (255, 255, 255), 1, cv2.CV_AA)
      wf.write(disp)
      # cv2.imshow('frame', disp)

      fr_wait = int(1000.0 / fr)
      fr_wait = 1
      if cv2.waitKey(fr_wait) & 0xFF == ord('q'):
          break
  cv2.destroyAllWindows()


def play_vid_hsv(vid_array, mx_scale=3, overlay_key=True, overlay_offset=None, fr=15, overlay_text=None):
  """ Play the flow frames in the array, using OpenCV.

    Args:
      vid_array (np.array): A [w, h, 2, f] array containing f frames of flow data.
      mx_scale (optional(int)): Value used to scale hsv output.
      overlay_key (optional(bool)): Determines if displayed frames with contain
      an hsv in the top right corner. Default is True.
      fr (optional(int)): Frame rate to use for video display.
      overlay_text (optional(list)): A list of strings that will be drawn onto
      the displayed frame.
  """
  print vid_array.shape[-1]
  logger = utilfx.get_logger(__name__)
  for i in xrange(vid_array.shape[-1]):
    frame = vid_array[..., i]
    disp = draw_hsv(frame)
    disp *= mx_scale
    if overlay_key:
      hsv_key = draw_hsv(make_hsv_key())
      # hsv_key = scipy.misc.imresize(hsv_key, 0.5)
      hsv_key = scipy.misc.imresize(hsv_key, 0.05)
      # print hsv_key.shape
      disp[:hsv_key.shape[0], -hsv_key.shape[1]:, :] = hsv_key
    disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
    if overlay_text is not None or overlay_offset is not None:
      font = cv2.FONT_HERSHEY_SIMPLEX
      disp_step_y = 50
      disp_step_y_ctr = 0
      if overlay_offset is not None:
        offset_str = i + overlay_offset
        disp_step_y_ctr += 1
        cv2.putText(disp, 'Offset: %s' % offset_str, (10, disp_step_y * disp_step_y_ctr), font, 0.8, (255, 255, 255), 1, cv2.CV_AA)
      for s in overlay_text:
        disp_step_y_ctr += 1
        cv2.putText(disp, '%s' % s, (10, disp_step_y * disp_step_y_ctr), font, 0.8, (255, 255, 255), 1, cv2.CV_AA)
    logger.debug('fr:%s, min: %s, max: %s' % (i, disp.min(), disp.max()))
    # Display the resulting frame
    cv2.imshow('frame', disp)

    fr_wait = int(1000.0 / fr)
    if cv2.waitKey(fr_wait) & 0xFF == ord('q'):
        break
  cv2.destroyAllWindows()


def imshow(flow, mask=True, viz_type='hsv', mx_scale=1, display=True):
  """
    Show the flow frame, using vector visualization.
  """
  if viz_type == 'hsv':
    disp = draw_hsv(flow.copy(), mx_scale=mx_scale)
  elif viz_type == 'flow':
    disp = draw_flow(flow.copy())
  else:
    raise NotImplementedError('viz_type %s is not supported' % viz_type)
  # plt.imshow(disp)
  # plt.show()
  if display:
    cv2.imshow('frame', disp)
    cv2.waitKey(1000)
    # if cv2.waitKey(5000) & 0xFF == ord('q'):
        # break
    cv2.destroyAllWindows()
  return disp


def draw_flow(flow, img=None, step=16):
  """ Visualize the flow using vector lines.

    Args:
      flow (np.array): Optical flow to visualize.
      img (optional(np.array)): Background image to overlay the flow on.
      If None, will use a black frame. Defaults to None.
      step (optional(int)): Flow vectors will only be drawn at every "step"
      pixels. This is just subsampling the flow for cleaner visualization.

    Returns:
      vis (np.array): An image of the flow, visualized using vectors.
  """
  if img is None:
    # img = np.zeros((flow.shape[0], flow.shape[1], 3))
    img = np.zeros((flow.shape[0], flow.shape[1]), dtype=np.uint8)
  h, w = img.shape[:2]
  y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)
  fx, fy = flow[y, x].T
  lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
  lines = np.int32(lines + 0.5)
  vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  cv2.polylines(vis, lines, 0, (0, 255, 0))
  for (x1, y1), (x2, y2) in lines:
      cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
  return vis


def draw_hsv(flow, gamma=None, mx_scale=1):
  """ Visualize the flow using an HSV (Hue, Saturation Value) image. Good for
  dense flow.

    Args:
      flow (np.array): Optical flow to visualize.
      gamma (optional(float)): Value used to gamma encode. If None, will not
      use gamma encoding. Defaults to none. DEPRECATED.
      mx_scale (optional(int)): Value to scale the flow by, used to increase
      contrast.

    Returns:
      vis (np.array): An image of the flow, visualized using HSV. Flow
      direction is mapped to color and flow magnitude is mapped to value.
      Saturation is constant at 255.
  """
  h, w = flow.shape[:2]
  fx, fy = flow[:, :, 0] * mx_scale, flow[:, :, 1] * mx_scale
  # ang = np.arctan2(fy, fx) + np.pi
  # to match http://hci.iwr.uni-heidelberg.de/Static/correspondenceVisualization/
  ang = np.arctan2(fx, fy) + np.pi
  v = np.sqrt(fx * fx + fy * fy)
  hsv = np.zeros((h, w, 3), np.uint8)

  # # map direction: color (hue), magnitude: value
  hsv[..., 0] = ang*(180/np.pi/2)
  hsv[..., 1] = 255
  hsv[..., 2] = np.minimum(v, 255)

  # # map direction: color (hue), magnitude: saturation
  # hsv[..., 0] = ang * (180 / np.pi / 2)
  # hsv[..., 1] = np.minimum(v, 255)
  # hsv[..., 2] = 255

  bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

  if gamma is not None:
    logger = utilfx.get_logger(__name__)
    logger.debug('GAMMA ENCODING, HSV TYPE: %s, max: %s' % (bgr.dtype, bgr.max()))
    bgr = bgr.astype(np.float) / 255.0
    bgr = np.power(bgr, gamma)

  return bgr


def draw_hsv_with_key(flow, mx_scale=1):
  """ Visualize the flow using an HSV (Hue, Saturation Value) image, with a
  HSV key in the top-right corner.

    Args:
      flow (np.array): Optical flow to visualize.
      mx_scale (optional(int)): Value to scale the flow by, used to increase
      contrast.

    Returns:
      vis (np.array): An image of the flow, visualized using HSV. Flow
      direction is mapped to color and flow magnitude is mapped to value.
      Saturation is constant at 255. Contains an HSV key in the top-right corner.
  """
  out_img = draw_hsv(flow, mx_scale=mx_scale)
  hsv_key = draw_hsv(make_hsv_key())
  hsv_key = scipy.misc.imresize(hsv_key, 0.5)
  out_img[:hsv_key.shape[0], -hsv_key.shape[1]:, :] = hsv_key
  return out_img


def draw_warp(flow, img=None):
  """ Visualize the flow using a warped-glitchy visualization. Not very useful,
  consider using draw_hsv() instead.

    Args:
      flow (np.array): Optical flow to visualize.
      img (optional(np.array)): Background image to overlay the flow on.
      If None, will use a black frame. Defaults to None.
      mx_scale (optional(int)): Value to scale the flow by, used to increase
      contrast.

    Returns:
      vis (np.array): An image of the flow, visualized using HSV. Flow
      direction is mapped to color and flow magnitude is mapped to value.
      Saturation is constant at 255. Contains an HSV key in the top-right corner.
  """
  logger = utilfx.get_logger(__name__)
  logger.warn('Consider using draw_hsv() or draw_flow to visualize flow, instead of warp.')
  h, w = flow.shape[:2]
  if img is None:
    img = np.zeros((h, w, 3))
  flow = -flow
  flow[:, :, 0] += np.arange(w)
  flow[:, :, 1] += np.arange(h)[:, np.newaxis]
  res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
  return res
