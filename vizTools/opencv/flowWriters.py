import cv2
import ioTools
import cvVideo
import vizTools


class BasicWriter(object):
  """docstring for BasicWriter"""
  def __init__(self):
    self._writer = None
    self._write_fx = None
    self.wfn_stem = None
    self._isWriterOpen = False

  def __enter__(self):
    self._writer.__enter__()
    self._isWriterOpen = True

  def __exit__(self, type, value, traceback):
    self._writer.__exit__(type, value, traceback)
    self._isWriterOpen = False

  @classmethod
  def setup_writer(cls, writer_type, wfn, shape, fps, fourcc=None):
    if writer_type.lower() == 'mplt':
      (writer, write_fx) = ioTools.matplotlib_writer(wfn, shape, fps=fps)
    elif writer_type.lower() == 'cv':
      writer = cvVideo.CVVideo(wfn, cv2.cv.CV_FOURCC(*fourcc), fps, (shape[1], shape[0]), True)
      write_fx = writer.write
    else:
      raise ValueError('argument writer_type must be one of %s' % vizTools.VID_WRITER_TYPES.__str__())
    return (writer, write_fx)

  def set_wfn(self, wfn):
    if self._isWriterOpen:
      raise ValueError('Requested filename change on an open file')
    else:
      self.wfn = wfn
      (self._writer, self._write_fx) = BasicWriter.setup_writer(self.writer_type, self.wfn, self.shape, self.fps, self.fourcc)

  def update_writer(self, frame, overlay_img=None, data=None):
    # process and write frame
    self.write(self.process(frame, overlay_img=overlay_img, data=data))

  def write(self, frame):
    self._write_fx(frame)

  def process(self, frame, overlay_img=None, data=None):
    return frame

  def cvt_frame_color(self, frame, cvt_color):
    if cvt_color is not None:
      return cv2.cvtColor(frame, cvt_color)
    else:
      return frame


class FlowWriter(BasicWriter):
  """ Base class for all writers used with the methods in flow2.py """
  def __init__(self):
    super(FlowWriter, self).__init__()
    # Default parameters for flow writers
    self.writer_type = 'mplt'
    self.fps = 15
    self.mx_scale = 1
    self.cvt_color = None
    self.fourcc = 'mp4v'
    self.wfn_stem = 'FLOWWRITER'


class BasicFlowWriter(FlowWriter):
  """ Write flow using an HSV vizualization. """
  def __init__(self, shape, wfn, writer_type='mplt', fps=15, mx_scale=5, cvt_color=None, fourcc='mp4v'):
    super(BasicFlowWriter, self).__init__()
    self.shape = shape
    self.wfn = wfn
    self.writer_type = writer_type
    self.fps = fps
    self.mx_scale = mx_scale
    self.cvt_color = cvt_color
    self.fourcc = fourcc
    self.wfn_stem = 'flowbasic'
    (self._writer, self._write_fx) = BasicWriter.setup_writer(self.writer_type, self.wfn, self.shape, self.fps, self.fourcc)

  def process(self, frame, overlay_img=None, data=None):
    frame = self.cvt_frame_color(frame, self.cvt_color)
    return frame


class HSVFlowWriter(FlowWriter):
  """ Write flow using an HSV vizualization. """
  def __init__(self, shape, wfn, writer_type='mplt', fps=15, mx_scale=5, cvt_color=None, fourcc='mp4v'):
    super(HSVFlowWriter, self).__init__()
    self.shape = shape
    self.wfn = wfn
    self.writer_type = writer_type
    self.fps = fps
    self.mx_scale = mx_scale
    self.cvt_color = cvt_color
    self.fourcc = fourcc
    self.wfn_stem = 'hsv'
    (self._writer, self._write_fx) = BasicWriter.setup_writer(self.writer_type, self.wfn, self.shape, self.fps, self.fourcc)

  def process(self, frame, overlay_img=None, data=None):
    hsv_frame = vizTools.draw_hsv_with_key(frame, mx_scale=self.mx_scale)
    hsv_frame = self.cvt_frame_color(hsv_frame, self.cvt_color)
    return hsv_frame


class VectorFlowWriter(FlowWriter):
  """ Write flow using a vector vizualization. """
  def __init__(self, shape, wfn, writer_type='mplt', fps=15, mx_scale=1, step=16, cvt_color=None, fourcc='mp4v'):
    super(VectorFlowWriter, self).__init__()
    self.shape = shape
    self.wfn = wfn
    self.writer_type = writer_type
    self.fps = fps
    self.mx_scale = mx_scale
    self.step = step
    self.cvt_color = cvt_color
    self.fourcc = fourcc
    self.wfn_stem = 'vector'
    (self._writer, self._write_fx) = BasicWriter.setup_writer(self.writer_type, self.wfn, self.shape, self.fps, self.fourcc)

  def process(self, frame, overlay_img=None, data=None):
    vx_frame = vizTools.draw_flow(frame, overlay_img, step=self.step)
    self.cvt_frame_color(vx_frame, self.cvt_color)
    return vx_frame
