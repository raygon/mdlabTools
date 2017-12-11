import cv2


class CVVideo(object):
  """ A class that provides a context handler for OpenCV VideoWriter objects.
    This allows OpenCV Writers to be handled using Python's 'with' statement.
  """
  def __init__(self, filename, fourcc, fps, frame_size, is_color=True):
    self.filename = filename
    self.fourcc = fourcc
    self.fps = fps
    self.frame_size = frame_size
    self.is_color = is_color

  def __enter__(self):
    self._writer = cv2.VideoWriter(self.filename, self.fourcc, self.fps, self.frame_size)
    return self

  def __exit__(self, type, value, traceback):
    self._writer.release()
    self._writer = None

  def write(self, frame):
    self._writer.write(frame)
