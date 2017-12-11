import sparseTools
import cv2
import ioTools
import utilfx


class GoalieTrial(object):
  """ Representation of a single trial from the goalie task.

    Args:
      vid (np.array): Array of RGB video data, usually in 'StartToHitMx'.
      frameLaunchEnd (np.array): An array containing the MATLAB index (1-indexed)
      of the motion event and the index of the last frame of the video.
      lor (np.array): An array containing the 'left' or 'right' label of the
      trial.
      file_id (optional(str)): A description used to identify the trial.

  """
  def __init__(self, vid, frameLaunchEnd, lor, file_id=None):
    self.frameLaunchEnd = frameLaunchEnd
    self.lor = lor
    self.vid = vid
    self.file_id = file_id

  @classmethod
  def from_file(cls, filename):
    data = ioTools.load_from_mat(filename)
    obj = cls(data['StartToHitMx'], data['frameLaunchEnd'], data['lor'], filename)
    return obj

  @staticmethod
  def write_trial(wfn, vid, frameLaunchEnd, lor, file_id=None, read_mode='w'):
    if wfn.endswith('.mat'):
      out_data = {
        'StartToHitMx': vid,
        'frameLaunchEnd': frameLaunchEnd,
        'lor': lor,
        'file_id': file_id
      }
      ioTools.write_to_mat(out_data, wfn, read_mode=read_mode)
    elif wfn.endswith('.hdf5'):
      logger = utilfx.get_logger(__name__)
      logger.error("GoalieTrial writer doesn't support HDF", exc_info=True)
      raise NotImplementedError("GoalieTrial writer doesn't support HDF")
    else:
      logger = utilfx.get_logger(__name__)
      logger.error('Unsupported outpt filetype "%s"' % wfn)
      raise NotImplementedError('Unsupported outpt filetype "%s"' % wfn)

  def play_vid(self):
    for i in xrange(self.vid.shape[-1]):
      frame = self.vid[:, :, :, i]
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    cv2.destroyAllWindows()

  def write(self, wfn, read_mode='w'):
    GoalieTrial.write_trial(self.vid, self.frameLaunchEnd, self.lor, self.file_id, read_mode=read_mode)

  def write_sparse_flow(self, flow_img, write_filename):
    flow_img = sparseTools.flatten_vid_array(flow_img)
    flow_img = sparseTools.to_sparse(flow_img)
    out_data = {'StartToHitMx': flow_img, 'frameLaunchEnd': self.frameLaunchEnd, 'lor': self.lor}
    ioTools.write_to_mat(out_data, write_filename, force=True)
