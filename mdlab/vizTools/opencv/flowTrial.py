import cv2
import ioTools
import utilfx
import h5py
# from hdfTools import walk_hdf


class FlowTrial(object):
  """
  Representation of a optical flow output from a single trial from the goalie task.
  """
  def __init__(self, vid, frameLaunchEnd, lor, file_id=None):
    self.frameLaunchEnd = frameLaunchEnd
    self.lor = lor
    self.vid = vid
    self.file_id = file_id

  def __str__(self):
    s = '----- FlowTrial Object --------------\n'
    s += 'Vid Shape       : %s\n' % self.vid.shape.__str__()
    s += 'frameLaunchEnd  : %s\n' % self.frameLaunchEnd
    s += 'Left/Right (lor): %s\n' % self.lor
    s += 'File ID         : %s\n' % self.file_id
    s += '-------------------------------------'
    return s

  @staticmethod
  def write_flow(wfn, vid, frameLaunchEnd, lor, file_id=None, read_mode='w'):
    out_data = {
      'StartToHitMx': vid,
      'frameLaunchEnd': frameLaunchEnd,
      'lor': lor,
      'file_id': file_id
    }
    if wfn.endswith('.mat'):
      ioTools.write_to_mat(out_data, wfn, read_mode=read_mode)
    elif wfn.endswith('.hdf5'):
      out_data['lor'] = FlowTrial.lor_to_bin(lor)
      ioTools.write_to_hdf(out_data, wfn, read_mode=read_mode)
    else:
      logger = utilfx.get_logger(__name__)
      logger.error('Unsupported outpt filetype "%s"' % wfn, exc_info=True)
      raise NotImplementedError('Unsupported outpt filetype "%s"' % wfn)

  @staticmethod
  def lor_to_bin(lor_str):
    label = lor_str
    if 'right' == label.lower():
      label = 1
    elif 'left' == label.lower():
      label = 0
    else:
      logger = utilfx.get_logger(__name__)
      logger('FlowTrial lor is not "left" or "right".', exc_info=True)
      raise ValueError('FlowTrial lor is not "left" or "right".')
    return label

  @classmethod
  def from_file(cls, filename):
    if filename.endswith('.mat'):
      # data = ioTools.loadmat(filename)
      data = ioTools.load_from_mat(filename)
      obj = cls(data['StartToHitMx'], data['frameLaunchEnd'], data['lor'], filename)
    elif filename.endswith('.hdf5'):
      raise NotImplementedError()
      with h5py.File(filename, 'r') as hdf_obj:
        gn = walk_hdf(hdf_obj)
        if len(gn) != 1:
          raise ValueError('FlowTrial cannot load HDF file')
        gn = gn[0]
        obj = cls(hdf_obj[gn]['vid'].value, hdf_obj[gn]['frameLaunchEnd'].value, hdf_obj[gn]['lor'].value, filename)
    else:
      raise NotImplementedError('FlowTrial does not support loading from filetype %s' % filename.split('.')[-1])
    return obj

  def lor_as_bin(self):
    return FlowTrial.lor_to_bin(self.lor[0])

  def play_vid(self):
    for i in xrange(self.vid.shape[-1]):
      frame = self.vid[:, :, :, i]
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    cv2.destroyAllWindows()

  def write(self, wfn, read_mode='w'):
    FlowTrial.write_flow(wfn, self.vid, self.frameLaunchEnd, self.lor, self.file_id, read_mode=read_mode)
