from re import search as re_search
import subprocess
from mdlab.utilfx import printiv, make_iterable
# import pdb
# class SlurmJob(object):
#   def __init__(self, time=None, memory=None, n_cores=None, n_nodes=None, n_gpus=None, gpu_type=None):


def sort_and_freeze_kwargs(kwargs_sort_keys, **kwargs):
  return SlurmJob.sort_and_freeze_kwargs(kwargs_sort_keys, **kwargs)


def args_to_fire_cli(*args):
  return SlurmJob.args_to_fire_cli(*args)


def kwargs_to_fire_cli(**kwargs):
  return SlurmJob.kwargs_to_fire_cli(**kwargs)


def wrap_fire(src_file, fx):
  return SlurmJob.wrap_fire(src_file, fx)


class SlurmJob(object):
  def __init__(self, base_resource_str, qos_group='mcdermott', job_name='SlurmJob',
          output='slurm-%j.out', mail_type='ALL', mail_user='raygon@mit.edu'):
    self.base_resource_str = base_resource_str
    self.qos_group = qos_group
    self.job_name = job_name
    self.output = output
    self.mail_type = mail_type
    self.mail_user = mail_user

  def get_resource_str(self):
    return '%s --job-name=%s --output=%s --mail-type=%s --mail-user=%s' % (self.base_resource_str,
            self.job_name, self.output, self.mail_type, self.mail_user)

  @staticmethod
  def sort_and_freeze_kwargs(kwargs_sort_keys, **kwargs):
    if kwargs_sort_keys is None:
      frozen_kwargs = tuple(kwargs)
    else:
      frozen_kwargs = [(k, kwargs[k]) for k in kwargs_sort_keys if k in kwargs]
      frozen_kwargs.extend([(k, kwargs[k]) for k in kwargs if k not in kwargs_sort_keys])
      frozen_kwargs = tuple(frozen_kwargs)
    return frozen_kwargs

  @staticmethod
  def args_to_fire_cli(*args):
    try:
      dict(args)
      # pdb.set_trace()
      k, v = zip(*args)
      k_cli = ['--%s' % x for x in k]
      args_str = ' '.join(['%s %s' % (xk, xv) for xk, xv in zip(k_cli, v)])
    except (ValueError, TypeError):
      args_str = ' '.join(['%s' % x for x in args])
    args_str = args_str.strip()
    return args_str

  @staticmethod
  def kwargs_to_fire_cli(**kwargs):
    frozen_kwargs_keys = kwargs.keys()
    kwargs_str = ['--%s=%s' % (k, kwargs[k]) for k in frozen_kwargs_keys]
    kwargs_str = ' '.join(kwargs_str)
    kwargs_str = kwargs_str.strip()
    return kwargs_str

  @staticmethod
  def wrap_fire(src_file, fx):
    def _fx(*args, **kwargs):
      # args_str = ' '.join(['%s' % x for x in args])
      # pdb.set_trace()
      args_str = args_to_fire_cli(*args)
      kwargs_str = kwargs_to_fire_cli(**kwargs)
      # out_str = 'python %s %s %s %s' % (src_file, fx.__name__, args_str, kwargs_str)
      # out_str = 'python %s wrap_context %s %s %s' % (src_file, fx.__name__, args_str, kwargs_str)
      out_str = 'python %s %s %s %s' % (src_file, fx.__name__, args_str, kwargs_str)
      out_str = out_str.strip()
      # pdb.set_trace()
      return out_str
    return _fx

  # def submit_job_core(self, fire_cli_str, use_qos=False, dry=False):
  #   qos_str = ''
  #   if use_qos:
  #     qos_str = '--qos %s' % self.qos_group

  #   slurm_str = 'sbatch %s %s %s' % (self.resource_str, qos_str, fire_cli_str)

  #   if dry:
  #     print('DRY ---> %s' % slurm_str)
  #   else:
  #     subprocess.run(slurm_str, shell=True)

  def submit_job(self, src_file, fx, use_qos=False, dependency=None,  array=None, dry=False, verbose=1):
    if dependency is None:
      dependency_str = ''
    elif isinstance(dependency, int) or isinstance(dependency, list) or isinstance(dependency, tuple):
      dependency_str = '--dependency=afterok:' + ':'.join(['%s' % x for x in make_iterable(dependency)])
    else:
      dependency_str = dependency
    dependency_str = dependency_str.strip()

    if array:  #TODO fix this hack
      old_output = self.output
      self.output = 'slurm_%A_%a.out'

    if array is None:
      array_str = ''
    elif isinstance(array, range):
      array_str = '--array=%s-%s' % (min(array), max(array))
    elif isinstance(array, int) or isinstance(array, list) or isinstance(array, tuple):
      array_str = '--array=' + ','.join(['%s' % x for x in make_iterable(array)])
    else:
      array_str = array
    array_str =  array_str.strip()


    def _fx(*args, **kwargs):
      qos_str = ''
      if use_qos:
        qos_str = '--qos %s' % self.qos_group
      qos_str = qos_str.strip()

      # pdb.set_trace()
      fire_cli_str = wrap_fire(src_file, fx)(*args, **kwargs)
      slurm_str = 'sbatch %s %s %s %s --wrap="echo %s; %s"' % (array_str, dependency_str, qos_str, self.get_resource_str(), fire_cli_str, fire_cli_str)
      # fire_cli_str = 'python dfGenerateStimuliSubsetManager.py square \$SLURM_ARRAY_TASK_ID'
      # slurm_str = 'sbatch %s %s %s %s --wrap="echo %s; %s"' % (array_str, dependency_str, qos_str, self.get_resource_str(), fire_cli_str, fire_cli_str)
      slurm_str = slurm_str.strip()

      if array:  #TODO fix this hack
        old_output = self.output
        self.output = 'slurm_%A_%a.out'

      if dry:
        printiv('DRY ---> %s' % slurm_str, verbose, 2)
        job_id = 'dry'
      else:
        job_id = subprocess.run(slurm_str, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        match = re_search('\d+', job_id)
        if match:
          job_id = int(match.group(0))
      out_args = (job_id, slurm_str)
      printiv(out_args, verbose)
      return out_args
    return _fx


def test_function2(arg1, arg2, arg3=None, arg4=True):
  return 10


def test_function(arg1, arg3=None, arg4=True):
  return 10


def slurm_test_fx(x, power=2):
  for i in range(x):
    print(i**power)
  return 'done'


def submit_slurm_test():
  sj = SlurmJob('--mem=200 -t 01:00')
  (job_id, _) = sj.submit_job(__file__, slurm_test_fx)(20)
  sj.submit_job(__file__, slurm_test_fx, dependency=job_id)(10, 3)


def test_SlurmJob():
  # s = wrap_fire(test_function)('one', 'two')
  # print(s)
  # s = wrap_fire(test_function)('one', 'two', arg4=False)
  # print(s)

  sj = SlurmJob('--mem=2000 -t 0-2')
  s = sj.submit_job(__file__, test_function, use_qos=True, dependency=12345, dry=True)('one', 'two')
  s = sj.submit_job(__file__, test_function, dry=True)(1, 2, arg3=666)
  s = sj.submit_job(__file__, test_function, dry=True)('oneXXX')
  s = sj.submit_job(__file__, test_function, use_qos=True, dependency=[12345,789,9999], dry=True)('one', 'two')



if __name__ == '__main__':
  test_SlurmJob()
