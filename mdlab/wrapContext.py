import inspect
import os
import pickle

import pdb
import pprint
ppr = pprint.PrettyPrinter()


class WrapContext(object):
  def __init__(self):
    self.is_registered = False

  @staticmethod
  def extract_stack_data(mode='slim'):
    mode = mode.lower()
    stack = inspect.stack()
    # valid_stack_data = [inspect.getargvalues(s.frame) for (i, s) in enumerate(stack) if s.function == 'wrapped_context_fx']
    # valid_stack_data = [inspect.getargvalues(s.frame) for (i, s) in enumerate(stack)]
    # pdb.set_trace()
    # valid_stack_data = [('Stack Frame: %s' % (i + min_ind), s.function, inspect.getargvalues(s.frame)) for (i, s) in enumerate(stack[1:min_ind])]
    valid_stack_data = []

    min_ind = 0
    if mode is None or mode == 'standard' or mode == 'slim':
      min_ind = min([i for (i, s) in enumerate(stack) if s.function == 'wrapped_context_fx'])
      frames_to_process = enumerate(stack[1:min_ind])
    elif mode == 'all':
      frames_to_process = enumerate(stack)
    else:
      ValueError('Unrecognized mode: %s' % mode)

    frame_0_args = inspect.getargvalues(stack[-1].frame)
    for (i, s) in frames_to_process:
      'Stack Frame: %s' % (i + min_ind)
      s_frame = s.frame
      s_arg_info = inspect.getargvalues(s_frame)
      s_args = s_arg_info.args
      s_varargs = s_arg_info.varargs
      s_keywords = s_arg_info.keywords
      s_locals = s_arg_info.locals
      s_fx_name = s.function
      if i == 0 and s_fx_name == 'wrapped_write_fx':
        s_fx_name = '_' + s_fx_name
      try:
        s_fx = s_locals[s_fx_name] if s_fx_name in s_locals else frame_0_args.locals[s_fx_name]
      except:
        s_fx = None
      s_signature = inspect.signature(s_fx) if s_fx is not None else None
      # pdb.set_trace()

      s_info = {'frame': i + min_ind}
      s_info['fx'] = {'name': s_fx_name, 'callable': s_fx, 'signature': s_signature}
      s_info['positional'] = dict([(a, s_locals[a]) for a in s_args])  # positional args
      s_info['varargs'] = {s_varargs: s_locals[s_varargs]} if s_varargs is not None else None
      s_info['keywords'] = {s_keywords: s_locals[s_keywords]} if s_keywords is not None else None
      valid_stack_data.append(s_info)

      # ppr.pprint(s_arg_info)
      # ppr.pprint(s_info)
      # print()
      # pdb.set_trace()
    # pdb.set_trace()
    return valid_stack_data

  @staticmethod
  def wrapped_write(wrapped_write_fx):
    def _wrapped_write_fx(wfn, *args, **kwargs):
      stack_wfn = os.path.join(os.path.dirname(wfn), 'runInfoStacks')
      stack_data = WrapContext.extract_stack_data()
      # pdb.set_trace()
      try:
        with open(stack_wfn + '.pkl', 'wb') as wf:
          pickle.dump(stack_data, wf)
      except Exception:
        os.remove(stack_wfn + '.pkl')  # cleanup failed attempt at pickling
        with open(stack_wfn + '.txt', 'w') as wf:
          wf.write(pprint.pformat(stack_data))
      # with open(stack_wfn, 'w') as wf:
        # wf.write(str(inspect.stack()))
      # pdb.set_trace()
      return wrapped_write_fx(wfn, *args, **kwargs)
    return _wrapped_write_fx

  def register(self, **kwargs):
    # self.registered_args = args
    self.registered_kwargs = kwargs
    self.registered_kwargs['wrap_context'] = self.wrap_context

  def wrap_context(self, fx):
    if not self.is_registered:
      AttributeError('WrapContext instance has not been registered. Call `WrapContext.register` before using `wrap_context`')
    # DANGER HACK holy shit this is a hack -- get the function object if passed
    # in from the command line as a string, DANGER, who knows what this does
    if not callable(fx):
      print(self.registered_kwargs)
      fx = next(x for x in self.registered_kwargs.values() if x.__name__ == fx)
      # pdb.set_trace()
    def wrapped_context_fx(*args, **kwargs):
      print('Fired wrapped context')
      return fx(*args, **kwargs)
    return wrapped_context_fx
