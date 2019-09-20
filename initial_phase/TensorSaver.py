import tensorpack as tp
import os
import numpy as np
import tensorflow as tf


class TensorSaver(tp.Callback):
  def __init__(self, dir, steps):
    """
    Args:
        names(list): list of string, the names of the tensors to print.
    """
    self._dir = dir
    self._steps = steps if isinstance(steps, list) else list(range(steps))

    self._vars = []
    self._updates = []
    self._uptovar = {}

  def _setup_graph(self):

    self._vars = tf.trainable_variables(scope=None)
    self._updates = []
    updates = [x for x in tf.get_default_graph().get_operations() if 'train_op/update_' in x.name]
    for v in self._vars:
      vn = v.name.split(':')[0]
      vup = [x for x in updates if vn in x.name]
      assert len(vup) == 1
      vup = vup[0]
      self._updates.append(vup)
      self._uptovar[vup] = vn
      self._uptovar[v.op] = vn
    self._fetches = [x.op for x in self._vars]+self._updates


  def _before_run(self, _):
    return self._fetches

  def _after_run(self, _, vals):
    if self.local_step not in self._steps: return

    args = vals.results
    for op, v in zip(self._fetches, args):
      vname = self._uptovar[op].replace('/', '_')
      fname = '%04d__%s__%s' % (self.local_step, vname, 'g' if 'train_op/update_' in op.name else 'v')
      path = os.path.join(self._dir, fname)
      np.save(path, v)