from __future__ import division
from __future__ import unicode_literals


import numpy as np

def undo_transforms(y, transformers):
  for transformer in reversed(transformers):
    if transformer.transform_y:
      y = transformer.untransform(y)
  return y

class BalancingTransformer:
  def __init__(self,
               transform_X=False,
               transform_y=False,
               transform_w=False,
               dataset=None,
               ):
    self.dataset = dataset
    self.transform_X = transform_X
    self.transform_y = transform_y
    self.transform_w = transform_w
    assert transform_X or transform_y or transform_w
    assert (transform_X + transform_y + transform_w) == 1
    assert not transform_X
    assert not transform_y
    assert transform_w

    y = self.dataset.y
    w = self.dataset.w
    np.testing.assert_allclose(sorted(np.unique(y)), np.array([0., 1.]))
    weights = []
    for ind, task in enumerate(self.dataset.get_task_names()):
      task_w = w[:, ind]
      task_y = y[:, ind]
      task_y = task_y[task_w != 0]
      num_positives = np.count_nonzero(task_y)
      num_negatives = len(task_y) - num_positives
      if num_positives > 0:
        pos_weight = float(num_negatives) / num_positives
      else:
        pos_weight = 1
      neg_weight = 1
      weights.append((neg_weight, pos_weight))
    self.weights = weights

  def transform(self, dataset):
    return dataset.transform(lambda X, y, w: self.transform_array(X, y, w))

  def transform_array(self, X, y, w):
    w_balanced = np.zeros_like(w)
    for ind, task in enumerate(self.dataset.get_task_names()):
      task_y = y[:, ind]
      task_w = w[:, ind]
      zero_indices = np.logical_and(task_y == 0, task_w != 0)
      one_indices = np.logical_and(task_y == 1, task_w != 0)
      w_balanced[zero_indices, ind] = self.weights[ind][0]
      w_balanced[one_indices, ind] = self.weights[ind][1]
    return (X, y, w_balanced)
