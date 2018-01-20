from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

def mixing_layer(name, glimpse, refresh, decay_ratio=0.1):
  """
    mixing layer, mix new features with older memories
    Args:
      name: a string, memory variable is named as name+'_memory'
      glimpse: a tensor, new feature
      refresh: a tf.bool variable, if True the memory restarts as zero
      decay_ratio: a scalar indicating the decay rate of memory

    Returns:
      belief: a tensor, mixed input tensor
  """
  initializer = tf.constant_initializer([0.])
  shape = glimpse.get_shape().as_list()
  memory = tf.get_variable(name=name+'_memory',shape=shape, initializer=initializer, trainable=False)
  # decay memory or refresh memory when new stream starts
  memory = tf.assign(memory,
      tf.cond(refresh, lambda: tf.zeros_like(glimpse,tf.float32), 
                        lambda: memory*tf.cast(decay_ratio, tf.float32)),
      validate_shape=False)
  # update memory
  memory_update = tf.assign(memory, glimpse+memory, validate_shape=False)
  # update belief
  belief = memory_update
  belief = tf.reshape(belief, shape)

  return belief
