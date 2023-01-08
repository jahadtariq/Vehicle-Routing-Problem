from __future__ import print_function

import json
import math
import os
import sys
import time
from datetime import datetime
import numpy as np 

import tensorflow as tf  
import scipy.misc

try:
    from StringIO import StringIO 
except ImportError:
    from io import BytesIO

print_grad = True

class printOut(object):
  def __init__(self,f=None ,stdout_print=True):
    self.out_file = f
    self.stdout_print = stdout_print

  def print_out(self, s, new_line=True):
    if isinstance(s, bytes):
      s = s.decode("utf-8")

    if self.out_file:
      self.out_file.write(s)
      if new_line:
        self.out_file.write("\n")
    self.out_file.flush()

    if self.stdout_print:
      print(s, end="", file=sys.stdout)
      if new_line:
        sys.stdout.write("\n")
      sys.stdout.flush()

  def print_time(self,s, start_time):
    self.print_out("%s, time %ds, %s." % (s, (time.time() - start_time) +"  " +str(time.ctime()) ))
    return time.time()

  def print_grad(self,model, last=False):
    if print_grad:
      for tag, value in model.named_parameters():
        if value.grad is not None:
          self.print_out('{0: <50}'.format(tag)+ "\t-- value:" \
            +'%.12f' % value.norm().data[0]+ "\t -- grad: "+ str(value.grad.norm().data[0]))
        else:
          self.print_out('{0: <50}'.format(tag)+ "\t-- value:" +\
            '%.12f' % value.norm().data[0])
      self.print_out("-----------------------------------")
      if last:
        self.print_out("-----------------------------------")
        self.print_out("-----------------------------------")

def get_time():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def to_np(x):
  return x.data.cpu().numpy()

def to_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)   

def extract(xVar):
  global yGrad
  yGrad = xVar
  print(yGrad)

def extract_norm(xVar):
  global yGrad
  yGradNorm = xVar.norm() 
  print(yGradNorm)

class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def _single_cell(unit_type, num_units, forget_bias, dropout, prt,
                                 residual_connection=False, device_str=None):

    if unit_type == "lstm":
        prt.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units,
                forget_bias=forget_bias)
    elif unit_type == "gru":
        prt.print_out("  GRU", new_line=False)
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))
        prt.print_out("  %s, dropout=%g " %(type(single_cell).__name__, dropout),
                                        new_line=False)

    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
        prt.print_out("  %s" % type(single_cell).__name__, new_line=False)

    return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
                             forget_bias, dropout, mode, prt, num_gpus, base_gpu=0):
    cell_list = []
    for i in range(num_layers):
        prt.print_out("  cell %d" % i, new_line=False)
        dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        single_cell = _single_cell(
                unit_type=unit_type,
                num_units=num_units,
                forget_bias=forget_bias,
                dropout=dropout,
                prt=prt,
                residual_connection=(i >= num_layers - num_residual_layers),
                device_str=get_device_str(i + base_gpu, num_gpus),
        )
        prt.print_out("")
        cell_list.append(single_cell)

    return cell_list

