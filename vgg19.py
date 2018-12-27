import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os


IMAGE_W = 800
IMAGE_H = 600
CONTENT_IMG =  './images/Taipei101.jpg'
STYLE_IMG = './images/StarryNight.jpg'
OUTOUT_DIR = './results'
OUTPUT_IMG = 'results.png'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
INI_NOISE_RATIO = 0.7
STYLE_STRENGTH = 500
ITERATION = 5000

CONTENT_LAYERS =[('conv4_2',1.)]
STYLE_LAYERS=[('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]


MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,1,3))

def build_net(ntype, nin, nwb=None):
  if ntype == 'conv':
    return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME')+ nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                  strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i,):
  weights = vgg_layers[i][0][0][0][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[i][0][0][0][0][1]

  # print("bias: ", bias)
  # print("bias.type: ", type(bias))
  # print("="*50)

  bias = tf.constant(np.reshape(bias, (bias.size)))
  return weights, bias

def build_content_loss(p, x):
  print("p.shape: ", p)
  print("x.shape: ", x)
  print("="*50)

  M = p.shape[1].value*p.shape[2].value
  N = p.shape[3].value

  loss = (1./(2* (N**0.5) * (M**0.5))) * tf.reduce_sum(tf.pow((x - p),2))
  return loss


def gram_matrix(x, bs, area, depth):
  x1 = tf.reshape(x,(bs, area,depth))
  g = tf.matmul(tf.transpose(x1, perm=[0,2,1]), x1)
  return g

def gram_matrix_val(x, bs, area, depth):
  x1 = tf.reshape(x, (bs, area, depth))
  # g = np.dot(x1.T, x1)
  g = tf.matmul(tf.transpose(x1, perm=[0,2,1]), x1)
  return g

def build_style_loss(a, x):
  bs = a.shape[0].value
  M = a.shape[1].value*a.shape[2].value
  N = a.shape[3].value

  A = gram_matrix_val(a, bs, M, N )
  G = gram_matrix(x, bs, M, N )
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
  return loss

def read_image(path):
  image = scipy.misc.imread(path)
  image = scipy.misc.imresize(image,(IMAGE_H,IMAGE_W))
  image = image[np.newaxis,:,:,:]
  image = image - MEAN_VALUES
  return image

def write_image(path, image):
  image = image + MEAN_VALUES
  image = image[0]
  image = np.clip(image, 0, 255).astype('uint8')
  scipy.misc.imsave(path, image)