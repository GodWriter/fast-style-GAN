import tensorflow as tf

# tf.pad()
# t = tf.constant([[1, 2], [3, 4]])
# paddings = tf.constant([[2, 1], [1, 1]])
#
# a = tf.pad(t, paddings, "CONSTANT")


# sess = tf.Session()
# print(sess.run(a))

# t = tf.constant([[[1, 2, 3], [4, 5, 6]],
#                  [[1, 2, 3], [4, 5, 6]],
#                  [[1, 2, 3], [4, 5, 6]]])

# tf.split()
# t = tf.random_normal(shape=[10, 4, 5])
# split0, split1, split2 = tf.split(t, [2, 3, 5], 0)
#
# sess = tf.Session()
# shape0, shape1, shape2 = sess.run([tf.shape(split0), tf.shape(split1), tf.shape(split2)])
#
# print("shape0: ", shape0)
# print("shape1: ", shape1)
# print("shape2: ", shape2)

# tensorflow广播测试
# a = tf.constant([[[1, 1, 1], [2, 2, 2]],
# #                  [[3, 3, 3], [4, 4, 4]],
# #                  [[5, 5, 5], [6, 6, 6]]])
# # b = tf.constant([[[1, 1, 1], [1, 1, 1]]])
# # c = a - b
# #
# # sess = tf.Session()
# # a_shape, b_shape, c_shape = sess.run([tf.shape(a), tf.shape(b), tf.shape(c)])
# #
# # print(sess.run(c))
# # print("a.shape: ", a_shape)
# # print("b.shape: ", b_shape)
# # print("c.shape: ", c_shape)
import scipy.io as io

vgg_rawnet = io.loadmat('../vgg-model/imagenet-vgg-verydeep-19.mat')
vgg_layers = vgg_rawnet.keys()
print(vgg_layers)