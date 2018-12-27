from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from vgg19 import *
from utils import *
import scipy.io as io

class style_GAN_(object):
    def __init__(self,
                 sess,
                 epoch,
                 batch_size,
                 checkpoint_dir,
                 result_dir,
                 log_dir,
                 model_dir,
                 dataset_name,
                 vgg_path,
                 style_strength):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.vgg_path = vgg_path
        self.style_strength = style_strength
        self.content_layers = [('conv4_2',1.)]
        self.style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]

        if dataset_name == 'mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.input_channel = 3

            # WGAN-GP parameter
            self.lambd = 0.25
            self.disc_iters = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64

            # get number of batched for a single epoch
            self.num_batches = len(mnist.train.images) // self.batch_size
        else:
            raise NotImplementedError

    def build_vgg19(self, path):
        net = {}
        vgg_rawnet = io.loadmat(path)
        vgg_layers = vgg_rawnet['layers'][0]
        net['input'] = tf.Variable(np.zeros((self.batch_size, self.input_height, self.input_width, self.input_channel)).astype('float32'))
        net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0))
        net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2))
        net['pool1'] = build_net('pool', net['conv1_2'])
        net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5))
        net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7))
        net['pool2'] = build_net('pool', net['conv2_2'])
        net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10))
        net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12))
        net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14))
        net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16))
        net['pool3'] = build_net('pool', net['conv3_4'])
        net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19))
        net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21))
        net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23))
        net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25))
        net['pool4'] = build_net('pool', net['conv4_4'])
        net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28))
        net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30))
        net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32))
        net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34))
        net['pool5'] = build_net('pool', net['conv5_4'])

        return net

    def generator(self, image, training, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            # Les border effects when padding a little before passing through ..
            print("image: ", image)
            image = tf.pad(image, [[0,0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
            print("image: ", image)

            conv1 = relu(bn(conv2d(image, 32, 9, 9, 1, 1, name='g_conv1'),
                          is_training=is_training, scope="g_bn1"))
            conv2 = relu(bn(conv2d(conv1, 64, 3, 3, 2, 2, name='g_conv2'),
                            is_training=is_training, scope='g_bn2'))
            conv3 = relu(bn(conv2d(conv2, 128, 3, 3, 2, 2, name='g_conv3'),
                            is_training=is_training, scope='g_bn3'))
            print("conv1: ", conv1)
            print("conv2: ", conv2)
            print("conv3: ", conv3)

            res1 = residual(conv3, 128, 3, 1, name='g_res1')
            res2 = residual(res1, 128, 3, 1, name='g_res2')
            res3 = residual(res2, 128, 3, 1, name='g_res3')
            res4 = residual(res3, 128, 3, 1, name='g_res4')
            res5 = residual(res4, 128, 3, 1, name='g_res5')
            print("res1: ", res1)
            print("res2: ", res2)
            print("res3: ", res3)
            print("res4: ", res4)
            print("res5: ", res5)

            deconv1 = relu(bn(resize_conv2d(res5, 64, 3, 2, training=training, name='g_deconv1'),
                              is_training=is_training, scope='g_deconv1'))
            deconv2 = relu(bn(resize_conv2d(deconv1, 32, 3, 2, training=training, name='g_deconv2'),
                              is_training=is_training, scope='g_deconv2'))
            deconv3 = relu(bn(resize_conv2d(deconv2, 3, 9, 1, training=training, name='g_deconv3'),
                              is_training=is_training, scope='g_deconv3'))
            print("deconv1: ", deconv1)
            print("deconv2: ", deconv2)
            print("deconv3: ", deconv3)

            y = (deconv3 + 1) * 127.5

            # Remove border effect reducing padding.
            height = tf.shape(y)[1]
            width = tf.shape(y)[2]
            y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height-20, width-20, -1]))
            print("y: ", y)
            print("="*30)

            return y

    def discriminator(self, image, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            print("image: ", image)
            image = tf.pad(image, [[0,0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
            print("image: ", image)

            conv1 = lrelu(bn(conv2d(image, 32, 9, 9, 1, 1, name='d_conv1'),
                          is_training=is_training, scope="d_bn1"))
            conv2 = lrelu(bn(conv2d(conv1, 64, 3, 3, 2, 2, name='d_conv2'),
                            is_training=is_training, scope='d_bn2'))
            conv3 = lrelu(bn(conv2d(conv2, 128, 3, 3, 2, 2, name='d_conv3'),
                            is_training=is_training, scope='d_bn3'))
            print("conv1: ", conv1)
            print("conv2: ", conv2)
            print("conv3: ", conv3)

            # use the residual with lrelu function
            res1 = lrelu_residual(conv3, 128, 3, 1, name='d_res1')
            res2 = lrelu_residual(res1, 128, 3, 1, name='d_res2')
            res3 = lrelu_residual(res2, 128, 3, 1, name='d_res3')
            res4 = lrelu_residual(res3, 128, 3, 1, name='d_res4')
            res5 = lrelu_residual(res4, 128, 3, 1, name='d_res5')
            print("res1: ", res1)
            print("res2: ", res2)
            print("res3: ", res3)
            print("res4: ", res4)
            print("res5: ", res5)
            print("="*50)

            features = tf.reshape(res5, [self.batch_size, -1])
            fc1 = lrelu(bn(linear(features, 1024, scope="d_fc1"),
                           is_training=is_training, scope='d_bn4'))

            out_logit = linear(fc1, 1, scope='d_fc2')

            return out_logit

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.input_channel]
        bs = self.batch_size

        """ Graph Input """
        # To let the computation be easy, we give the same batch_size style image
        self.raw_image = tf.placeholder(tf.float32, [bs] + image_dims, name='raw_images')
        self.sty_image = tf.placeholder(tf.float32, [bs] + image_dims, name='sty_image')

        """ G and D's value get """
        G = self.generator(self.raw_image, training=True, is_training=True, reuse=False)
        print("G: ", G)
        D_real_logits = self.discriminator(self.raw_image, is_training=True, reuse=False)
        D_fake_logits = self.discriminator(G, is_training=True, reuse=True)

        """ Vgg19 value get"""
        vgg19_1 = self.build_vgg19(self.vgg_path)
        vgg19_2 = self.build_vgg19(self.vgg_path)

        """ Loss Function """
        # content loss
        vgg19_1['input'].assign(G)
        vgg19_2['input'].assign(self.raw_image)
        self.cost_content = sum(map(lambda l: l[1]*build_content_loss(vgg19_1[l[0]],
                                                                  vgg19_2[l[0]]), self.content_layers))
        # style loss
        # it seems that the shape will give some bug
        vgg19_1['input'].assign(G)
        vgg19_2['input'].assign(self.raw_image)
        self.cost_style = sum(map(lambda l: l[1]*build_style_loss(vgg19_1[l[0]],
                                                              vgg19_2[l[0]]), self.style_layers))

        self.cost_total = self.cost_content + self.style_strength * self.cost_style

        # discriminator loss
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)
        self.d_loss = d_loss_real + d_loss_fake

        # generator loss
        self.g_loss = - d_loss_fake

        """ Gradient Penalty """
        alpha = tf.random_uniform(shape=self.raw_image.get_shape(), minval=0., maxval=1.)
        differences = G - self.raw_image
        interpolates = self.raw_image + (alpha * differences)
        D_inter = self.discriminator(interpolates, is_training=True, reuse=True)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
        gradients_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.d_loss += self.lambd * gradients_penalty

        """ Training """
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate,
                                                  beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate,
                                                  beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)
            # For style optim, we also just renew the generator parameters
            # self.s_optim = tf.train.AdamOptimizer(self.learning_rate,
            #                                       beta1=self.beta1).minimize(self.cost_total, var_list=g_vars)

        """ Testing """
        # for test
        self.fake_images = self.generator(self.raw_image, training=False, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.cost_content_ = tf.summary.scalar("cost_content", self.cost_content)
        self.cost_style_ = tf.summary.scalar("cost_style", self.cost_style)
        self.cost_total_ = tf.summary.scalar("cost_total_", self.cost_total)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.s_sum = tf.summary.merge([self.cost_content_, self.cost_style_, self.cost_total_])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # save to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' +
                                            self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            counter = checkpoint_counter
            print( "[*] Load SUCCESS")
        else:
            start_epoch = 0
            counter = 1
            print(" [!] Load failed...")

        # get the style image
        def getstyle():
            pass

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            for idx in range(0, self.num_batches):
                batch_images, _ = mnist.train.next_batch(self.batch_size)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.raw_image: batch_images})
                self.writer.add_summary(summary_str, counter)

                # update style network
                _, summary_str, content_loss, style_loss, total_loss = self.sess.run([self.s_optim, self.s_sum, self.cost_content, self.cost_style_, self.cost_total_],
                                                                                     feed_dict={self.raw_image: batch_images, self.sty_image: batch_images})
                self.writer.add_summary(summary_str, counter)

                # update G network
                if (counter-1) % self.disc_iters == 0:
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                           feed_dict={self.raw_image: batch_images})
                    self.writer.add_summary(summary_str, counter)

                counter += 1

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss:%.8f, content_loss: %.8f, style_loss: %.8f, total_loss: %.8f"
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, content_loss, style_loss, total_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.raw_image: batch_images})
                    tot_num_samples = min(self.sample_run, self.batch_size)
                    # floor(-2.5) == -3
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)


    def load(self, checkpoint_dir):
        import re

        print("[*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        # get all the trained model's name and the newest model
        # This can be test when training, writing to the blog
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # get the final name before / of url
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """
        samplex, _ = mnist.train.next_batch(self.batch_size)
        samples = self.sess.run(self.fake_images, feed_dict={self.raw_image: samplex})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + '_epoch%03d' % epoch + '_test_all_classes.png')