from __future__ import division

import time
import collections

from ops import *
from vgg19 import *
from utils import *
from data_loader import shipData

class style_GAN_(object):
    def __init__(self,
                 sess,
                 epoch,
                 batch_size,
                 folder_path,
                 style_image_path,
                 checkpoint_dir,
                 result_dir,
                 log_dir,
                 model_dir,
                 dataset_name,
                 net,
                 loss_ratio,
                 content_layer_ids,
                 style_layers_ids):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.folder_path = folder_path
        self.style_image_path = style_image_path
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.net = net
        self.loss_ratio = loss_ratio
        self.CONTENT_LAYERS = collections.OrderedDict(sorted(content_layer_ids.items()))
        self.STYLE_LAYERS = collections.OrderedDict(sorted(style_layers_ids.items()))
        print("CONTENT_LAYERS: ", self.CONTENT_LAYERS)
        print("STYLE_LAYERS: ", self.STYLE_LAYERS)
        print("="*50)

        if dataset_name == 'shipData':
            # parameters
            self.input_height = 32
            self.input_width = 32
            self.input_channel = 3
            self.shape = [self.input_height, self.input_width]

            # WGAN-GP parameter
            self.lambd = 0.25
            self.disc_iters = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 32

            # get number of batched for a single epoch
            self.dataloader = shipData(self.shape, self.folder_path)
            print()
            self.num_batches = int(self.dataloader.data_len / self.batch_size)
        else:
            raise NotImplementedError



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
        self.sty_image = tf.placeholder(tf.float32, [1] + image_dims, name='sty_image')

        """ G and D's value get """
        G = self.generator(self.raw_image, training=True, is_training=True, reuse=False)
        print("G1: ", G)
        D_real_logits = self.discriminator(self.raw_image, is_training=True, reuse=False)
        D_fake_logits = self.discriminator(G, is_training=True, reuse=True)
        print("G2: ", G)

        """ Loss Function """
        # get content-layer-feature for content loss
        content_layers = self.net.feed_forward(self.raw_image, scope='content')
        self.Ps = {}
        for id in self.CONTENT_LAYERS:
            print("content_id: ", id)
            self.Ps[id] = content_layers[id]

        # get style-layer-feature for style loss
        style_layers = self.net.feed_forward(self.sty_image, scope='style')
        self.As = {}
        for id in self.STYLE_LAYERS:
            print("style_id: ", id)
            self.As[id] = self._gram_matrix(style_layers[id])

        # get layer-values for G
        self.Fs = self.net.feed_forward(G, scope='mixed')
        print("G3: ", G)

        """ compute loss """
        L_content = 0
        L_style = 0
        for id in self.Fs:
            if id in self.CONTENT_LAYERS:
                # content loss
                F = self.Fs[id]
                P = self.Ps[id]

                bs, h, w, d = F.get_shape() # first return value is batch_size
                N = h.value*w.value # product of width and height
                M = d.value # number of filters

                w = self.CONTENT_LAYERS[id] # weight for this layer

                # add the bs.value
                L_content += w * (1./(2. * np.sqrt(M) * np.sqrt(N)) * bs.value) * tf.reduce_sum(tf.pow((F - P), 2))
            elif id in self.STYLE_LAYERS:
                # style loss
                F = self.Fs[id]

                bs, h, w, d = F.get_shape()
                N = h.value * w.value
                M = d.value

                w = self.STYLE_LAYERS[id]

                G_ = self._gram_matrix(F)
                A = self.As[id]

                # add the bs.value
                L_style += w * (1./(4 * N**2 * M**2 * bs.value)) * tf.reduce_sum(tf.pow((G_-A), 2))

        alpha = self.loss_ratio
        beta = 1

        self.L_content = L_content
        self.L_style = L_style
        self.L_total = alpha*L_content + beta*L_style

        print("G4: ", G)

        # discriminator loss
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)
        self.d_loss = d_loss_real + d_loss_fake

        # generator loss
        self.g_loss = - d_loss_fake

        """ Gradient Penalty """
        alpha = tf.random_uniform(shape=self.raw_image.get_shape(), minval=0., maxval=1.)
        print("G: ", G)
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
            self.s_optim = tf.train.AdamOptimizer(self.learning_rate,
                                                  beta1=self.beta1).minimize(self.L_total, var_list=g_vars)

        """ Testing """
        # for test
        self.fake_images = self.generator(self.raw_image, training=False, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.L_content_sum = tf.summary.scalar("cost_content", self.L_content)
        self.L_style_sum = tf.summary.scalar("cost_style", self.L_style)
        self.L_total_sum = tf.summary.scalar("cost_total_", self.L_total)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.s_sum = tf.summary.merge([self.L_content_sum, self.L_style_sum, self.L_total_sum])

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
            print("self.num_batches: ", self.num_batches)
            print("checkpoint_counter: ", checkpoint_counter)
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            counter = checkpoint_counter
            print( "[*] Load SUCCESS")
        else:
            start_epoch = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        style_image = load_style_image(self.style_image_path, self.shape)
        for epoch in range(start_epoch, self.epoch):

            for idx in range(0, self.num_batches):
                batch_images = self.dataloader.next_batch(self.batch_size)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.raw_image: batch_images})
                self.writer.add_summary(summary_str, counter)

                # update style network
                _, summary_str, L_content, L_style, L_total = self.sess.run([self.s_optim, self.s_sum, self.L_content, self.L_style, self.L_total],
                                                                                     feed_dict={self.raw_image: batch_images, self.sty_image: style_image})
                self.writer.add_summary(summary_str, counter)

                # update G network
                if (counter-1) % self.disc_iters == 0:
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                           feed_dict={self.raw_image: batch_images})
                    self.writer.add_summary(summary_str, counter)

                counter += 1

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss:%.8f, content_loss: %.8f, style_loss: %.8f, total_loss: %.8f"
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, L_content, L_style, L_total))

                # save training results for every 300 steps
                if np.mod(counter, 1) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.raw_image: batch_images})
                    tot_num_samples = min(self.sample_num, self.batch_size)
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
        samplex = self.dataloader.next_batch(self.batch_size)
        samples = self.sess.run(self.fake_images, feed_dict={self.raw_image: samplex})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + '_epoch%03d' % epoch + '_test_all_classes.png')

    def _gram_matrix(self, tensor):
        shape = tensor.get_shape()

        # Get the number of feature channels for the input tensor,
        # which is assumed to be from a convolutional layer with 4-dim
        num_channels = int(shape[3])

        # Reshape the tensor so it is a 2-dim matrix. This essentially
        # flattens the contents of each feature-channel.
        matrix = tf.reshape(tensor, shape=[1, -1, num_channels])

        # Calculate the Gram-matrix as the matrix-product of
        # the 2-dim matrix with itself. This calulates the
        # dot-products of all combinations of the feature-channels.
        gram = tf.matmul(tf.transpose(matrix, perm=[0,2,1]), matrix)

        return gram

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, 'style-gan.model'), global_step=step)