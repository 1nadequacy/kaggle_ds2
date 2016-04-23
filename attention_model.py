"""Recurrent attention model written in Theano."""

from disconnected import disconnected_grad
import adam

import theano
from theano import tensor as T
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from collections import OrderedDict


class Attention(object):

    def __init__(self, input_dim=(28, 28), input_channels=1,
                 down_sample=4, output_dim=10,
                 nh=100, std=0.1, image_buffer=0.15,
                 num_filters=4, filter_dim=3, pool_dim=2,
                 n_steps=5):
        assert input_dim[0] == input_dim[1]
        assert input_dim[0] % down_sample == 0
        assert input_dim[0] / down_sample % 2 == 0  # for conv pooling
        self.input_dim = input_dim[0]  # image dimension
        self.input_channels = input_channels  # number of channels in image
        self.down_sample = down_sample  # how much to downsample image for glimpse
        self.glimpse_dim = self.input_dim / self.down_sample
        self.output_dim = output_dim  # output dimension
        self.nh = nh  # hidden dimension
        self.std = std  # guassian std
        self.image_buffer = image_buffer  # proportional room from edge of image
        self.n_steps = n_steps  # number of steps for RNN
        self.pad = max(1, 2 + int(self.glimpse_dim / 2. - self.image_buffer * self.input_dim / 2.))
        self.num_filters = num_filters
        self.filter_dim = filter_dim
        self.pool_dim = pool_dim
        self.flatten_size = self.num_filters * self.glimpse_dim ** 2 / self.pool_dim ** 2

    def setup_params(self):
        self.srng = RandomStreams(seed=888)

        self.W_l = theano.shared(random_uniform((2, self.nh)))
        self.b_l = theano.shared(np.zeros(2, dtype=theano.config.floatX))  # TODO

        self.W_a = theano.shared(random_uniform((self.output_dim, self.nh)))
        self.b_a = theano.shared(np.zeros(self.output_dim, dtype=theano.config.floatX))

        self.W_hx = theano.shared(random_uniform((self.nh, self.flatten_size)))
        self.b_hx = theano.shared(np.zeros(self.nh, dtype=theano.config.floatX))
        self.W_hg = theano.shared(random_uniform((self.nh, self.flatten_size)))
        self.b_hg = theano.shared(np.zeros(self.nh, dtype=theano.config.floatX))

        fan_in = self.input_channels * self.filter_dim ** 2
        fan_out = self.num_filters * self.filter_dim ** 2
        self.Wconv_x = theano.shared(random_uniform((self.num_filters, self.input_channels, self.filter_dim, self.filter_dim),
                                                    scale=1. / (fan_in + fan_out) ** 0.5))
        self.Wconv_g = theano.shared(random_uniform((self.num_filters, self.input_channels, self.filter_dim, self.filter_dim),
                                                    scale=1. / (fan_in + fan_out) ** 0.5))

        self.W_hh = theano.shared(random_uniform((self.nh, self.nh)))
        self.b_hh = theano.shared(np.zeros(self.nh, dtype=theano.config.floatX))

        self.l0 = theano.shared(np.zeros(2, dtype=theano.config.floatX))
        self.h0 = theano.shared(np.zeros(self.nh, dtype=theano.config.floatX))

        self.base_reward = theano.shared(np.zeros(1, dtype=theano.config.floatX))

        self.nn_params = [
            self.W_a, self.b_a,
            self.W_hx, self.b_hx, self.W_hg, self.b_hg,
            self.Wconv_x, self.Wconv_g,
            self.W_hh, self.b_hh,
            self.h0]

    def sample_normal(self):
        return self.srng.normal(size=(2,), std=self.std)

    def gaussian_pdf(self, vec, mean):
        return (1.0 / (T.sqrt(2 * np.pi) * self.std) *
                T.exp(-0.5 * T.sqr(vec - mean).sum() / self.std ** 2))

    def location_network(self, h):
        return T.tanh(T.dot(self.W_l, h) + self.b_l)

    def glimpse_network(self, x, sampled_l):
        """x is input image, sampled_l is 2d focus location."""
        # clip location
        sampled_l = T.clip(sampled_l, self.image_buffer - 1, 1 - self.image_buffer)

        # map to pixels
        sampled_l = sampled_l * self.input_dim / 2.0 + self.input_dim / 2.0 + self.pad
        sampled_l = T.cast(T.round(sampled_l), 'int32')

        start_x = sampled_l[0] - self.glimpse_dim // 2
        end_x = sampled_l[0] + (self.glimpse_dim + 1) // 2
        start_y = sampled_l[1] - self.glimpse_dim // 2
        end_y = sampled_l[1] + (self.glimpse_dim + 1) // 2
        return x[:, :, start_x:end_x, start_y:end_y]

    def classifier_network(self, h):
        out = T.nnet.sigmoid(T.dot(self.W_a, h) + self.b_a)
        return out

    def conv_image_network(self, x, glimpse):
        conv_x = T.nnet.relu(T.nnet.conv2d(
            input=x, filters=self.Wconv_x,
            input_shape=(1, self.input_channels, self.glimpse_dim, self.glimpse_dim),
            filter_shape=self.Wconv_x.get_value().shape,
            border_mode='half'))
        conv_g = T.nnet.relu(T.nnet.conv2d(
            input=glimpse, filters=self.Wconv_g,
            input_shape=(1, self.input_channels, self.glimpse_dim, self.glimpse_dim),
            filter_shape=self.Wconv_g.get_value().shape,
            border_mode='half'))

        pooled_x = downsample.max_pool_2d(
            input=conv_x, ds=(self.pool_dim, self.pool_dim), 
            ignore_border=False)
        pooled_g = downsample.max_pool_2d(
            input=conv_g, ds=(self.pool_dim, self.pool_dim), 
            ignore_border=False)

        return pooled_x, pooled_g

    def image_network(self, x, glimpse):
        """
        x is full downsampled image
        glimpse is focused region

        """
        x, glimpse = self.conv_image_network(x, glimpse)

        h_x = T.tanh(T.dot(self.W_hx, x.flatten()) + self.b_hx)
        h_glimpse = T.tanh(T.dot(self.W_hg, glimpse.flatten()) + self.b_hg)

        return h_x, h_glimpse

    def recurrence(self, l_p, h_p, x, x_downsampled):
        """
        l_p is sampled location
        h_p is previous hidden state
        x is full image
        x_downsampled is downsampled image

        """
        g_t = self.glimpse_network(x, l_p)
        h_x, h_glimpse = self.image_network(x_downsampled, g_t)
        h_t = T.tanh(T.dot(self.W_hh, h_p) + h_x + h_glimpse + self.b_hh)
        a_t = self.classifier_network(h_t)

        # sample next location
        l_t = self.location_network(h_t)
        sample = self.sample_normal() + l_t
        sampled_l = sample # T.tanh(sample)

        # grad for reinforce algorithm
        sampled_pdf = self.gaussian_pdf(disconnected_grad(sample), l_t)
        wl_grad = T.grad(T.log(sampled_pdf), self.W_l)

        return sampled_l, h_t, a_t, wl_grad

    def full_network(self, x):
        """x is original image."""
        x_downsampled = downsample.max_pool_2d(
            x[:, :, self.pad:-self.pad, self.pad:-self.pad],
            (self.down_sample, self.down_sample),
            ignore_border=False)

        [l_ts, h_ts, a_ts, wl_grads], _ = theano.scan(
            fn=self.recurrence,
            outputs_info=[self.l0, self.h0, None, None],
            non_sequences=[x, x_downsampled],
            n_steps=self.n_steps)

        wl_grad = T.mean(wl_grads[:-1], axis=0)
        return a_ts[-1], l_ts, wl_grad

    def loss(self, y, out):
        return T.sum(T.sqr(y - out))

    def reward(self, loss):
        return (self.output_dim / 2.0 - loss) / self.output_dim

    def get_functions(self):
        x = T.ftensor4('x')  # input
        y = T.fvector('y')  # expected output
        lr = T.scalar('lr')  # learning rate

        pad_x = T.zeros([1, self.input_channels] +
                        [2 * self.pad + self.input_dim] * 2)
        pad_x = T.set_subtensor(
            pad_x[:, :, self.pad:-self.pad, self.pad:-self.pad], x)

        out, locations, wl_grad = self.full_network(pad_x)
        loss = self.loss(y, out)
        reward = self.reward(loss)

        rl_gradient = (reward - self.base_reward).sum() * wl_grad / T.sqrt(T.sum(wl_grad ** 2))
        br_gradient, = T.grad(T.sqr(reward - self.base_reward).sum(),
                              [self.base_reward])

        train_updates = adam.Adam(loss, self.nn_params)
        train_updates[self.W_l] = self.W_l + lr * rl_gradient  # gradient ascent
        train_updates[self.base_reward] = self.base_reward - lr * br_gradient

        train = theano.function(
            inputs=[x, y, lr],
            outputs=[out, loss, reward],
            updates=train_updates)

        predict = theano.function(
            inputs=[x],
            outputs=[out, locations])

        return train, predict


def random_uniform(shape, scale=0.1):
    return scale * np.random.uniform(-1.0, 1.0, shape).astype(theano.config.floatX)
