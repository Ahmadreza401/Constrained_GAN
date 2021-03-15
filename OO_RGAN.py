import numpy as np
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpt
import colorsys as cls
import statistics as stat
from sklearn.model_selection import train_test_split
import csv
import pickle
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio
import Synth_data as sd


class RGAN:

    def generator(self, z, c=None):
        with tf.variable_scope("generator") as scope:
            # each step of the generator takes a random seed + the conditional embedding
            # repeated_encoding = tf.tile(c, [1, tf.shape(z)[1]])
            # repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(z)[0], tf.shape(z)[1],
            #                                                    cond_dim])
            # generator_input = tf.concat([repeated_encoding, z], 2)

            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_g, state_is_tuple=True)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                sequence_length=[seq_length] * batch_size,
                inputs=z)
            rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
            logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
            output_2d = tf.nn.tanh(logits_2d)
            output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
        return output_3d

    def discriminator(self, x, c=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            # correct?
            if reuse:
                scope.reuse_variables()

            # each step of the generator takes one time step of the signal to evaluate +
            # its conditional embedding
            # repeated_encoding = tf.tile(c, [1, tf.shape(x)[1]])
            # repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(x)[0], tf.shape(x)[1],
            #                                                    cond_dim])
            # decoder_input = tf.concat([repeated_encoding, x], 2)

            cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_units_d, state_is_tuple=True,
                                           reuse=tf.get_variable_scope().reuse)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=x)
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, self.hidden_units_g])
            logits = tf.matmul(rnn_outputs_flat, W_out_D) + b_out_D

            # logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D

            output = tf.nn.sigmoid(logits)
        return output, logits

    # Latent Space Sampler
    def sample_Z(self, batch_size, seq_length, latent_dim, use_time=False, use_noisy_time=False):
        sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
        if use_time:
            print('WARNING: use_time has different semantics')
            sample[:, :, 0] = np.linspace(0, 1.0 / seq_length, num=seq_length)
        return sample

    def train_generator(self, batch_idx, offset):
        # update the generator
        for g in range(G_rounds):
            _, G_loss_curr = self.sess.run([G_solver, G_loss],
                                      feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time,)})
        return G_loss_curr


    def train_discriminator(self, batch_idx, offset):
        # update the discriminator
        for d in range(D_rounds):
            # using same input sequence for both the synthetic data and the real one,
            # probably it is not a good idea...
            X_mb = self.get_batch(train_seqs, batch_idx + d + offset, batch_size)
            _, D_loss_curr = self.sess.run([D_solver, D_loss],
                                      feed_dict={self.X: X_mb,
                                                 self.Z: self.sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)})

        return D_loss_curr

    def __init__(self):

        self.lr = 0.1
        batch_size = 30

        seq_length = 100
        num_generated_features = 1
        hidden_units_d = 100
        self.hidden_units_g = 100
        latent_dim = 10  # dimension of the random latent space
        # cond_dim = train_targets.shape[1]  # dimension of the condition




        Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
        W_out_G = tf.Variable(tf.truncated_normal([self.hidden_units_g, num_generated_features]))
        b_out_G = tf.Variable(tf.truncated_normal([num_generated_features]))

        X = tf.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
        W_out_D = tf.Variable(tf.truncated_normal([self.hidden_units_d, 1]))
        b_out_D = tf.Variable(tf.truncated_normal([1]))

        G_sample = self.generator(Z)
        D_real, D_logit_real = self.discriminator(X)
        D_fake, D_logit_fake = self.discriminator(G_sample, reuse=True)

        generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
        discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                                             labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                             labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                        labels=tf.ones_like(D_logit_fake)))

        D_solver = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(D_loss, var_list=discriminator_vars)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator_vars)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def fit(self, X, a):



        if not os.path.isdir(experiment_id):
            os.mkdir(experiment_id)

        X_mb_vis = self.get_batch(train_seqs, 0, batch_size)
        # plot the ouput from the same seed
        vis_z = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)

        vis_sample = self.sess.run(G_sample, feed_dict={Z: vis_z})

        # Denormalizing vis_sample
        # vis_sample = ReadInput.denormalize(vis_sample)
        sd.save_plot_sample(vis_sample[0:7], '0', 'first_sample_data', path=experiment_id)

        # visualise some real samples
        vis_real = np.float32(vali_seqs[np.random.choice(len(vali_seqs), size=batch_size), :, :])

        # Denormalizing vis_real
        # vis_real = ReadInput.denormalize(vis_real)
        sd.save_plot_sample(samples[0:7], '0', 'real_data', path=experiment_id)

        # trace = open('./experiments/traces/' + identifier + '.trace.csv', 'w')
        # fields_names = ['D_loss', 'G_loss']
        # writer = csv.DictWriter(trace, fieldnames= fields_names)
        # trace.write('epoch D_loss G_loss time\n')
        print('epoch\tD_loss\tG_loss\ttime\n')

        samples, peak_times, mean_peak_times, magnitude_peaks, mean_magnitudes = sd.continuous_input(case=2, tipe='periodic')

        sd.save_plot_sample(samples[0:7], '0', 'first_sample_data', path='test', show=True)

        train_seqs, vali_test = train_test_split(samples, test_size=0.4)
        vali_seqs, test_seqs = train_test_split(vali_test, test_size=0.6)

        print("data loaded.")

        # training config

        num_epochs = 12
        D_rounds = 1  # number of rounds of discriminator training
        G_rounds = 3  # number of rounds of generator training
        use_time = False  # use one latent dimension as time


        experiment_id = 'RGAN'
        identifier = id
        num_epochs = 200

        d_costs = []
        g_costs = []
        t0 = time.time()

        for num_epoch in range(num_epochs):
            # we use D_rounds + G_rounds batches in each iteration
            for batch_idx in range(0, int(len(train_seqs) / self.batch_size) - (D_rounds + G_rounds), D_rounds + G_rounds):
                # we should shuffle the data instead
                if num_epoch % 2 == 0:
                    G_loss_curr = self.train_generator(batch_idx, 0)
                    D_loss_curr = self.train_discriminator(batch_idx, G_rounds)
                else:
                    D_loss_curr = self.train_discriminator(batch_idx, 0)
                    G_loss_curr = self.train_generator(batch_idx, D_rounds)

            d_costs.append(D_loss_curr)
            g_costs.append(G_loss_curr)

            # plt.clf()
            # plt.plot(d_costs, label='discriminator cost')
            # plt.plot(g_costs, label='generator cost')
            # plt.legend()
            # plt.savefig(experiment_id + '/cost_vs_iteration.png')

            t = time.time() - t0
            print(num_epoch, '\t', D_loss_curr, '\t', G_loss_curr, '\t', t)

            # record/visualise
            # writer.writerow({'D_loss': D_loss_curr, 'G_loss': G_loss_curr})
            # trace.flush()
            # if num_epoch % 10 == 0:
            #     trace.flush()

            vis_sample = self.sess.run(self.G_sample, feed_dict={self.Z: vis_z})
            sd.save_plot_sample(vis_sample[0:7], '_' + '_generated' + "_epoch" + str(num_epoch).zfill(4),
                             identifier, path=experiment_id)
            # plotting.vis_sine_waves(vis_sample, seq_length, identifier=identifier, idx=num_epoch + 1)
        return None






















































