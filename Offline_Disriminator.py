import tensorflow as tf
from Synth_data import random_input
import pickle
import numpy as np

class Off_Discrim:

    def discriminator(self, data, config, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            # correct?
            if reuse:
                scope.reuse_variables()

            x = tf.layers.flatten(data)
            x = tf.layers.dense(x, config[0], tf.nn.relu)
            x = tf.layers.dense(x, config[1], tf.nn.relu)
            x = tf.layers.dense(x, config[2], tf.nn.relu)
            x = tf.layers.dense(x, config[3], tf.nn.relu)
            x = tf.layers.dense(x, 1, tf.nn.sigmoid)

            # output = tf.nn.sigmoid(logits)
        return x, 0


    def __init__(self, config, batch_size, seq_length):

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.D = [seq_length, 1]

        self.X = tf.placeholder(tf.float32, [batch_size, seq_length, 1])
        self.N = tf.placeholder(tf.float32, [batch_size, seq_length, 1])

        self.test_vector = tf.placeholder(tf.float32, [15000, seq_length, 1])

        # self.W_out_D = tf.Variable(tf.truncated_normal([hidden_units_d, 1]))
        # self.b_out_D = tf.Variable(tf.truncated_normal([1]))

        self.D_real, self.D_logit_real = self.discriminator(self.X, config)
        self.D_fake, self.D_logit_fake = self.discriminator(self.N, config, reuse=True)

        self.prob_tf, _ = self.discriminator(self.test_vector, config, reuse=True)

        self.discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

        # self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real,
        #                                                                      labels=tf.ones_like(self.D_logit_real)))
        # self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake,
        #                                                                      labels=tf.zeros_like(self.D_logit_fake)))

        self.D_loss_real = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=tf.ones_like(self.D_real),
                                                                               output=self.D_real))

        self.D_loss_fake = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=tf.zeros_like(self.D_fake),
                                                                               output=self.D_fake))

        self.D_loss = self.D_loss_real + self.D_loss_fake

        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(self.D_loss, var_list=self.discriminator_vars)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)


    def fit(self, train_data, train_size, trained_epoch, noise_data, tyype, trained_on, round, path_save, train_with=''):

        train_data = train_data.reshape([len(train_data)] + self.D)

        choices = np.random.choice(len(train_data), size=train_size, replace=False)
        train_data = train_data[choices]

        if train_with == 'other_data':
            all_datas = ['Periodic', 'Smooth', 'Cnsm', 'Wind-Power', 'Wind-Speed', 'Solar']

            list_data = []

            for data_type in all_datas:
                if data_type == tyype:
                    continue
                if data_type == 'Periodic':
                    signals_temp = random_input(seq_length=100, num_samples=22000)
                    signals_temp = signals_temp.flatten()
                    extra = len(signals_temp) % self.seq_length
                    if extra != 0:
                        signals_temp = signals_temp[:-extra]
                    list_data.append(signals_temp)
                    continue

                f = open('Saved-vecs/' + data_type + '/all-data' + '.pckl', 'rb')
                signals_temp = pickle.load(f)
                f.close()

                signals_temp = signals_temp.flatten()

                if data_type == 'Cnsm':
                    signals_temp = np.concatenate((signals_temp, signals_temp, signals_temp, signals_temp))

                extra = len(signals_temp) % self.seq_length
                if extra != 0:
                    signals_temp = signals_temp[:-extra]

                list_data.append(signals_temp)

        num_epochs = 300
        d_costs = []

        for epoch in range(num_epochs):
            if train_with == 'other_data':
                noise_cur = list_data[epoch % 5]
                noise_cur = noise_cur.reshape([-1] + self.D)

                choices = np.random.choice(len(noise_cur), size=train_size, replace=False)
                noise_cur = noise_cur[choices]
            elif train_with == 'early_epochs':
                if epoch < 100:
                    noise_cur = noise_data[epoch % 10]

                elif epoch < 200:
                    noise_cur = noise_data[epoch % 10 + 10]
                else:
                    noise_cur = noise_data[epoch % 10 + 20]
            elif train_with == 'relative':
                noise_cur = noise_data[epoch % 10]
            elif train_with == 'gen_set_3':
                noise_cur = noise_data[epoch % 3]
            elif train_with == 'gen_set_2':
                noise_cur = noise_data[epoch % 2]

            noise_cur = noise_cur.reshape([len(noise_cur)] + self.D)

            for i in range(len(train_data) // self.batch_size):

                batch_real = train_data[i * self.batch_size:(i + 1) * self.batch_size]
                batch_noise = noise_cur[i*self.batch_size:(i+1)*self.batch_size]

                _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss],feed_dict={self.X: batch_real,
                                                     self.N: batch_noise})

            d_costs.append(D_loss_curr)

            print('Epoch-- ' + str(epoch) + ' is done!')

        self.saver.save(self.sess, path_save + trained_on + trained_epoch + '-' + train_with + '-' + str(train_size)
                        + '-' + round)






























































