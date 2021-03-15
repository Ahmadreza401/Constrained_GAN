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

import json

from numpy import median, vstack, einsum

experiment_id = './' + 'CGAN-RNN'
identifier = 'Single_Peak'

########################
# Functions
########################


def get_batch(samples, batch_idx, batch_size):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    return samples[start_pos:end_pos]


def save_plot_sample(samples, idx, identifier, show = False, n_samples=6, num_epochs=None, ncol=2, path='./'):
    assert n_samples <= samples.shape[0]
    assert n_samples % ncol == 0
    sample_length = samples.shape[1]

    if not num_epochs is None:
        col = cls.hsv_to_rgb((1, 1.0 * (idx) / num_epochs, 0.8))
    else:
        col = 'grey'

    x_points = np.arange(sample_length)

    nrow = int(n_samples / ncol)
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    for m in range(nrow):
        for n in range(ncol):
            # first column
            sample = samples[n * nrow + m, :, 0]
            axarr[m, n].plot(x_points, sample, color=col)
            axarr[m, n].set_ylim(0, 1)
    for n in range(ncol):
        axarr[-1, n].xaxis.set_ticks(range(0, sample_length, int(sample_length / 4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace=0.15)
    if show:
        plt.show()
    else:
        fig.savefig(path + "/" + identifier + str(idx).zfill(4) + ".png")
        print(path + "/" + identifier + str(idx).zfill(4) + ".png")
    plt.clf()
    plt.close()
    return


def generate_input(seq_length=30, num_samples=28*5*100, num_signals=1,
        freq_low=1, freq_high=5, amplitude_low = 0.5, amplitude_high=0.9, case=1, tipe='periodic', **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    argmaxs = []
    maxs = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            # f = np.random.uniform(low=freq_high, high=freq_low)  # frequency
            A = np.random.uniform(low=amplitude_low, high=amplitude_high)  # amplitude
            offset = 0
            # offset
            # offset = np.random.uniform(low=-np.pi / 10, high=np.pi / 10)
            # if case == 1:
            #     A = 1
            # elif case == 2:
            #     A = np.random.uniform(low=0.30, high=0.99)  # amplitude
            if tipe == 'periodic':
                A = np.random.uniform(low=0.30, high=0.99)  # amplitude
                signals.append(A * np.sin(np.pi * 4 * ix / float(seq_length) + offset))
            elif tipe == 'single':
                signals.append(A * -np.cos(np.pi * 2 * ix / float(seq_length) + offset))

            # ###
            argmaax = np.argmax(np.array(signals))
            maax = np.max(np.array(signals))
        samples.append(np.array(signals).T)
        argmaxs.append(argmaax)
        maxs.append(maax)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples, dtype=np.float32)
    argmaxs = np.array(argmaxs, dtype=np.float32)
    maxs = np.array(maxs, dtype=np.float32)
    mean_argmaxs = np.round(np.mean(argmaxs))
    mean_maxs = np.mean(maxs)
    return samples, argmaxs, mean_argmaxs, maxs, mean_maxs

def soft_arg_max(matrix, indexs, axis=1):
    multed = tf.scalar_mul(10000, matrix)
    softed_mul = tf.nn.softmax(multed, axis)

    resahped_softed = tf.reshape(softed_mul, shape=[softed_mul.shape[0]*softed_mul.shape[1]])

    elems = (resahped_softed, indexs)
    softed_arg_maxs_unshaped = tf.map_fn(lambda x: x[0] * x[1], elems, dtype=tf.float32)

    softed_arg_maxs = tf.reshape(softed_arg_maxs_unshaped, shape=[softed_mul.shape[0], softed_mul.shape[1], -1])
    soft_arg_maxs = tf.reduce_sum(softed_arg_maxs, 1)

    return soft_arg_maxs

def mmd(constrained, core):


    g_2 = tf.Graph()

    with g_2.as_default():
        sess1 = tf.Session()
        # Has been vali_seqs
        heuristic_sigma_training = median_pairwise_distance(vali_seqs)
        best_mmd2_so_far = 1000

        batch_multiplier = 5000 // batch_size
        eval_size = batch_multiplier * batch_size
        eval_eval_size = int(0.2 * eval_size)

        eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
        eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
        n_sigmas = 2

        sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(
            value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))

        sess1.run(tf.global_variables_initializer())

        mmd2, that = mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)

        # with tf.variable_scope("SIGMA_optimizer"):
        #     sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
        #     # sigma_solver = tf.train.AdamOptimizer().minimize(-that, var_list=[sigma])
        #     # sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
        # sigma_opt_iter = 2000
        # sigma_opt_thresh = 0.001
        # sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]

        eval_Z = sample_Z(eval_size, seq_length, latent_dim, use_time)

        eval_sample = np.empty(shape=(eval_size, seq_length, 1))

        for i in range(batch_multiplier):
            eval_sample[i * batch_size:(i + 1) * batch_size, :, :] = sess.run(G_sample, feed_dict={
                Z: eval_Z[i * batch_size:(i + 1) * batch_size]})
        eval_sample = np.float32(eval_sample)

        eval_real = np.float32(
            vali_seqs[np.random.choice(len(vali_seqs), size=eval_size), :, :])

        eval_eval_real = eval_real[:eval_eval_size]
        eval_test_real = eval_real[eval_eval_size:]
        eval_eval_sample = eval_sample[:eval_eval_size]
        eval_test_sample = eval_sample[eval_eval_size:]

        ## MMD
        # reset ADAM variables
        # sess.run(tf.initialize_variables(sigma_opt_vars))
        # sigma_iter = 0
        # that_change = sigma_opt_thresh * 2
        # old_that = 0
        # while that_change > sigma_opt_thresh and sigma_iter < sigma_opt_iter:
        #     new_sigma, that_np, _ = sess.run([sigma, that, sigma_solver],
        #                                      feed_dict={eval_real_PH: eval_eval_real, eval_sample_PH: eval_eval_sample})
        #     that_change = np.abs(that_np - old_that)
        #     old_that = that_np
        #     sigma_iter += 1
        # opt_sigma = sess.run(sigma)

        mmd2, ratio = sess1.run(mix_rbf_mmd2_and_ratio(constrained, core, biased=False, sigmas=sigma))
        # tf.reset_default_graph()

    return mmd2, ratio


########################
# Loading Data, Training Configuration
########################


# samples, peak_times, mean_peak_times, magnitude_peaks, mean_magnitudes = generate_input(case=2, tipe='periodic')
samples2, peak_times, mean_peak_times, magnitude_peaks, mean_magnitudes = sd.continuous_input(seq_length=100, case=2, tipe='periodic')

# f = open('RGAN-1/sv/01-solar-all' + '.pckl', 'rb')
# samples1 = pickle.load(f)
# f.close()

samples = samples2.reshape([-1, 144, 1])
# samples = samples1[:35000]

type_exp = 'Perodic'

# save_plot_sample(samples[0:7], '0', 'first_sample_2-' + type_exp, path='test', show=True)

train_seqs, vali_test = train_test_split(samples, test_size=0.4)
vali_seqs, test_seqs = train_test_split(vali_test, test_size=0.6)

# print ("data loaded.")
# print(identifier)

# training config
lr = 0.1
batch_size = 30
num_epochs = 300
D_rounds = 1  # number of rounds of discriminator training
G_rounds = 3  # number of rounds of generator training
use_time = False  # use one latent dimension as time

seq_length = train_seqs.shape[1]
num_generated_features = train_seqs.shape[2]
hidden_units_d = 100
hidden_units_g = 100
latent_dim = 10  # dimension of the random latent space
# cond_dim = train_targets.shape[1]  # dimension of the condition


########################
# GENERATOR, DISCRIMINATOR
########################

def generator(z, c=None):
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


def discriminator(x, c= None, reuse=False):
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

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=x)
        rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits = tf.matmul(rnn_outputs_flat, W_out_D) + b_out_D

        # logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D

        output = tf.nn.sigmoid(logits)
    return output, logits


# Latent Space Sampler
def sample_Z(batch_size, seq_length, latent_dim, use_time=False, use_noisy_time=False):
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    if use_time:
        print('WARNING: use_time has different semantics')
        sample[:, :, 0] = np.linspace(0, 1.0/seq_length, num=seq_length)
    return sample


########################
# Trainers
########################


def train_generator(batch_idx, offset):
    # update the generator
    for g in range(G_rounds):
        _, G_loss_curr = sess.run([G_solver, G_loss],
                                  feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time,)})
    return G_loss_curr


def train_discriminator(batch_idx, offset):
    # update the discriminator
    for d in range(D_rounds):
        # using same input sequence for both the synthetic data and the real one,
        # probably it is not a good idea...
        X_mb = get_batch(train_seqs, batch_idx + d + offset, batch_size)
        _, D_loss_curr, p1, p2, pr1, pr2 = sess.run([D_solver, D_loss, D_fake, D_logit_fake, D_real, D_logit_real],
                                  feed_dict={X: X_mb,
                                             Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)})

    return D_loss_curr


########################
# Training
########################

def initial_plots(experiment_name=experiment_id, id=identifier, epcs=num_epochs):

    experiment_id = experiment_name
    identifier = id
    num_epochs = epcs

    # directory where the data will be saved
    if not os.path.isdir(experiment_id):
        os.mkdir(experiment_id)

    X_mb_vis = get_batch(train_seqs, 0, batch_size)
    # plot the ouput from the same seed
    vis_z = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)

    vis_sample = sess.run(G_sample, feed_dict={Z: vis_z})

    # Denormalizing vis_sample
    # vis_sample = ReadInput.denormalize(vis_sample)
    save_plot_sample(vis_sample[0:7], '0', 'first_sample_2-' + type_exp, path=experiment_id)

    # visualise some real samples
    vis_real = np.float32(vali_seqs[np.random.choice(len(vali_seqs), size=batch_size), :, :])

    # Denormalizing vis_real
    # vis_real = ReadInput.denormalize(vis_real)
    save_plot_sample(samples[0:7], '0', 'real_data_2-' + type_exp, path=experiment_id)

    # trace = open('./experiments/traces/' + identifier + '.trace.csv', 'w')
    # fields_names = ['D_loss', 'G_loss']
    # writer = csv.DictWriter(trace, fieldnames= fields_names)
    # trace.write('epoch D_loss G_loss time\n')
    print('epoch\tD_loss\tG_loss\ttime\n')

    return vis_z


def GAN_train(vis_z, experiment_name=experiment_id, id=identifier, epcs=num_epochs):

    experiment_id = experiment_name
    identifier = id
    num_epochs = epcs

    d_costs = []
    g_costs = []
    te = t0 = time.time()

    for num_epoch in range(num_epochs):

        G_loss_curr = D_loss_curr = 0
        # we use D_rounds + G_rounds batches in each iteration
        for batch_idx in range(0, int(len(train_seqs) / batch_size) - (D_rounds + G_rounds), D_rounds + G_rounds):
            # we should shuffle the data instead
            if num_epoch % 2 == 0:
                G_loss_curr = train_generator(batch_idx, 0)
                D_loss_curr = train_discriminator(batch_idx, G_rounds)
            else:
                D_loss_curr = train_discriminator(batch_idx, 0)
                G_loss_curr = train_generator(batch_idx, D_rounds)

        d_costs.append(D_loss_curr)
        g_costs.append(G_loss_curr)

        # plt.clf()
        # plt.plot(d_costs, label='discriminator cost')
        # plt.plot(g_costs, label='generator cost')
        # plt.legend()
        # plt.savefig(experiment_id + '/cost_vs_iteration.png')

        te = time.time() - te
        print(num_epoch, '\t', D_loss_curr, '\t', G_loss_curr, '\t', te)
        te = time.time()

        # record/visualise
        # writer.writerow({'D_loss': D_loss_curr, 'G_loss': G_loss_curr})
        # trace.flush()
        # if num_epoch % 10 == 0:
        #     trace.flush()
        # if num_epoch % 50 == 0:

        if num_epoch % 50 == 0:
            saver.save(sess, 'RGAN-1/sv/' + type_exp + '/2-' + type_exp + '-epoch--' + str(num_epoch))

            vis_sample = sess.run(G_sample, feed_dict={Z: vis_z})
            save_plot_sample(vis_sample[0:7], "_epoch" + str(num_epoch).zfill(4),
                             identifier, path=experiment_id)
        # print('A set of samples printed has printed !.')
        # plotting.vis_sine_waves(vis_sample, seq_length, identifier=identifier, idx=num_epoch + 1)

    t_overall = time.time() - t0

    print('Learning is finished after ' + str(t_overall) + ' Seconds.')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost curves')

    ax1.plot(range(len(d_costs)), d_costs, label='D-cost')
    ax1.plot(range(len(g_costs)), g_costs, label='G-costs')
    ax1.legend(loc="upper right")
    # plt.show()
    fig1.savefig("RGAN-1/00_"+ type_exp + '-2' + "costs.jpg", dpi=300)



    # samples_final = []
    # while len(samples_final) < 1000:
    #     samples_z = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
    #     samples_real = sess.run(G_sample, feed_dict={Z: samples_z})
    #     samples_real.reshape([len(samples_real), seq_length])
    #
    #     for samp_real in samples_real:
    #         if len(samples_final) < 1000:
    #             samples_final.append(samp_real)
    #
    # #Saving Sample in a .pckl file.
    # f = open('RGAN-1/sv/samples_final' + '.pckl', 'wb')
    # pickle.dump(samples_final, f)
    # f.close()

    # samples_final = np.array(samples_final).reshape(([-1]))
    # samples_final = list(samples_final)

    # to_store = {'Number of Samples': 1000, 'Overall_time': t_overall, 'Cost_gen': g_costs,
    #             'Cost_disc': d_costs}
    # with open("RGAN-1/sv/final_samples.json", "w") as f:
    #     json.dump(to_store, f)


def train(experiment_name, id, epcs):

    viz = initial_plots(experiment_name=experiment_name, id= id, epcs=epcs)
    GAN_train(viz, experiment_name=experiment_name, id= id, epcs=epcs)


########################
# Applying Constraints
########################


def apply_constraint_1(runs):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    momentum_lr = 0.001

    mean_peak_times_real = np.ones([batch_size, 1], dtype=float)
    mean_peak_times_real[:, :] = mean_peak_times

    alpha = np.ones([batch_size, 1], dtype=float)

    indexs = np.ones([seq_length], dtype=float)
    indexs[0] = 1
    for i in range(1, seq_length):
        indexs[i] = i
    new_indexs = indexs

    for i in range(batch_size-1):
        new_indexs = np.concatenate((new_indexs, indexs))

    z_star = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
    x_test = get_batch(test_seqs, np.random.randint(low=0, high=test_seqs.shape[0]/batch_size), batch_size)
    lammbda = 0.05

    ### Starting Gradient Descent with Momentum for the first constraint

    # Z_const = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
    # X_opt = tf.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
    means_x_p = tf.placeholder(tf.float32, [batch_size, 1])
    alpha_x = tf.placeholder(tf.float32, [batch_size, 1])
    indexs_tensor = tf.placeholder(tf.float32, [batch_size*seq_length])

    m = 0
    v = 0

    for i in range(runs):

        actual_distance = tf.reduce_sum(tf.contrib.layers.flatten(tf.square(G_sample - X)), 1)

        arg_maxs = soft_arg_max(G_sample, indexs_tensor, 1)

        first_constraint_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.log(tf.abs(soft_arg_max(G_sample, indexs_tensor, 1)
                                                                                - means_x_p) - alpha_x)), 0)

        grad_first_constraint = tf.gradients(first_constraint_loss, Z)

        total_loss = actual_distance - lammbda*G_loss + first_constraint_loss

        gradients_total_loss = tf.gradients(total_loss, Z)

        grad_current_loss, fet_arg_maxs = sess.run([gradients_total_loss, grad_first_constraint], feed_dict={Z: z_star, X: x_test, means_x_p: mean_peak_times_real,
                                                                      alpha_x: alpha, indexs_tensor: new_indexs})

        ## Aplying the Momentum to the Gradient
        m_prev = np.copy(m)
        v_prev = np.copy(v)
        m = beta1 * m_prev + (1 - beta1) * grad_current_loss[0]
        v = beta2 * v_prev + (1 - beta2) * np.multiply(grad_current_loss[0], grad_current_loss[0])
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))

        z_star += - np.true_divide(momentum_lr * m_hat, (np.sqrt(v_hat) + eps))
        z_star = np.clip(z_star, -1, 1)
        print ('iteration  ' + i.__str__())

    return z_star

def apply_constraint_2(runs, alpha, core='average'):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    momentum_lr = 0.001

    core_value = np.ones([batch_size, 1], dtype=float)

    if core == 'average':
        core_value[:, :] = mean_magnitudes
    elif core == 'mode':
        med = stat.stdev()
        rounded_max = np.round(magnitude_peaks, decimals=2)
        mode = stat.mode(rounded_max)
        core_value[:, :] = mode

    alpha_vector = np.ones([batch_size, 1], dtype=float)
    alpha_vector[:, :] = alpha

    indexs = np.ones([seq_length, 1], dtype=float)
    indexs[1, 0] = 1
    for i in range(1, indexs.shape[0]):
        indexs[i, 0] = i


    z_star = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
    x_test = get_batch(test_seqs, np.random.randint(low=0, high=test_seqs.shape[0]/batch_size), batch_size)
    lammbda = 0.05

    ### Starting Gradient Descent with Momentum for the first constraint

    # Z_const = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
    # X_opt = tf.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
    core_function = tf.placeholder(tf.float32, [batch_size, 1])
    alpha_y = tf.placeholder(tf.float32, [batch_size, 1])

    m = 0
    v = 0

    for i in range(runs):

        actual_distance = tf.reduce_sum(tf.contrib.layers.flatten(tf.square(G_sample - X)), 1)

        # Sketch: |tf.max(G) - means_y_p| - beta
        second_constraint_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.log(tf.abs(tf.reduce_max(G_sample, 1)
                                                                                - core_function) - alpha_y)), 0)

        total_loss = actual_distance -lammbda*G_loss + second_constraint_loss

        gradients_total_loss = tf.gradients(total_loss, Z)

        grad_current_loss = sess.run(gradients_total_loss, feed_dict={Z: z_star, X: x_test, core_function: core_value,
                                                                      alpha_y: alpha_vector})

        ## Aplying the Momentum to the Gradient
        m_prev = np.copy(m)
        v_prev = np.copy(v)
        m = beta1 * m_prev + (1 - beta1) * grad_current_loss[0]
        v = beta2 * v_prev + (1 - beta2) * np.multiply(grad_current_loss[0], grad_current_loss[0])
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))

        z_star += - np.true_divide(momentum_lr * m_hat, (np.sqrt(v_hat) + eps))
        z_star = np.clip(z_star, -1, 1)
        print ('iteration  ' + i.__str__())

    return z_star


#########################
'''Fetch Samples'''


def fetch_samples_RGAN(num, file_path):
    saver.restore(sess, file_path)

    samples_final_f = []
    while len(samples_final_f) < num:
        samples_z_f = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
        samples_gen_f = sess.run(G_sample, feed_dict={Z: samples_z_f})
        samples_gen_f.reshape([len(samples_gen_f), seq_length])

        for samp_gen_f in samples_gen_f:
            if len(samples_final_f) < num:
                samples_final_f.append(samp_gen_f)
            else:
                break

    samples_final_f = np.array(samples_final_f)
    return samples_final_f
    # Saving Sample in a .pckl file.
    # f = open('RGAN-1/sv/01-RGAN-' + type_exp + '.pckl', 'wb')
    # pickle.dump(samples_final, f)
    # f.close()

########################
# TensorFlow Variable Setting
########################


# CG = tf.placeholder(tf.float32, [batch_size, train_targets.shape[1]])
# CD = tf.placeholder(tf.float32, [batch_size, train_targets.shape[1]])
Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
W_out_G = tf.Variable(tf.truncated_normal([hidden_units_g, num_generated_features]))
b_out_G = tf.Variable(tf.truncated_normal([num_generated_features]))

X = tf.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
W_out_D = tf.Variable(tf.truncated_normal([hidden_units_d, 1]))
b_out_D = tf.Variable(tf.truncated_normal([1]))

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample, reuse=True)

generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                                     labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                     labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(D_loss, var_list=discriminator_vars)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator_vars)




########################
# Running The Session
########################

sess = tf.Session()
saver = tf.train.Saver()
condition = 'trained_average'
input = 'periodic'

ckpt = tf.train.get_checkpoint_state('st-2-120/')
file_names = []
alphas_vis = []
iterations_str = ''
labels = []

# rounded_max = np.round(magnitude_peaks, decimals=2)
# mode = stat.mode(rounded_max)
#
# med = stat.median(rounded_max)

# men = stat.mean(magnitude_peaks)

generate_sampeles = False

if condition == 'trained_average' and 0:

    if input == 'periodic':
        # alphas_vis = [0.14, 0.24, 0.31, 0.40, 0.50]
        alphas_vis = [0.11, 0.14, 0.21, 0.24, 0.26, 0.28, 0.31, 0.33]
        # file_names = ['periodic-2/average-0.11', 'periodic-2/average-0.21', 'periodic-2/average-0.26',
        #               'periodic-2/average-0.28', 'periodic-2/average-0.33']
        # file_names = ['periodic-2/average-0.10', 'periodic-2/average-0.20', 'periodic-2/average-0.30',
        #               'periodic-2/average-0.40', 'periodic-2/average-0.50']
        file_names = ['periodic-2/average-0.11', 'periodic-2/average-0.10', 'periodic-2/average-0.21',
                      'periodic-2/average-0.20', 'periodic-2/average-0.26','periodic-2/average-0.28',
                      'periodic-2/average-0.30', 'periodic-2/average-0.33']
        labels = ["alpha = 0.11", "alpha = 0.14", "alpha = 0.21", "alpha = 0.24", "alpha = 0.26", "alpha = 0.28",
                  "alpha = 0.31", "alpha = 0.33"]
        iterations_str = '-400'

    elif input == 'single':
        file_names = ['single-peak/average-0.05', 'single-peak/average-0.10', 'single-peak/average-0.15',
                      'single-peak/average-0.20', 'single-peak/average-0.25']
        alphas_vis = [0.05, 0.10, 0.15, 0.20, 0.25]
        labels = ["alpha = 0.05", "alpha = 0.10", "alpha = 0.15", "0.20", "0.25"]
        iterations_str = '-300'

    color_sequence = [ '#1f77b4', '#aec7e8', '#e377c2', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94',  '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

    samples_reshaped = np.reshape(samples, newshape=[-1, seq_length])

    realsample = np.ones(shape=[seq_length])
    for i in range(samples_reshaped.shape[0]):
        max_cur = np.round(max(samples_reshaped[i, :]), 2)
        rounded_mean = np.round(mean_magnitudes, 2)

        if max_cur == rounded_mean:
            realsample[:] = samples_reshaped[i, :]
            break

    plt.clf()

    plt.plot(realsample, color= '#7f7f7f', label= "Real Sample", linewidth=3)

    # fig = plt.figure()
    omegas = []
    epsilons = []
    ps = []
    for i in range(len(alphas_vis)):
        f = open(file_names[i] + '.pckl', 'rb')
        z_opt = pickle.load(f)
        f.close()
        saver.restore(sess, file_names[i] + iterations_str)
        vis_sample = sess.run(G_sample, feed_dict={Z: z_opt})

        vis_sample_reshaped = np.reshape(a=vis_sample, newshape=[-1, seq_length])

        maxes = np.max(vis_sample_reshaped, axis=1)

        upper_graphs = np.ones(shape=[3, seq_length])
        lower_graphs = np.ones(shape=[3, seq_length])

        lower = upper = 0

        # Computing \omega
        satisfied = 0
        total = vis_sample_reshaped.shape[0]
        constrained = []


        for j in range(vis_sample_reshaped.shape[0]):
            if (mean_magnitudes + alphas_vis[i] < np.max(vis_sample_reshaped[j, :])) or (mean_magnitudes - alphas_vis[i] > np.max(vis_sample_reshaped[j, :])):
                satisfied += 1
                constrained.append(vis_sample_reshaped[j, :])

                if (mean_magnitudes + alphas_vis[i] < np.max(vis_sample_reshaped[j, :])) and (upper < 3):
                    upper_graphs[upper, :] = vis_sample_reshaped[j, :]
                    upper += 1

                elif (mean_magnitudes - alphas_vis[i] > np.max(vis_sample_reshaped[j, :])) and (lower < 3):
                    lower_graphs[lower, :] = vis_sample_reshaped[j, :]
                    lower += 1

        omega = satisfied / total

        omegas.append(omega)

        num_constrained = len(constrained)
        real = []
        num_real = 0
        cur = 0

        while num_real < num_constrained:
            max_cur = np.round(max(samples_reshaped[cur, :]), 2)
            rounded_mean = np.round(mean_magnitudes, 2)

            if max_cur == rounded_mean:
                real.append(samples_reshaped[cur, :])
                num_real += 1

            cur += 1

        array_constrained = np.array(constrained)
        array_real = np.array(real)


        # Computing  \epsilon

        if (omega != 0):
            epsilon, ratio = mmd(constrained=array_constrained, core=array_real)
            epsilons.append(epsilon)

            epsilon1, ratio1 = mmd(constrained=array_real, core=array_real)

            print('mmd of two similar' + str(epsilon1))
        else:
            epsilons.append(np.inf)


        # Compuing p
        samples_constrained_tf = tf.placeholder(tf.float32, [num_constrained, seq_length, num_generated_features])

        D_constrained, D_constrained_logits = discriminator(samples_constrained_tf, reuse=True)

        constrained_3d = np.reshape(array_constrained, newshape=[-1, seq_length, num_generated_features])

        p_v, p_v_1 = sess.run(fetches=[D_constrained, D_constrained_logits], feed_dict={samples_constrained_tf: constrained_3d})
        p = np.mean(p_v)
        p_1 = np.mean(p_v_1)

        ps.append(p)



        # Adding to the plot
        if lower > 0 or upper > 0:
            for j in range(lower):
                plt.plot(lower_graphs[j, :], color= color_sequence[i], label=labels[i])
            for j in range(upper):
                plt.plot(upper_graphs[j, :], color=color_sequence[i], label=labels[i])

        print('alpha =' + str(alphas_vis[i]) + ',    omega = ' + str(omega) + ',    epsilon = ' + str(epsilon) + ',    p = ' + str(p) +
              ',     p_1 = ' + str(p_1) )

    plt.legend(loc='1')
    plt.show()

    plt.savefig('periodic/different-alphas.jpg')

    plt.clf()

    plt.plot(alphas_vis, omegas, label='omega')
    plt.plot(alphas_vis, epsilons, label='epsilon')
    plt.plot(alphas_vis, ps, label='p')

    plt.xlabel('alpha')
    plt.legend()

    plt.show()


elif ckpt and ckpt.model_checkpoint_path and 0:
    # saver1 = tf.train.import_meta_graph('st-1-12/st_GAN-12.meta')
    # saver1.restore(sess, tf.train.latest_checkpoint('st-1-12/'))

    # saver.restore(sess, 'st-2-120/st_GAN-120')

    # Plotting after the restore
    # vis_z = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
    # vis_sample = sess.run(G_sample, feed_dict={Z: vis_z})
    # peaks = np.max(vis_sample, 1)
    # print(peaks)
    # save_plot_sample(vis_sample[0:7], '0', 'first_sample_data', path=experiment_id, show=True)

    # Compute the optimum z with respect to constraint_1
    # paths = ['single-peak/average-0.05', 'single-peak/average-0.10', 'single-peak/average-0.15',
    #          'single-peak/average-0.20', 'single-peak/average-0.25']
    # saver.restore(sess, 'CGAN-2-500/CGAN-Periodic-500')

    # vis_z = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
    #
    # vis_sample = sess.run(G_sample, feed_dict={Z: vis_z})
    #
    # maxes = np.max(vis_sample, axis=1)
    #
    # # save_plot_sample(vis_sample[0:7], '0', 'first_sample_data', path='test', show=True)
    #
    # max_max = np.max(magnitude_peaks)
    # min_max = np.min(magnitude_peaks)

    # alphas = [0.14, 0.24, 0.31, 0.40, 0.50]
    alphas = [0.11, 0.21, 0.26, 0.28, 0.33]


    # file_names = ['periodic-2/average-0.10', 'periodic-2/average-0.20', 'periodic-2/average-0.30',
    #               'periodic-2/average-0.40', 'periodic-2/average-0.50']
    # file_names = ['4-periodic/mode-0.10', '4-periodic/mode-0.20', '4-periodic/mode-0.30',
    #               '4-periodic/mode-0.40', '4-periodic/mode-0.50']
    file_names = ['periodic-2/average-0.11', 'periodic-2/average-0.21', 'periodic-2/average-0.26',
                  'periodic-2/average-0.28', 'periodic-2/average-0.33']

    # alphas = [0.053, 0.103, 0.153, 0.203, 0.253]
    # file_names = ['single-peak/average-0.05', 'single-peak/average-0.10', 'single-peak/average-0.15',
    #          'single-peak/average-0.20', 'single-peak/average-0.25']
    for i in range(4, 5):
        saver.restore(sess, 'CGAN-2-500/CGAN-Periodic-500')
        z_optimum = apply_constraint_2(runs=400, alpha=alphas[i], core='average')
        saver.save(sess, file_names[i], global_step=400)
        f = open(file_names[i]+'.pckl', 'wb')
        pickle.dump(z_optimum, f)
        f.close()

    # z_optimum = apply_constraint_2(runs=300, alpha=0.05, core='average')

    # Plotting with z_optimum
    # vis_sample = sess.run(G_sample, feed_dict={Z: z_optimum})
    # peaks = np.max(vis_sample, 1)
    # print(peaks)
    # save_plot_sample(vis_sample[0:7], '0', 'first_sample_data', path=experiment_id, show=True)


elif generate_sampeles:

    saver.restore(sess, 'RGAN-1/sv/1-'+type_exp)

    samples_final = []
    while len(samples_final) < 5000:
        samples_z = sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
        samples_gen = sess.run(G_sample, feed_dict={Z: samples_z})
        samples_gen.reshape([len(samples_gen), seq_length])

        for samp_gen in samples_gen:
            if len(samples_final) < 5000:
                samples_final.append(samp_gen)
            else:
                break

    # Saving Sample in a .pckl file.
    f = open('RGAN-1/sv/01-RGAN-' + type_exp + '.pckl', 'wb')
    pickle.dump(samples_final, f)
    f.close()

elif generate_sampeles and False:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    train(experiment_name='RGAN-1', id=type_exp + '-2', epcs=300)

    saver.save(sess, 'RGAN-1/sv/' + type_exp + '/2-' + type_exp)


########################
# Assessing the Results
########################
