import time
import tensorflow as tf
from cough_detector_10s_1c_max_pools import CoughDetectorModel
from torch.utils import data
from data_loader_sound import SpectogramLoader
import os
import numpy as np

epsilon = 1e-7


class Model(object):
    def __init__(self, opt, train_data, val_data, for_pb_maker=False):
        if for_pb_maker:
            self.make_pb_from_checkpoint(opt.checkpoint_path, opt.output_pb_path)
            return
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(opt.checkpoint_dir, 'summary', 'train'))
        self.val_summary_writer = tf.summary.FileWriter(os.path.join(opt.checkpoint_dir, 'summary', 'val'))

        self.train_data = train_data
        self.val_data = val_data
        self.build_graph(opt)

    def build_graph(self, opt):

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)

        config.gpu_options.allow_growth = True

        tf.Graph().as_default()

        if opt.use_gpu:
            tf.device('/gpu:0')
        else:
            tf.device('/cpu:0')

        self.sess = tf.Session(config=config)

        self.net_input = tf.placeholder(tf.float32, shape=(None, 64, 311, 1), name="input")
        self.ground_truth = tf.placeholder(tf.float32, shape=(None, 1, 311, 1), name="ground_truth")
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self.audios = tf.placeholder(tf.float32, shape=[None, None], name="audio")
        self.fs = tf.placeholder(tf.float32, shape=[None], name="sample_rate")
        self.cough_places = tf.placeholder(tf.string, shape=[None], name='cough_places')
        self.is_training = tf.placeholder(tf.bool, shape=[])

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        cough_detector_model = CoughDetectorModel()
        self.net_output = cough_detector_model.forward(self.net_input, self.is_training)

        with tf.name_scope("cross_entropy"):
            self.loss = tf.losses.sigmoid_cross_entropy(self.ground_truth, self.net_output, label_smoothing=0.2)
            # mask = tf.subtract(tf.cast(self.ground_truth.shape[2], tf.float32),
            #                    tf.reduce_sum(self.ground_truth, axis=2, keepdims=True)) / \
            #        tf.abs(tf.reduce_sum(self.ground_truth, axis=2, keepdims=True) - epsilon) * self.ground_truth
            #
            # self.loss = tf.reduce_mean(
            #     mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ground_truth, logits=self.net_output))

        self.train_loader = data.DataLoader(
            SpectogramLoader(dataset=self.train_data), batch_size=opt.batch_size, shuffle=True, num_workers=4)

        self.val_loader = data.DataLoader(
            SpectogramLoader(dataset=self.val_data), batch_size=opt.batch_size, shuffle=True, num_workers=4)

        tf.summary.scalar("cross_entropy_loss", self.loss)

        self.summary_right_cough_places = tf.summary.text("right_cough_places", self.cough_places)

        self.summary_audio = tf.summary.audio("audio", self.audios, self.fs[0])

        self.summary_input_spectogram = tf.summary.image('input_spectogram', self.net_input)

        self.summary_ground_truth = tf.summary.image('ground_truth', tf.cast(tf.concat([self.ground_truth, self.ground_truth, self.ground_truth,
                                                   self.ground_truth, self.ground_truth], axis=1)*255., tf.uint8))

        self.summary_predicted_values = tf.summary.image('predicted_values', tf.cast(tf.concat([self.net_output, self.net_output, self.net_output,
                                                       self.net_output, self.net_output], axis=1)*255., tf.uint8))

        tf.summary.scalar('learning_rate', self.learning_rate)

    def train(self, opt):

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                  global_step=self.global_step)

        self.sess.run(tf.global_variables_initializer())

        if opt.checkpoint_to_restore:
            self.restore(opt.checkpoint_to_restore)

        # summarize
        merged_summary = tf.summary.merge_all()
        val_summary = tf.summary.merge([self.summary_input_spectogram, self.summary_ground_truth,
                                        self.summary_predicted_values, self.summary_audio, self.summary_right_cough_places])

        self.train_summary_writer.add_graph(self.sess.graph)
        self.val_summary_writer.add_graph(self.sess.graph)

        num_iterations = self.train_loader.__len__()
        num_iterations_val = self.val_loader.__len__()

        print '-----------------------------------'
        print "Training data size is: " + str(num_iterations * opt.batch_size)
        print '-----------------------------------\n'

        epoch = 0
        while epoch < opt.num_epochs:
            print 'lr = ', opt.lr
            time.sleep(2)
            for iterations, (input_spectograms, ground_truths, audios, fs, cough_places) in enumerate(self.train_loader):

                input_spectograms = input_spectograms.numpy()
                ground_truths = ground_truths.numpy()
                audios = audios.numpy()
                fs = fs.numpy()
                cough_places = np.array(cough_places)

                start_time = time.time()

                if iterations % 40 == 0:
                    _, loss, summ, out = self.sess.run([self.train_step, self.loss, merged_summary, self.net_output],
                                  feed_dict={self.net_input: input_spectograms, self.ground_truth: ground_truths,
                                             self.learning_rate: opt.lr, self.audios: audios, self.fs: fs,
                                             self.cough_places: cough_places, self.is_training: True})

                    self.train_summary_writer.add_summary(summ, num_iterations * epoch + iterations)
                else:
                    _, loss, out = self.sess.run([self.train_step, self.loss, self.net_output],
                                                  feed_dict={self.net_input: input_spectograms,
                                                             self.ground_truth: ground_truths,
                                                             self.learning_rate: opt.lr, self.is_training: True})

                delta_time = time.time() - start_time

                print "epoch %s, iteration: %s / %s, batch time: %s, loss: %s" \
                      % (epoch, iterations, num_iterations, delta_time, loss)

                if opt.checkpoint_freq != 0 and (iterations+1) % opt.checkpoint_freq == 0:
                    saver = tf.train.Saver()
                    suffix = 'epoch=' + str(epoch+1)
                    saver.save(self.sess, os.path.join(os.path.join(opt.checkpoint_dir, suffix)+'/',
                                                  'fns_it=' + str(iterations) + '.ckpt'))
                    with open(os.path.join(opt.checkpoint_dir, suffix)+'/training_arguments.txt', 'w') as f:
                        f.write(str(opt))

                # print out
                # raw_input('ptc1')

            # Validation step
            val_total_loss = 0
            for iter, (input_spectograms, ground_truths, audios, fs, cough_places) in enumerate(self.val_loader):
                start_time = time.time()

                val_feed_dict = {
                    self.net_input: input_spectograms,
                    self.ground_truth: ground_truths,
                    self.audios: audios, self.fs: fs,
                    self.cough_places: cough_places,
                    self.is_training: False
                }

                output, loss, val_summ = self.sess.run([self.net_output, self.loss, val_summary], feed_dict=val_feed_dict)
                # print output
                # raw_input('ptc2')

                val_total_loss += loss

                delta_time = time.time() - start_time

                print "Validation step", "epoch %s, iteration: %s / %s, batch time: %s, loss: %s" \
                                         % (epoch, iter, num_iterations_val, delta_time, val_total_loss / num_iterations_val)

            summary = tf.Summary()
            summary.value.add(tag="cross_entropy_loss", simple_value=val_total_loss / num_iterations_val)

            self.val_summary_writer.add_summary(summary, num_iterations * (epoch+1))
            self.val_summary_writer.add_summary(val_summ, num_iterations * (epoch+1))

            epoch += 1
            opt.lr = opt.lr_decay * opt.lr

    def restore(self, checkpoint_path):
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        saver.restore(self.sess, ckpt)

    def make_pb_from_checkpoint(self, checkpoint_path, output_pb_path):

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)

        tf.Graph().as_default()
        tf.device('/cpu:0')

        self.sess = tf.Session(config=config)

        net_input = tf.placeholder(tf.float32, shape=(None, 64, 311, 1), name="input")

        cough_detector_model = CoughDetectorModel()
        net_output = cough_detector_model.forward(net_input, False)

        self.restore(checkpoint_path)

        for node in self.sess.graph.as_graph_def().node:
            print node.name
            node.device = ""
            # print node
        variable_graph_def = self.sess.graph.as_graph_def()
        optimized_net = tf.graph_util.convert_variables_to_constants(self.sess, variable_graph_def, ['output'])
        path, name = os.path.split(output_pb_path)
        tf.train.write_graph(optimized_net, path, name, False)
