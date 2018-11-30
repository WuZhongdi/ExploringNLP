#coding=utf-8

import pickle
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq


class RNN:

    def __init__(self):
        self.text = ""
        self.word_num = 0
        self.text_in_lines = ""
        self.vocab = ""
        self.vocab2int = {}
        self.int2vocab = {}
        self.punc_list = []
        self.punc_rep = []
        self.punc2rep = {}
        self.rep2punc = {}
        self.int_text = ""
        self.epochs = 50
        self.batch_size = 128
        self.RNN_size = 256
        self.embedding_size = 256
        self.seq_len = 32
        self.learning_rate = 0.01
        self.output_rate = 5
        self.layer_num = 3
        self.dropout_keep_rate = 0.8
        self.vocab_size = 0
        self.batch_num = 0
        self.num_per_batch = 0
        if not os.path.exists(r".\saves"):
            os.mkdir(r".\saves")
        self.saving_dir = r".\saves\save"

    def load_text(self, path):

        file_path = os.path.join(path)
        with open(file_path, 'r', encoding="utf-8") as f:
            self.text = f.read()
        return

    def set_parameters(self, **kwargs):

        self.word_num = 100000
        if "word_num" in kwargs:
            self.word_num = kwargs["word_num"]
        self.epochs = 200
        if "epochs" in kwargs:
            self.epochs = kwargs["epochs"]
        self.batch_size = 128
        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]
        self.RNN_size = 256
        if "RNN_size" in kwargs:
            self.RNN_size = kwargs["RNN_size"]
        self.embedding_size = 256
        if "embedding_size" in kwargs:
            self.embedding_size = kwargs["embedding_size"]
        self.seq_len = 32
        if "seq_len" in kwargs:
            self.seq_len = kwargs["seq_len"]
        self.learning_rate = 0.01
        if "learning_rate" in kwargs:
            self.learning_rate = kwargs["learning_rate"]
        self.output_rate = 5
        if "output_rate" in kwargs:
            self.output_rate = kwargs["output_rate"]
        self.layer_num = 3
        if "layer_num" in kwargs:
            self.layer_num = kwargs["layer_num"]
        self.dropout_keep_rate = 0.8
        if "dropout_keep_rate" in kwargs:
            self.dropout_keep_rate = kwargs["dropout_keep_rate"]
        self.saving_dir = r".\saves"
        if "saving_dir" in kwargs:
            self.saving_dir = kwargs["saving_dir"]
        self.saving_dir = os.path.join(self.saving_dir, "save")
        return

    def pre_process(self):

        print("Now pre processing text data ...")
        self.text = self.text[:self.word_num]
        self.text_in_lines = self.text.split("\n")
        self.text_in_lines = [line for line in self.text_in_lines if len(line) != 0]
        self.text_in_lines = [line.strip() for line in self.text_in_lines]
        pattern = re.compile(r"\[.*\]")
        self.text_in_lines = [pattern.sub("", line) for line in self.text_in_lines]
        pattern = re.compile(r"<.*>")
        self.text_in_lines = [pattern.sub("", line) for line in self.text_in_lines]
        pattern = re.compile(r"\\r")
        self.text_in_lines = [pattern.sub("",line) for line in self.text_in_lines]
        print("Now building dictionary ...")
        self.text = "".join(self.text_in_lines)
        self.vocab = set(self.text)
        self.vocab_size = len(self.vocab)
        self.vocab2int = {word: index for index, word in enumerate(self.vocab)}
        self.int2vocab = dict(enumerate(self.vocab))
        self.punc_list = ["。", "，", "“", "”", "：", "；", "！", "？", "（", "）"]
        self.punc_rep = ["J", "D", "S", "X", "M", "F", "G", "W", "Z", "Y"]
        for i in range(len(self.punc_list)):
            self.punc2rep[self.punc_list[i]] = self.punc_rep[i]
        for i in range(len(self.punc_rep)):
            self.rep2punc[self.punc_rep[i]] = self.punc_list[i]
        self.int_text = [self.vocab2int[word] for word in self.text]
        pickle.dump((self.int_text, self.vocab2int, self.int2vocab, self.punc2rep, self.rep2punc)
                    , open('pre_process_items.p', 'wb'))
        return

    def get_inputs(self):
        inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
        targets = tf.placeholder(tf.int32, [None, None], name="targets")
        lr = tf.placeholder(tf.float32, name="learning_rate")
        return inputs, targets, lr

    def get_init_cell(self, batch_size):
        cell = tf.contrib.rnn.BasicLSTMCell(self.RNN_size)
        drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_rate)
        cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(self.layer_num)])
        init_state = cell.zero_state(batch_size, tf.float32)
        init_state = tf.identity(init_state, name="init_state")
        return cell, init_state

    def get_embed(self, input_data):
        embedding = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size]
                                                    , stddev=0.1), dtype=tf.float32
                                , name="embedding")
        return tf.nn.embedding_lookup(embedding, input_data, name="after_embed")

    def build_rnn(self, cell, inputs):
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        final_state = tf.identity(final_state, name="final_state")
        return outputs, final_state

    def build_nn(self, cell, input_data):
        embed = self.get_embed(input_data)
        outputs, final_state = self.build_rnn(cell, embed)
        logits = tf.contrib.layers.fully_connected(outputs, self.vocab_size, activation_fn=None
                                                   , weights_initializer=tf.truncated_normal_initializer(stddev=0.1)
                                                   , biases_initializer=tf.zeros_initializer())
        return logits, final_state

    def get_batch(self):
        self.num_per_batch = self.batch_size * self.seq_len
        self.batch_num = len(self.int_text) // self.num_per_batch
        input_data = np.array(self.int_text[:self.batch_num*self.num_per_batch])
        target_data = np.array(self.int_text[1:self.batch_num*self.num_per_batch + 1])
        inputs = input_data.reshape(self.batch_size, -1)
        targets = target_data.reshape(self.batch_size, -1)
        inputs = np.split(inputs, self.batch_num, 1)
        targets = np.split(targets, self.batch_num, 1)
        batches = np.array(list(zip(inputs, targets)))
        batches[-1][-1][-1][-1] = batches[0][0][0][0]
        return batches

    def train(self):
        train_graph = tf.Graph()
        with train_graph.as_default():
            input_text, targets, lr = self.get_inputs()
            input_data_shape = tf.shape(input_text)
            cell, initial_state = self.get_init_cell(input_data_shape[0])
            logits, final_state = self.build_nn(cell, input_text)
            probs = tf.nn.softmax(logits, name="probs")
            cost = seq2seq.sequence_loss(
                logits,
                targets,
                tf.ones([input_data_shape[0], input_data_shape[1]]))
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
        print("Now training Model...")
        batches = self.get_batch()
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(self.epochs):
                state = sess.run(initial_state, {input_text: batches[0][0]})
                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        input_text: x,
                        targets: y,
                        initial_state: state,
                        lr: self.learning_rate
                    }
                    train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

                    if (epoch_i * len(batches) + batch_i) % self.output_rate == 0:
                        print("Epoch {:>3} Batch{:>4}/{}  train_loss = {:.3f}".format(
                            epoch_i,
                            batch_i,
                            len(batches),
                            train_loss
                        ))

            saver = tf.train.Saver()
            saver.save(sess, self.saving_dir)
            pickle.dump((self.seq_len, self.saving_dir), open('params.p', 'wb'))
            print("Model Training Done and Saved!")

        return


def work():
    luxun = RNN()
    luxun.load_text(r".\data\鲁迅全集.txt")
    luxun.set_parameters(word_num=200000, learning_rate=0.02, epochs=20, output_rate=10)
    luxun.pre_process()
    luxun.train()


def main():
    work()


if __name__ == '__main__':
    main()
