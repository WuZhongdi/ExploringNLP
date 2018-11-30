import tensorflow as tf
import numpy as np
import pickle


class RNNw:

    def __init__(self):
        self.vocab2int = None
        self.int2vocab = None
        self.punc2rep = None
        self.rep2punc = None
        self.seq_len = None
        self.loading_dir = None

    def preparing(self):
        _, self.vocab2int, self.int2vocab, self.punc2rep, self.rep2punc = pickle.load(open('pre_process_items.p', mode='rb'))
        self.seq_len, self.loading_dir = pickle.load(open('params.p', mode='rb'))

    def get_tensors(self, loaded_graph):

        inputs = loaded_graph.get_tensor_by_name("inputs:0")
        initial_state = loaded_graph.get_tensor_by_name("init_state:0")
        final_state = loaded_graph.get_tensor_by_name("final_state:0")
        probs = loaded_graph.get_tensor_by_name("probs:0")
        return inputs, initial_state, final_state, probs

    def choose_word(self, p):
        p = list(p)
        choose_p = 0
        pre_word = ""
        for i in range(len(p)):
            if p[i] > choose_p:
                choose_p = p[i]
                pre_word = self.int2vocab[i]
        return pre_word

    def get_novel(self, start_word, **kwargs):
        novel_len = 500
        if "novel_len" in kwargs:
            novel_len = kwargs["novel_len"]
        end_at_punc = True
        if "end_at_punc" in kwargs:
            end_at_punc = kwargs["end_at_punc"]
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(self.loading_dir + ".meta")
            loader.restore(sess, self.loading_dir)
            input_text, initial_state, final_state, probs = self.get_tensors(loaded_graph)
            gen_sentence = [start_word]
            prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
            for article_n in range(novel_len):
                now_input = [[self.vocab2int[word] for word in gen_sentence[-self.seq_len:]]]
                now_seq_len = len(now_input[0])

                p, prev_state = sess.run(
                    [probs, final_state],
                    {input_text: now_input, initial_state: prev_state}
                )
                pred_word = self.choose_word(p[0][now_seq_len-1])
                if end_at_punc and pred_word == "。":
                    break
                gen_sentence.append(pred_word)
        novel = "".join(gen_sentence)
        new_novel = ""
        for word in novel:
            if word in self.rep2punc:
                new_novel = new_novel + self.rep2punc[word]
            else:
                new_novel = new_novel + word
        novel = new_novel
        if end_at_punc:
            novel = novel + "。"
        print(novel)


def work():
    luxun = RNNw()
    luxun.preparing()
    luxun.get_novel("夫")


def main():
    work()


if __name__ == '__main__':
    main()
