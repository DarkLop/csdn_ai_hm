#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 兼容性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""宋词转向量"""
"""
作业评价标准

    学员需要提交下述图片作为作业成果，该文件在embedding脚本运行完之后输出。

    图片中意义接近的词，如数字等(参考图中红圈标记)，距离比较近（一这个数字是个特例，离其他数字比较远）。-60分

    该文件中位置相近的字没有明确相似性的，不予及格。

    提供一个文档，说明自己对embedding的理解，代码的分析，以及对上述图片的结果分析和认识。-40分
"""
"""
要点提示

    全宋词资料不同于英文，不使用分词，这里直接将每个单字符作为一个word。

    全宋词全文共6010个不同的单字符，这里只取出现次数最多的前5000个单字符。

    后面的RNN训练部分，需要使用embedding的dictionary, reversed_dictionary,请使用json模块的save方法将这里生成的两个字典保存起来。
utils中也提供了一个字典的生成方法，RNN作业部分，如果不使用这个作业生成的embedding.npy文件作为model的embeding参数
（参考model的build方法中的embedding_file参数）的时候可以使用这个utils中提供的方法直接生成这两个字典文件。

    matplotlib中输出中文的时候会出现乱码，请自行搜索如何设置matplotlib使之可以输出中文。

    按照tensorflow官方代码中给出的设置，运行40W个step可以输出一个比较好的结果，四核CPU上两三个小时左右。

    对于文本的处理，可以搜到很多不同的处理方式，大部分文本处理都要删掉所有的空格，换行，标点符号等等。
这里的训练可以不对文本做任何处理。

    本作业中，涉及大量中文的处理，因为python2本身对UTF-8支持不好，另外官方对python2的支持已经快要结束了，推荐本项目使用python3进行。

## #word2vec中，可以使用如下代码来保存最终生成的embeding
## np.save('embedding.npy', final_embeddings)
"""

# 导入模块
import argparse
import io
import sys
import collections
import math
import os
import random
import json

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

import utils

# 设置 print 编码为 utf-8，防止打印出错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding = "utf-8")

# 1 读取数据
def read_data(filename = "QuanSongCi.txt", punctuation = 0):
    if punctuation == 0:
        return utils.read_data(filename)
    else:
        ### 处理标点（逗号、顿号与句号）与回车
        with open(filename, mode = "r", encoding = "utf-8") as f:
            lines = tf.compat.as_str(f.read()).replace("，", "").replace("。", "").replace("、", "").split()
            data = list(word for line in lines for word in line)
        return data

# 2 建立字典
def build_dataset(words, n_words):
    return utils.build_dataset(words, n_words)

# 数据下标
data_index = 0

# 3 生成训练batch
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape = (batch_size), dtype = np.int32)
    labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
    
    ## 选取的个数
    span = 2 * skip_window + 1
    
    ## 临时缓冲区，双端队列
    buffer = collections.deque(maxlen = span)

    if data_index + span > len(data):
        data_index = 0

    ## 添加 span 个数据追加到buffer
    buffer.extend(data[data_index : data_index + span])
    data_index += span

    ## 循环
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips) # 随机获取 num_skips 字

        ### 迭代所有使用的字
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]

        ### 如果已经超出数据长度
        if data_index == len(data):
            buffer.clear()
            buffer.extend(data[:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    data_index = (data_index + len(data) - span) % len(data)

    return batch, labels

# 4 获取模型
def create_model( \
    vocabulary, \
    vocabulary_size, \
    valid_examples, \
    batch_size = 128, \
    embedding_size = 128, \
    skip_window = 1, \
    num_skips = 2, \
    num_sampled = 64):

    graph = tf.Graph()

    with graph.as_default():
        ### 输入数据
        train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
        train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

        ### 使用CPU计算
        with tf.device("/cpu:0"):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            #### 构造 NCE loss 变量
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        ### 计算 NCE loss
        loss = tf.reduce_mean( \
            tf.nn.nce_loss(weights = nce_weights, \
                        biases = nce_biases, \
                        labels = train_labels, \
                        inputs = embed, \
                        num_sampled = num_sampled, \
                        num_classes = vocabulary_size))

        ### 构造 SGD ，学习率 1.0
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        ### 计算 minibatch 和 embeddings 的余弦相似度
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

        ### varible 初始化器
        init = tf.global_variables_initializer()

    return graph, train_inputs, train_labels, loss, optimizer, normalized_embeddings, similarity, init

# 5 训练
def train( \
    data, \
    vocabulary, \
    vocabulary_size, \
    reverse_dictionary, \
    num_steps = 400001, \
    batch_size = 128, \
    embedding_size = 128, \
    skip_window = 1, \
    num_skips = 2, \
    num_sampled = 64, \
    valid_size = 16, \
    valid_window = 100):

    valid_examples = np.random.choice(valid_window, valid_size, replace = False)

    graph, train_inputs, train_labels, loss, optimizer, normalized_embeddings, similarity, init = create_model(
        vocabulary, \
        vocabulary_size, \
        valid_examples, \
        batch_size = batch_size, \
        embedding_size = embedding_size, \
        skip_window = skip_window, \
        num_skips = num_skips, \
        num_sampled = num_sampled)

    with tf.Session(graph = graph) as session:
        ### 初始化variable
        init.run()
        print("Initialized")

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            #### 启动SGD与loss
            _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            #### Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                    log_str = "Nearest to %s:" % valid_word

                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)

                    print(log_str)

        final_embeddings = normalized_embeddings.eval()
    
    return final_embeddings

# 6 嵌入可视化
def plot_with_labels(low_dim_embs, labels, filename, fontproperties = None):

    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize = (18, 18)) # in inches

    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate( \
            label, \
            xy = (x, y), \
            xytext = (5, 2), \
            textcoords = "offset points", \
            ha = "right", \
            va = "bottom", \
            fontproperties = fontproperties)
    plt.savefig(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type = str, default = "./QuanSongCi.txt")
    parser.add_argument("--train", type = int, default = 1)
    parser.add_argument("--output_dir", type = str, default = "./output")
    parser.add_argument("--plt", type = int, default = 1)
    parser.add_argument("--font_path", type = str, default = "C:/Windows/Fonts/simfang.ttf")
    parser.add_argument("--num_steps", type = int, default = 400001)
    parser.add_argument("--punctuation", type = int, default = 0)
    FLAGS, unparsed = parser.parse_known_args()

    print(FLAGS)
    print("unparsed args" + str(unparsed))

    reverse_dictionary = None
    final_embeddings = None

    if FLAGS.train != 0:
        # 1 读取的数据
        vocabulary = read_data(filename = FLAGS.file_path, punctuation = FLAGS.punctuation)

        # 取出频率高的字
        vocabulary_size = 5000

        # 2 数据变量
        data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

        # 将字典保存成json
        with open(os.path.join(FLAGS.output_dir, "dictionary.json"), mode = "w", encoding = "utf-8") as f:
            f.write(json.dumps(dictionary))
        with open(os.path.join(FLAGS.output_dir, "reverse_dictionary.json"), mode = "w", encoding = "utf-8") as f:
            f.write(json.dumps(reverse_dictionary))

        # 3 生成训练数据 batch
        batch, labels = generate_batch(data, batch_size = 8, num_skips = 2, skip_window = 1)

        for i in range(8):
            print(batch[i], reverse_dictionary[batch[i]], "->", labels[i, 0], reverse_dictionary[labels[i, 0]])

        # 训练 step
        num_steps = FLAGS.num_steps

        # 4 模型 5 训练
        final_embeddings = train(data, vocabulary, vocabulary_size, reverse_dictionary, num_steps = num_steps)

        # 保存embeddings
        np.save(os.path.join(FLAGS.output_dir, "embedding.npy"), final_embeddings)
    elif FLAGS.output_dir is not None:
        try:
            with open(os.path.join(FLAGS.output_dir, "dictionary.json"), mode = "r", encoding = "utf-8") as f:
                dictionary = json.loads(f.read())
                reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
            final_embeddings = np.load(os.path.join(FLAGS.output_dir, "embedding.npy"))
        except FileNotFoundError as ex:
            print("embedding.npy or dictionary.json is not found.")
            print("Path: " + FLAGS.output_dir)
            print(ex)

    if FLAGS.plt != 0 and reverse_dictionary is not None and final_embeddings is not None:
        # 6 嵌入可视化
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm

            tsne = TSNE(perplexity = 30, n_components = 2, init = "pca", n_iter = 5000, method = "exact")
            plot_only = 500
            low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
            labels = [reverse_dictionary[i] for i in xrange(plot_only)]
            fontproperties = fm.FontProperties(fname = FLAGS.font_path)
            plot_with_labels(low_dim_embs, labels, os.path.join(FLAGS.output_dir, "hm_tsne.png"), fontproperties = fontproperties)

        except ImportError as ex:
            print("Please install sklearn, matplotlib, and scipy to show embeddings.")
            print(ex)


