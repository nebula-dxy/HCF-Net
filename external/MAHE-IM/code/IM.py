# -*- coding:utf-8 -*-

"""
MAHE-IM: the algorithm of this paper
(In addition, we provide four other IM algorithms that apply common network embedding methods)
DeepWalk-IM: the IM algorithm of applying DeepWalk embedding method
node2vec-IM: the IM algorithm of applying node2vec embedding method
LINE-IM: the IM algorithm of applying LINE embedding method
SDNE-IM: the IM algorithm of applying SDNE embedding method
"""
from model import DeepWalk, LINE, Node2Vec, SDNE, MetaPathGenerator, metapath2vec
import networkx as nx
import gensim
import os
from seed_selection import select_seed
import tensorflow as tf
import argparse
from relevant_vector import metapath_rele


def IM(method, edge, a_a, p_a, p_c, a_c):
    if method == 'DeepWalk-IM':
        G = nx.Graph()
        G.add_edges_from(edge)
        model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
        model.train(window_size=5, iter=3)
        embeddings = model.get_embeddings()
        writeData(output_dir + 'DBLP_data_mining_DeepWalk.txt', embeddings)
        vec = relevant_vector(output_dir + 'DBLP_data_mining_DeepWalk.txt')
        s = select_seed(vec)
        f = open(output_dir + 'DBLP_data_mining_DeepWalk-IM_seed.txt', 'w')
        f.write(s)
        # return s

    elif method == 'LINE-IM':
        G = nx.Graph()
        G.add_edges_from(edge)
        model = LINE(G, embedding_size=128, order='second')
        model.train(batch_size=1024, epochs=10, verbose=2)
        embeddings = model.get_embeddings()
        writeData(output_dir + 'DBLP_data_mining_LINE.txt', embeddings)
        vec = relevant_vector(output_dir + 'DBLP_data_mining_LINE.txt')
        s = select_seed(vec)
        f = open(output_dir + 'DBLP_data_mining_LINE-IM_seed.txt', 'w')
        f.write(s)
        # return s

    elif method == 'node2vec-IM':
        G = nx.DiGraph()
        G.add_edges_from(edge)
        model = Node2Vec(G, walk_length=40, num_walks=10,
                         p=0.75, q=1.25, workers=1, use_rejection_sampling=0)
        model.train(window_size=5, iter=3)
        embeddings = model.get_embeddings()
        writeData(output_dir + 'DBLP_data_mining_node2vec.txt', embeddings)
        vec = relevant_vector(output_dir + 'DBLP_data_mining_node2vec.txt')
        s = select_seed(vec)
        f = open(output_dir + 'DBLP_data_mining_node2vec-IM_seed.txt', 'w')
        f.write(s)
        # return s

    elif method == 'SDNE-IM':
        G = nx.Graph()
        G.add_edges_from(edge)
        model = SDNE(G, hidden_size=[256, 128], )
        model.train(batch_size=3000, epochs=10, verbose=2)
        embeddings = model.get_embeddings()
        writeData(output_dir + 'DBLP_data_mining_SDNE.txt', embeddings)
        vec = relevant_vector(output_dir + 'DBLP_data_mining_SDNE.txt')
        s = select_seed(vec)
        f = open(output_dir + 'DBLP_data_mining_SDNE-IM_seed.txt', 'w')
        f.write(s)
        # return s

    elif method == 'MAHE-IM':
        mpg = MetaPathGenerator()
        mpg.read_data(a_a, p_a, p_c, a_c)
        outfilename23 = output_dir + 'apa.txt'
        outfilename22 = output_dir + 'aca.txt'
        outfilename21 = output_dir + 'aaa.txt'
        mpg.generate_random_aa(outfilename21, 2, 2)
        mpg.generate_random_aca(outfilename22, 2, 1)
        mpg.generate_random_apa(outfilename23, 2, 1)
        outfilename31 = output_dir + 'aaaa.txt'
        outfilename32 = output_dir + 'acaa.txt'
        outfilename33 = output_dir + 'apaa.txt'
        outfilename34 = output_dir + 'apca.txt'
        mpg.generate_random_aa(outfilename31, 2, 3)
        mpg.generate_random_acaa(outfilename32, 2, 1)
        mpg.generate_random_apaa(outfilename33, 2, 1)
        mpg.generate_random_apca(outfilename34, 2, 1)
        outfilename41 = output_dir + 'aaaaa.txt'
        outfilename42 = output_dir + 'apaca.txt'
        outfilename43 = output_dir + 'apapa.txt'
        outfilename44 = output_dir + 'acaca.txt'
        outfilename45 = output_dir + 'apcpa.txt'
        mpg.generate_random_aa(outfilename41, 2, 4)
        mpg.generate_random_apaca(outfilename42, 2, 1)
        mpg.generate_random_apa(outfilename43, 2, 2)
        mpg.generate_random_aca(outfilename44, 2, 2)
        mpg.generate_random_apcpa(outfilename45, 2, 1)
        start_metapath2vec(output_dir + 'aaa.txt', output_dir + 'aaa_embedding.txt', 'a')
        start_metapath2vec(output_dir + 'aca.txt', output_dir + 'aca_embedding.txt', 'ac')
        start_metapath2vec(output_dir + 'apa.txt', output_dir + 'apa_embedding.txt', 'ap')
        start_metapath2vec(output_dir + 'aaaa.txt', output_dir + 'aaaa_embedding.txt', 'a')
        start_metapath2vec(output_dir + 'acaa.txt', output_dir + 'acaa_embedding.txt', 'ac')
        start_metapath2vec(output_dir + 'apaa.txt', output_dir + 'apaa_embedding.txt', 'ap')
        start_metapath2vec(output_dir + 'apca.txt', output_dir + 'apca_embedding.txt', 'apc')
        start_metapath2vec(output_dir + 'aaaaa.txt', output_dir + 'aaaaa_embedding.txt', 'a')
        start_metapath2vec(output_dir + 'apaca.txt', output_dir + 'apaca_embedding.txt', 'apc')
        start_metapath2vec(output_dir + 'apapa.txt', output_dir + 'apapa_embedding.txt', 'ap')
        start_metapath2vec(output_dir + 'acaca.txt', output_dir + 'acaca_embedding.txt', 'ac')
        start_metapath2vec(output_dir + 'apcpa.txt', output_dir + 'apcpa_embedding.txt', 'apc')
        relevancy = 0.31
        vec = metapath_rele(relevancy)
        s = select_seed(vec)
        f = open(output_dir + 'MAHE-IM_seed.txt', 'w')
        f.write(s)
        # return s


def writeData(filename, embeddings):
    '''
    Write data to filename.

    Args:
        filename: The output file name.
        embeddings: The embedding matrix.
        node2id: Same definition as in getData().
    '''
    number = 0
    with open(filename, 'w') as f:
        for i, v in embeddings.items():
            if str(i)[0] == 'a':
                number = number + 1
                f.write(str(i) + ' ' + ' '.join(str(i) for i in v) + '\n')

    e = len(v)
    with open(filename, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(str(number) + ' ' + str(e) + '\n' + content)


def start_metapath2vec(filename, outfilename, type):
    parser = argparse.ArgumentParser(description='Metapath2Vec')

    parser.add_argument('-file', dest='filename', default=filename,
                        help='The random walks filename')
    parser.add_argument('-embed_dim', dest='embed_dim', default=128, type=int, help='The length of latent embedding')
    parser.add_argument('-n', dest='neighbour_size', default=5, type=int, help='The neighbourhood size k')
    parser.add_argument('-epoch', dest='epoch', default=10, type=int,
                        help='Num of iterations for heterogeneous skipgram')
    parser.add_argument('-types', dest='typestr', default=type, type=str,
                        help='Specify types occurring in the data')
    parser.add_argument('-batch', dest='batch_size', default=4, type=int,
                        help='The number of the data used in each iter')
    parser.add_argument('-neg', dest='neg_size', default=3, type=int, help='The size of negative samples')
    parser.add_argument('-gpu', dest='gpu', default='0', help='Run the model on gpu')
    parser.add_argument('-l2', dest='l2', default=1e-3, type=float, help='L2 regularization scale (default 0.001)')
    parser.add_argument('-lr', dest='learning_rate', default=1e-2, type=float, help='Learning rate.')
    parser.add_argument('-outname', dest='outname', default=outfilename,
                        help='Name of the output file.')

    args = parser.parse_args()
    # set_gpu(args.gpu)

    tf.compat.v1.reset_default_graph()
    model = metapath2vec(args)

    config = tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=10)
    # config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess)


def relevant_vector(input):
    model = gensim.models.KeyedVectors.load_word2vec_format(input)
    keys = model.vocab.keys()
    relevant_vec = []
    for key in keys:
        relevant_vec.append(str(model.most_similar(key, topn=10)) + '\n')
    return relevant_vec


if __name__ == "__main__":

    root_dir = os.getcwd() + '/'  # 获取当前根目录
    data_dir = '/data' + '/'  # 数据存取的路径
    output_dir = '/output' + '/'  # 存放输出的路径
    a_a = []
    p_a = []
    p_c = []
    a_c = []
    edge = []
    with open(data_dir + 'DBLP_data_mining.txt') as f:
        for line in f:
            ss = line.strip().split(' ')
            if len(ss) != 2:
                print('wrong edges!')
            else:
                if (str(ss[0])[0] == 'a') and (str(ss[1])[0] == 'a'):
                    a_a.append(line)
                    aa = []
                    aa.append(ss[0])
                    aa.append(ss[1])
                    aaa = tuple(aa)
                    edge.append(aaa)
                    aa = []
                    aa.append(ss[1])
                    aa.append(ss[0])
                    aaa = tuple(aa)
                    edge.append(aaa)
                elif (str(ss[0])[0] == 'p') and (str(ss[1])[0] == 'a'):
                    p_a.append(line)
                    pa = []
                    pa.append(ss[0])
                    pa.append(ss[1])
                    paa = tuple(pa)
                    edge.append(paa)
                    pa = []
                    pa.append(ss[1])
                    pa.append(ss[0])
                    paa = tuple(pa)
                    edge.append(paa)
                elif (str(ss[0])[0] == 'p') and (str(ss[1])[0] == 'c'):
                    p_c.append(line)
                    pc = []
                    pc.append(ss[0])
                    pc.append(ss[1])
                    pcc = tuple(pc)
                    edge.append(pcc)
                    pc = []
                    pc.append(ss[1])
                    pc.append(ss[0])
                    pcc = tuple(pc)
                    edge.append(pcc)
                elif (str(ss[0])[0] == 'a') and (str(ss[1])[0] == 'c'):
                    a_c.append(line)
                    ac = []
                    ac.append(ss[0])
                    ac.append(ss[1])
                    acc = tuple(ac)
                    edge.append(acc)
                    ac = []
                    ac.append(ss[1])
                    ac.append(ss[0])
                    acc = tuple(ac)
                    edge.append(acc)
                elif (str(ss[0])[0] != 'a' or 'p' or 'c') or (str(ss[1])[0] != 'a' or 'p' or 'c'):
                    print('wrong the type of nodes!')
                    exit()
    # IM('MAHE-IM', edge, a_a, p_a, p_c, a_c)
    IM('DeepWalk-IM', edge, a_a, p_a, p_c, a_c)
    # IM('node2vec-IM', edge, a_a, p_a, p_c, a_c)
    # IM('LINE-IM', edge, a_a, p_a, p_c, a_c)
    # IM('SDNE-IM', edge, a_a, p_a, p_c, a_c)