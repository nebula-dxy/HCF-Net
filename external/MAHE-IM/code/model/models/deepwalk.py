# -*- coding:utf-8 -*-

"""




Reference:

    [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.(http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)



"""
from ..walker import RandomWalker
from gensim.models import Word2Vec


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(
            graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=100, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


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

import os
import networkx as nx
if __name__ == "__main__":
    root_dir = os.getcwd() + '/'  # 获取当前根目录
    data_dir = root_dir + 'DBLP_data_mining' + '/'  # 数据存取的路径
    output_dir = root_dir + 'output' + '/'  # 存放输出的路径
    a_a = []
    p_a = []
    p_c = []
    a_c = []
    edge = []
    # ff = open(data_dir + 'author_author.txt', 'w')
    with open(data_dir + 'DBLP_data_mining.txt') as f:
        for line in f:
            ss = line.strip().split(' ')
            if len(ss) != 2:
                print('wrong edges!')
            else:
                if (str(ss[0])[0] == 'a') and (str(ss[1])[0] == 'a'):
                    edge.append(line)
                    # ff.write(line)
                    # ff.write(ss[1] + ' ' + ss[0] + '\n')
                elif (str(ss[0])[0] == 'p') and (str(ss[1])[0] == 'a'):
                    edge.append(line)
                elif (str(ss[0])[0] == 'p') and (str(ss[1])[0] == 'c'):
                    edge.append(line)
                elif (str(ss[0])[0] == 'a') and (str(ss[1])[0] == 'c'):
                    edge.append(line)
                elif (str(ss[0])[0] != 'a' or 'p' or 'c') or (str(ss[1])[0] != 'a' or 'p' or 'c'):
                    print('wrong the type of nodes!')
                    exit()
    G = nx.Graph()
    G.add_edges_from(edge)
    model = DeepWalk(G, walk_length=10, num_walks=5, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()
    writeData(data_dir + 'DBLP_DM_DeepWalk.txt', embeddings)