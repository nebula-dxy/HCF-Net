# -*- coding:utf-8 -*-

"""
Generate comprehensive relevant nodes of each node based on metapaths.
"""

import gensim
import os

data_dir = '/data' + '/'  # 数据存取的路径
output_dir = '/output' + '/'  # 存放输出的路径

def metapath_rele(relevancy):
    k = []
    aaa = {}
    model2 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'aaa_embedding.txt')
    key2 = model2.wv.vocab.keys()
    k.append(key2)
    for key in key2:
        t = model2.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            aaa.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open('output_dir + aaa_relevant_vector.txt', 'w')
    # for i in aaa:
    #     f.write(str(i) + ';' + str(aaa[i]) + '\n')
    print('aaa')

    aca = {}
    model3 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'aca_embedding.txt')
    key3 = model3.wv.vocab.keys()
    k.append(key3)
    for key in key3:
        t = model3.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            aca.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'aca_relevant_vector.txt', 'w')
    # for i in aca:
    #     f.write(str(i) + ';' + str(aca[i]) + '\n')
    print('aca')

    apa = {}
    model4 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'apa_embedding.txt')
    key4 = model4.wv.vocab.keys()
    k.append(key4)
    for key in key4:
        t = model4.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            apa.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'apa_relevant_vector.txt', 'w')
    # for i in apa:
    #     f.write(str(i) + ';' + str(apa[i]) + '\n')
    print('apa')

    aaaa = {}
    model5 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'aaaa_embedding.txt')
    key5 = model5.wv.vocab.keys()
    k.append(key5)
    for key in key5:
        t = model5.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            aaaa.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'aaaa_relevant_vector.txt', 'w')
    # for i in aaaa:
    #     f.write(str(i) + ';' + str(aaaa[i]) + '\n')
    print('aaaa')

    acaa = {}
    model6 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'acaa_embedding.txt')
    key6 = model6.wv.vocab.keys()
    k.append(key6)
    for key in key6:
        t = model6.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            acaa.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'acaa_relevant_vector.txt', 'w')
    # for i in acaa:
    #     f.write(str(i) + ';' + str(acaa[i]) + '\n')
    print('acaa')

    apaa = {}
    model7 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'apaa_embedding.txt')
    key7 = model7.wv.vocab.keys()
    k.append(key7)
    for key in key7:
        t = model7.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            apaa.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'apaa_relevant_vector.txt', 'w')
    # for i in apaa:
    #     f.write(str(i) + ';' + str(apaa[i]) + '\n')
    print('apaa')

    apca = {}
    model8 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'apca_embedding.txt')
    key8 = model8.wv.vocab.keys()
    k.append(key8)
    for key in key8:
        t = model8.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            apca.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'apca_relevant_vector.txt', 'w')
    # for i in apca:
    #     f.write(str(i) + ';' + str(apca[i]) + '\n')
    print('apca')

    aaaaa = {}
    model9 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'aaaaa_embedding.txt')
    key9 = model9.wv.vocab.keys()
    k.append(key9)
    for key in key9:
        t = model9.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            aaaaa.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'aaaaa_relevant_vector.txt', 'w')
    # for i in aaaaa:
    #     f.write(str(i) + ';' + str(aaaaa[i]) + '\n')
    print('aaaaa')

    acaca = {}
    model10 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'acaca_embedding.txt')
    key10 = model10.wv.vocab.keys()
    k.append(key10)
    for key in key10:
        t = model10.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            acaca.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'acaca_relevant_vector.txt', 'w')
    # for i in acaca:
    #     f.write(str(i) + ';' + str(acaca[i]) + '\n')
    print('acaca')

    apapa = {}
    model11 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'apapa_embedding.txt')
    key11 = model11.wv.vocab.keys()
    k.append(key11)
    for key in key11:
        t = model11.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            apapa.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'apapa_relevant_vector.txt', 'w')
    # for i in apapa:
    #     f.write(str(i) + ';' + str(apapa[i]) + '\n')
    print('apapa')

    apaca = {}
    model12 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'apaca_embedding.txt')
    key12 = model12.wv.vocab.keys()
    k.append(key12)
    for key in key12:
        t = model12.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            apaca.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'apaca_relevant_vector.txt', 'w')
    # for i in apaca:
    #     f.write(str(i) + ';' + str(apaca[i]) + '\n')
    print('apaca')

    apcpa = {}
    model13 = gensim.models.KeyedVectors.load_word2vec_format(output_dir + 'apcpa_embedding.txt')
    key13 = model13.wv.vocab.keys()
    k.append(key13)
    for key in key13:
        t = model13.wv.most_similar(key, topn=10)
        for tt in t:
            deal = str(tt).strip().split(',')
            d0 = deal[0].strip().replace('(', '').strip("'")
            d1 = float(deal[1].strip().replace(')', '').strip("'"))
            apcpa.setdefault(key, []).append(str(d0) + ':' + str(d1))
    # f = open(output_dir + 'apcpa_relevant_vector.txt', 'w')
    # for i in apcpa:
    #     f.write(str(i) + ';' + str(apcpa[i]) + '\n')
    print('apcpa')

    kkk = key2
    for kk in k:
        if len(kkk) < len(kk):
            kkk = kk

    p = 0.5
    count = {}
    vec = []
    for key in kkk:
        if key in aaa:
            a2 = aaa[key]
            for aaa2 in a2:
                l = str(aaa2).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                count[l0] = (p * l1) / 3

        if key in aca:
            a3 = aca[key]
            for aaa3 in a3:
                l = str(aaa3).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * l1) / 3
                else:
                    count[l0] = count[l0] + (p * l1) / 3

        if key in apa:
            a4 = apa[key]
            for aaa4 in a4:
                l = str(aaa4).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * l1) / 3
                else:
                    count[l0] = count[l0] + (p * l1) / 3

        if key in aaaa:
            a5 = aaaa[key]
            for aaa5 in a5:
                l = str(aaa5).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * p * l1) / 4
                else:
                    count[l0] = count[l0] + (p * p * l1) / 4

        if key in acaa:
            a6 = acaa[key]
            for aaa6 in a6:
                l = str(aaa6).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * p * l1) / 4
                else:
                    count[l0] = count[l0] + (p * p * l1) / 4

        if key in apaa:
            a7 = apaa[key]
            for aaa7 in a7:
                l = str(aaa7).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * p * l1) / 4
                else:
                    count[l0] = count[l0] + (p * p * l1) / 4

        if key in apca:
            a8 = apca[key]
            for aaa8 in a8:
                l = str(aaa8).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * p * l1) / 4
                else:
                    count[l0] = count[l0] + (p * p * l1) / 4

        if key in aaaaa:
            a9 = aaaaa[key]
            for aaa9 in a9:
                l = str(aaa9).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * p * p * l1) / 5
                else:
                    count[l0] = count[l0] + (p * p * p * l1) / 5

        if key in acaca:
            a10 = acaca[key]
            for aaa10 in a10:
                l = str(aaa10).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * p * p * l1) / 5
                else:
                    count[l0] = count[l0] + (p * p * p * l1) / 5

        if key in apapa:
            a11 = apapa[key]
            for aaa11 in a11:
                l = str(aaa11).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * p * p * l1) / 5
                else:
                    count[l0] = count[l0] + (p * p * p * l1) / 5

        if key in apaca:
            a12 = apaca[key]
            for aaa12 in a12:
                l = str(aaa12).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * p * p * l1) / 5
                else:
                    count[l0] = count[l0] + (p * p * p * l1) / 5

        if key in apcpa:
            a13 = apcpa[key]
            for aaa13 in a13:
                l = str(aaa13).strip().split(':')
                l0 = str(l[0])
                l1 = float(l[1])
                if l0 not in count:
                    count[l0] = (p * p * p * l1) / 5
                else:
                    count[l0] = count[l0] + (p * p * p * l1) / 5

        countt = sorted(count.items(), key=lambda item: item[1], reverse=True)
        cc = []
        for cou in countt:
            co = str(cou).strip().replace('(', '').replace(')', '').split(',')
            if float(co[1]) > relevancy:
                cc.append(cou)
        if len(cc) > 10:
            vec.append(cc[:10])
        else:
            vec.append(cc)
    return vec