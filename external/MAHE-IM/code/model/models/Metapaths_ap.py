"""
Generate metapaths.
"""
import random
import os

class MetaPathGenerator:
    def __init__(self):
        self.author_author = dict()
        # self.conf_author = dict()
        # self.author_conf = dict()
        self.paper_author = dict()
        self.author_paper = dict()
        # self.conf_paper = dict()
        # self.paper_conf = dict()

    def read_data(self, a_a, p_a):
        for line in a_a:
            toks = line.strip().split(" ")
            if len(toks) == 2:
                a0, a1 = toks[0], toks[1]
                if a0 not in self.author_author:
                    self.author_author[a0] = []
                self.author_author[a0].append(a1)
                if a1 not in self.author_author:
                    self.author_author[a1] = []
                self.author_author[a1].append(a0)
        for line in p_a:
            toks = line.strip().split(" ")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_author:
                    self.paper_author[p] = []
                self.paper_author[p].append(a)
                if a not in self.author_paper:
                    self.author_paper[a] = []
                self.author_paper[a].append(p)
        # for line in p_c:
        #     toks = line.strip().split(" ")
        #     if len(toks) == 2:
        #         p, c = toks[0], toks[1]
        #         self.paper_conf[p] = c
        #         if c not in self.conf_paper:
        #             self.conf_paper[c] = []
        #         self.conf_paper[c].append(p)
        # for line in a_c:
        #     toks = line.strip().split(" ")
        #     if len(toks) == 2:
        #         a, c = toks[0], toks[1]
        #         if a not in self.author_conf:
        #             self.author_conf[a] = []
        #         self.author_conf[a].append(c)
        #         if c not in self.conf_author:
        #             self.conf_author[c] = []
        #         self.conf_author[c].append(a)

    def generate_random_aa(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        k = 1
        for author in self.author_author:
            author0 = author
            for j in range(0, numwalks): #wnum walks
                outline = str(author0)
                author = author0
                for i in range(0, walklength):
                    authorr = self.author_author[author]
                    numa = len(authorr)
                    if numa == 0:
                        outline = '#'
                        continue
                    aid = random.randrange(numa)
                    author = authorr[aid]
                    outline += " " + str(author)
                o = outline.strip().split(' ')
                if walklength == 2:
                    if len(o) == 3:
                        outfile.write(outline + "\n")
                        # print(k)
                        k = k + 1
                if walklength == 3:
                    if len(o) == 4:
                        outfile.write(outline + "\n")
                        # print(k)
                        k = k + 1
                if walklength == 4:
                    if len(o) == 5:
                        outfile.write(outline + "\n")
                        # print(k)
                        k = k + 1
        outfile.close()

    # def generate_random_aca(self, outfilename, numwalks, walklength):
    #     outfile = open(outfilename, 'w')
    #     k = 1
    #     for author in self.author_conf:
    #         author0 = author
    #         for j in range(0, numwalks):  # num walks
    #             outline = str(author0)
    #             author = author0
    #             for i in range(0, walklength):
    #                 confs = self.author_conf[author]
    #                 numc = len(confs)
    #                 if numc == 0:
    #                     outline = '#'
    #                     continue
    #                 confid = random.randrange(numc)
    #                 conf = confs[confid]
    #                 outline += " " + str(conf)
    #
    #                 authors = self.conf_author[conf]
    #                 numa = len(authors)
    #                 if numa == 0:
    #                     outline = '#'
    #                     continue
    #                 authorid = random.randrange(numa)
    #                 author = authors[authorid]
    #                 outline += " " + str(author)
    #             o = outline.strip().split(' ')
    #             if walklength == 1:
    #                 if len(o) == 3:
    #                     outfile.write(outline + "\n")
    #                     # print(k)
    #                     k = k + 1
    #             if walklength == 2:
    #                 if len(o) == 5:
    #                     outfile.write(outline + "\n")
    #                     # print(k)
    #                     k = k + 1
    #     outfile.close()

    def generate_random_apa(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        k = 1
        for author in self.author_paper:
            author0 = author
            for j in range(0, numwalks):  # wnum walks
                outline = str(author0)
                author = author0
                for i in range(0, walklength):
                    papers = self.author_paper[author]
                    nump = len(papers)
                    if nump == 0:
                        outline = '#'
                        continue
                    paperid = random.randrange(nump)
                    paper = papers[paperid]
                    outline += " " + str(paper)

                    authors = self.paper_author[paper]
                    numa = len(authors)
                    if numa == 0:
                        outline = '#'
                        continue
                    authorid = random.randrange(numa)
                    author = authors[authorid]
                    outline += " " + str(author)
                o = outline.strip().split(' ')
                if walklength == 1:
                    if len(o) == 3:
                        outfile.write(outline + "\n")
                        # print(k)
                        k = k + 1
                if walklength == 2:
                    if len(o) == 5:
                        outfile.write(outline + "\n")
                        # print(k)
                        k = k + 1
        outfile.close()

    # def generate_random_acaa(self, outfilename, numwalks, walklength):
    #     outfile = open(outfilename, 'w')
    #     k = 1
    #     for author in self.author_conf:
    #         author0 = author
    #         for j in range(0, numwalks): #wnum walks
    #             outline = str(author0)
    #             for i in range(0, walklength):
    #                 confs = self.author_conf[author]
    #                 numc = len(confs)
    #                 if numc == 0:
    #                     outline = '#'
    #                     continue
    #                 confid = random.randrange(numc)
    #                 conf = confs[confid]
    #                 outline += " " + str(conf)
    #
    #                 authors = self.conf_author[conf]
    #                 numa = len(authors)
    #                 if numa == 0:
    #                     outline = '#'
    #                     continue
    #                 authorid = random.randrange(numa)
    #                 author = authors[authorid]
    #                 outline += " " + str(author)
    #
    #                 if (author in self.author_author) and (len(self.author_author[author]) != 0):
    #                     authors2 = self.author_author[author]
    #                     numa2 = len(authors2)
    #                     if numa2 == 0:
    #                         outline = '#'
    #                         continue
    #                     authorid2 = random.randrange(numa2)
    #                     author2 = authors2[authorid2]
    #                     outline += " " + str(author2)
    #             o = outline.strip().split(' ')
    #             if len(o) == 4:
    #                 outfile.write(outline + "\n")
    #                 # print(k)
    #                 k = k + 1
    #     outfile.close()

    def generate_random_apaa(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        k = 1
        for author in self.author_paper:
            author0 = author
            for j in range(0, numwalks): #wnum walks
                outline = str(author0)
                for i in range(0, walklength):
                    papers = self.author_paper[author]
                    nump = len(papers)
                    if nump == 0:
                        outline = '#'
                        continue
                    paperid = random.randrange(nump)
                    paper = papers[paperid]
                    outline += " " + str(paper)

                    authors = self.paper_author[paper]
                    numa = len(authors)
                    if numa == 0:
                        outline = '#'
                        continue
                    authorid = random.randrange(numa)
                    author = authors[authorid]
                    outline += " " + str(author)

                    if (author in self.author_author) and (len(self.author_author[author]) != 0):
                        authors2 = self.author_author[author]
                        numa2 = len(authors2)
                        if numa2 == 0:
                            outline = '#'
                            continue
                        authorid2 = random.randrange(numa2)
                        author2 = authors2[authorid2]
                        outline += " " + str(author2)
                o = outline.strip().split(' ')
                if len(o) == 4:
                    outfile.write(outline + "\n")
                    # print(k)
                    k = k + 1
        outfile.close()

    # def generate_random_apca(self, outfilename, numwalks, walklength):
    #     outfile = open(outfilename, 'w')
    #     k = 1
    #     for author in self.author_paper:
    #         author0 = author
    #         for j in range(0, numwalks): #wnum walks
    #             outline = str(author0)
    #             for i in range(0, walklength):
    #                 papers = self.author_paper[author]
    #                 nump = len(papers)
    #                 if nump == 0:
    #                     outline = '#'
    #                     continue
    #                 paperid = random.randrange(nump)
    #                 paper = papers[paperid]
    #                 outline += " " + str(paper)
    #
    #                 conf = self.paper_conf[paper]
    #                 outline += " " + str(conf)
    #
    #                 authors2 = self.conf_author[conf]
    #                 numa2 = len(authors2)
    #                 if numa2 == 0:
    #                     outline = '#'
    #                     continue
    #                 authorid2 = random.randrange(numa2)
    #                 author2 = authors2[authorid2]
    #                 outline += " " + str(author2)
    #             o = outline.strip().split(' ')
    #             if len(o) == 4:
    #                 outfile.write(outline + "\n")
    #                 # print(k)
    #                 k = k + 1
    #     outfile.close()
    #
    # def generate_random_aacaa(self, outfilename, numwalks, walklength):
    #     outfile = open(outfilename, 'w')
    #     k = 1
    #     for author in self.author_author:
    #         author0 = author
    #         for j in range(0, numwalks): #wnum walks
    #             outline = str(author0)
    #             for i in range(0, walklength):
    #                 authorss = self.author_author[author]
    #                 numaa = len(authorss)
    #                 if numaa == 0:
    #                     outline = '#'
    #                     continue
    #                 authoridd = random.randrange(numaa)
    #                 authorr = authorss[authoridd]
    #                 outline += " " + str(authorr)
    #
    #                 confs = self.author_conf[authorr]
    #                 numc = len(confs)
    #                 if numc == 0:
    #                     outline = '#'
    #                     continue
    #                 confid = random.randrange(numc)
    #                 conf = confs[confid]
    #                 outline += " " + str(conf)
    #
    #                 authors = self.conf_author[conf]
    #                 numa = len(authors)
    #                 if numa == 0:
    #                     outline = '#'
    #                     continue
    #                 authorid = random.randrange(numa)
    #                 author = authors[authorid]
    #                 outline += " " + str(author)
    #
    #                 if (author in self.author_author) and (len(self.author_author[author]) != 0):
    #                     authors2 = self.author_author[author]
    #                     numa2 = len(authors2)
    #                     if numa2 == 0:
    #                         outline = '#'
    #                         continue
    #                     authorid2 = random.randrange(numa2)
    #                     author2 = authors2[authorid2]
    #                     outline += " " + str(author2)
    #             o = outline.strip().split(' ')
    #             if len(o) == 5:
    #                 outfile.write(outline + "\n")
    #                 # print(k)
    #                 k = k + 1
    #     outfile.close()

    def generate_random_aapaa(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        k = 1
        for author in self.author_author:
            author0 = author
            for j in range(0, numwalks): #wnum walks
                outline = str(author0)
                for i in range(0, walklength):
                    authorss = self.author_author[author]
                    numaa = len(authorss)
                    if numaa == 0:
                        outline = '#'
                        continue
                    authoridd = random.randrange(numaa)
                    authorr = authorss[authoridd]
                    outline += " " + str(authorr)

                    papers = self.author_paper[authorr]
                    nump = len(papers)
                    if nump == 0:
                        outline = '#'
                        continue
                    paperid = random.randrange(nump)
                    paper = papers[paperid]
                    outline += " " + str(paper)

                    authors = self.paper_author[paper]
                    numa = len(authors)
                    if numa == 0:
                        outline = '#'
                        continue
                    authorid = random.randrange(numa)
                    author = authors[authorid]
                    outline += " " + str(author)

                    if (author in self.author_author) and (len(self.author_author[author]) != 0):
                        authors2 = self.author_author[author]
                        numa2 = len(authors2)
                        if numa2 == 0:
                            outline = '#'
                            continue
                        authorid2 = random.randrange(numa2)
                        author2 = authors2[authorid2]
                        outline += " " + str(author2)
                o = outline.strip().split(' ')
                if len(o) == 5:
                    outfile.write(outline + "\n")
                    # print(k)
                    k = k + 1
        outfile.close()

    # def generate_random_apaca(self, outfilename, numwalks, walklength):
    #     outfile = open(outfilename, 'w')
    #     k = 1
    #     for author in self.author_paper:
    #         author0 = author
    #         for j in range(0, numwalks): #wnum walks
    #             outline = str(author0)
    #             for i in range(0, walklength):
    #                 papers = self.author_paper[author]
    #                 nump = len(papers)
    #                 if nump == 0:
    #                     outline = '#'
    #                     continue
    #                 paperid = random.randrange(nump)
    #                 paper = papers[paperid]
    #                 outline += " " + str(paper)
    #
    #                 authorss = self.paper_author[paper]
    #                 numaa = len(authorss)
    #                 if numaa == 0:
    #                     outline = '#'
    #                     continue
    #                 authoridd = random.randrange(numaa)
    #                 authorr = authorss[authoridd]
    #                 outline += " " + str(authorr)
    #
    #                 confs = self.author_conf[authorr]
    #                 numc = len(confs)
    #                 if numc == 0:
    #                     outline = '#'
    #                     continue
    #                 confid = random.randrange(numc)
    #                 conf = confs[confid]
    #                 outline += " " + str(conf)
    #
    #                 authors2 = self.conf_author[conf]
    #                 numa2 = len(authors2)
    #                 if numa2 == 0:
    #                     outline = '#'
    #                     continue
    #                 authorid2 = random.randrange(numa2)
    #                 author2 = authors2[authorid2]
    #                 outline += " " + str(author2)
    #             o = outline.strip().split(' ')
    #             if len(o) == 5:
    #                 outfile.write(outline + "\n")
    #                 # print(k)
    #                 k = k + 1
    #     outfile.close()
    #
    # def generate_random_apcpa(self, outfilename, numwalks, walklength):
    #     outfile = open(outfilename, 'w')
    #     k = 1
    #     for author in self.author_paper:
    #         author0 = author
    #         for j in range(0, numwalks): #wnum walks
    #             outline = str(author0)
    #             for i in range(0, walklength):
    #                 papers = self.author_paper[author]
    #                 nump = len(papers)
    #                 if nump == 0:
    #                     outline = '#'
    #                     continue
    #                 paperid = random.randrange(nump)
    #                 paper = papers[paperid]
    #                 outline += " " + str(paper)
    #
    #                 conf = self.paper_conf[paper]
    #                 outline += " " + str(conf)
    #
    #                 paperss = self.conf_paper[conf]
    #                 numpp = len(paperss)
    #                 if numpp == 0:
    #                     outline = '#'
    #                     continue
    #                 paperidd = random.randrange(numpp)
    #                 paperr = paperss[paperidd]
    #                 outline += " " + str(paperr)
    #
    #                 authors2 = self.paper_author[paperr]
    #                 numa2 = len(authors2)
    #                 if numa2 == 0:
    #                     outline = '#'
    #                     continue
    #                 authorid2 = random.randrange(numa2)
    #                 author2 = authors2[authorid2]
    #                 outline += " " + str(author2)
    #             o = outline.strip().split(' ')
    #             if len(o) == 5:
    #                 outfile.write(outline + "\n")
    #                 # print(k)
    #                 k = k + 1
    #     outfile.close()
    #
    # def generate_random_acpca(self, outfilename, numwalks, walklength):
    #     outfile = open(outfilename, 'w')
    #     k = 1
    #     for author in self.author_conf:
    #         author0 = author
    #         for j in range(0, numwalks):  # wnum walks
    #             outline = str(author0)
    #             for i in range(0, walklength):
    #                 confs = self.author_conf[author]
    #                 numc = len(confs)
    #                 if numc == 0:
    #                     outline = '#'
    #                     continue
    #                 confid = random.randrange(numc)
    #                 conf = confs[confid]
    #                 outline += " " + str(conf)
    #
    #                 paperss = self.conf_paper[conf]
    #                 numpp = len(paperss)
    #                 if numpp == 0:
    #                     outline = '#'
    #                     continue
    #                 paperidd = random.randrange(numpp)
    #                 paperr = paperss[paperidd]
    #                 outline += " " + str(paperr)
    #
    #                 conff = self.paper_conf[paperr]
    #                 outline += " " + str(conff)
    #
    #                 authors = self.conf_author[conff]
    #                 numa = len(authors)
    #                 if numa == 0:
    #                     outline = '#'
    #                     continue
    #                 authorid = random.randrange(numa)
    #                 author = authors[authorid]
    #                 outline += " " + str(author)
    #             o = outline.strip().split(' ')
    #             if len(o) == 5:
    #                 outfile.write(outline + "\n")
    #                 # print(k)
    #                 k = k + 1
    #     outfile.close()


if __name__ == "__main__":
    root_dir = os.getcwd() + '/'  # 获取当前根目录
    data_dir = root_dir + 'PubMed_lncRNA' + '/'  # 数据存取的路径
    output_dir = root_dir + 'output' + '/'  # 存放输出的路径
    a_a = []
    p_a = []

    ff = open(data_dir + 'author_author.txt', 'w')
    with open(data_dir + 'PubMed_lncRNA.txt') as f:
        for line in f:
            ss = line.strip().split(' ')
            if len(ss) != 2:
                print('wrong edges!')
            else:
                if (str(ss[0])[0] == 'a') and (str(ss[1])[0] == 'a'):
                    a_a.append(line)
                    ff.write(line)
                    ff.write(ss[1] + ' ' + ss[0] + '\n')
                elif (str(ss[0])[0] == 'p') and (str(ss[1])[0] == 'a'):
                    p_a.append(line)
                # elif (str(ss[0])[0] == 'p') and (str(ss[1])[0] == 'c'):
                #     p_c.append(line)
                # elif (str(ss[0])[0] == 'a') and (str(ss[1])[0] == 'c'):
                #     a_c.append(line)
                elif (str(ss[0])[0] != 'a' or 'p') or (str(ss[1])[0] != 'a' or 'p'):
                    print('wrong the type of nodes!')
                    exit()
    # IM('MAHE-IM', edge, a_a, p_a, p_c, a_c)
    mpg = MetaPathGenerator()
    mpg.read_data(a_a, p_a)
    outfilename23 = data_dir + 'apa.txt'
    # outfilename22 = data_dir + 'aca.txt'
    outfilename21 = data_dir + 'aaa.txt'
    mpg.generate_random_aa(outfilename21, 10, 2)
    # mpg.generate_random_aca(outfilename22, 10, 1)
    mpg.generate_random_apa(outfilename23, 10, 1)
    outfilename31 = data_dir + 'aaaa.txt'
    # outfilename32 = data_dir + 'acaa.txt'
    outfilename33 = data_dir + 'apaa.txt'
    # outfilename34 = data_dir + 'apca.txt'
    mpg.generate_random_aa(outfilename31, 10, 3)
    # mpg.generate_random_acaa(outfilename32, 10, 1)
    mpg.generate_random_apaa(outfilename33, 10, 1)
    # mpg.generate_random_apca(outfilename34, 10, 1)
    outfilename41 = data_dir + 'aaaaa.txt'
    # outfilename42 = data_dir + 'apaca.txt'
    outfilename43 = data_dir + 'apapa.txt'
    # outfilename44 = data_dir + 'acaca.txt'
    # outfilename45 = data_dir + 'apcpa.txt'
    # outfilename46 = data_dir + 'acpca.txt'
    mpg.generate_random_aa(outfilename41, 10, 4)
    # mpg.generate_random_apaca(outfilename42, 10, 1)
    mpg.generate_random_apa(outfilename43, 10, 2)
    # mpg.generate_random_aca(outfilename44, 10, 2)
    # mpg.generate_random_apcpa(outfilename45, 10, 1)
    # mpg.generate_random_acpca(outfilename46, 5, 1)