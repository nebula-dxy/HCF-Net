# -*- coding:utf-8 -*-

"""
Get seed set.
"""


def select_seed(input):
    print('select seed!')
    dic = {}
    j = 0
    for i in input:
        li = str(i).replace('b"', '').replace('(', '').replace('"', '').replace("'", '').replace('[', '').replace(
            ']', '').replace(')', '').strip()
        p = 0
        for i in li.split(','):
            p = p + 1
            if (p % 2) == 1:
                ii = str(i).replace('b\\', '').replace('\\', '').strip()
                if ii in dic:
                    dic[ii] += 1
                else:
                    dic[ii] = 1
                    j = j + 1

    s = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    sort = ''
    for ss in s:
        sss = list(ss)
        sss0 = sss[0]
        sort = sort + str(sss0) + '\n'
    return sort
