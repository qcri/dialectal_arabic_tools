#!/usr/bin/env python
# -*- coding: utf8 -*-
# Generate CoNLL format
import codecs

__author__ = 'disooqi'


input_file = r'/home/disooqi/PycharmProjects/dialectal_arabic_tools/data/glf.trg'
output_file = r'/home/disooqi/PycharmProjects/dialectal_arabic_tools/data/all_02.trg.conll'

with codecs.open(input_file, encoding='utf-8') as orig:
    with codecs.open(output_file, mode='a', encoding='utf-8') as conll:
        for line in orig:

            if line.strip() in['EOS', 'EOTWEET']:
                conll.write('\n')
                continue
            clitics = line.strip().split('+')
            for clitic in clitics:
                if len(clitic) == 1:
                    conll.write(clitic + '\tS\n')
                elif len(clitic) == 2:
                    conll.write(clitic[0] + '\tB\n')
                    conll.write(clitic[1] + '\tE\n')
                else:
                    conll.write(clitic[0] + '\tB\n')
                    for ch in clitic[1:-1]:
                        conll.write(ch + '\tM\n')
                    else:
                        conll.write(clitic[-1] + '\tE\n')
            else:
                conll.write('WB\tWB\n')
        else:
            conll.write('=======================================')


# filename = r'data_1.dev'
# with codecs.open(r'data/egy_seg/splits/'+filename+'.trg', encoding='utf-8') as orig:
#     with codecs.open(r'data/egy_seg/splits/'+filename+'.conll', mode='w', encoding='utf-8') as conll:
#         for line in orig:
#             # print(line.strip())
#             clitics = line.strip().split('+')
#             for clitic in clitics:
#                 if len(clitic) == 1:
#                     conll.write('S')
#                 elif len(clitic) == 2:
#                     conll.write('BE')
#                 else:
#                     conll.write('B'+(len(clitic)-2)*'M'+'E')
#             else:
#                 conll.write('\n')
