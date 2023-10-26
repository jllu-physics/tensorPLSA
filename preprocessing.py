import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, sent_tokenize
import nltk
import pickle

def parse_data(path2data, lmtz = WordNetLemmatizer(), tknz = TweetTokenizer()):
    index = 0
    user_dic = {}
    srcs = []
    tgts = []
    vots = []
    cmts = []
    line_number = 0
    stopwords = [word.lower() for word in nltk.corpus.stopwords.words('english')]
    ignore = set(['\'',',','.','[',']','(',')','{','}','-',':',';','|','/','*','@','#','$','%','^','&','=','~','+','<','>'] + stopwords)
    with open(path2data) as file:
        for raw_line in file:
            line = raw_line.rstrip()
            #if line_number < 100:
            #    print(line)
            if index == 0:
                if line[:4] != 'SRC:':
                    print(line_number,"invalid line", line)
                if line[4:] not in user_dic:
                    user_dic[line[4:]] = len(user_dic)
                srcs.append(user_dic[line[4:]])
            if index == 1:
                if line[:4] != 'TGT:':
                    print(line_number,"invalid line", line)
                if line[4:] not in user_dic:
                    user_dic[line[4:]] = len(user_dic)
                tgts.append(user_dic[line[4:]])
            if index == 2:
                if line[:4] != 'VOT:' or int(line[4:]) not in (1,0,-1):
                    print(line_number,"invalid line", line)
                vots.append(int(line[4:])+1)
            if index == 6:
                if line[:4] != 'TXT:':
                    print(line_number,"invalid line", line)
                comment = line[4:]
                if comment[:13] == '\'\'\'Support\'\'\'':
                    comment = comment[13:]
                elif comment[:12] == '\'\'\'Oppose\'\'\'':
                    comment = comment[12:]
                tokens = [lmtz.lemmatize(token.lower()) for token in tknz.tokenize(comment)]
                cmts.append([token for token in tokens if token not in ignore])
            index += 1
            if index == 8:
                index = 0
            line_number += 1
    return srcs, tgts, vots, cmts, user_dic

def count_word(cmts):
    ct = {}
    for cmt in cmts:
        for word in cmt:
            if word in ct:
                ct[word] += 1
            else:
                ct[word] = 1
    return ct

def make_word2index(word_counts, threshold):
    ct = word_counts
    word2index = {}
    #words = []
    for word in ct:
        if ct[word] > threshold:
            word2index[word] = len(word2index)
            #words.append(word)
    return word2index

def word_encoding(cmts, word2index):
    L = len(word2index)
    cwds = []
    for cmt in cmts:
        tmp =[]
        for word in cmt:
            if word in word2index:
                tmp.append(word2index[word])
            else:
                tmp.append(L)
        if len(tmp) == 0:
            tmp.append(L)
        cwds.append(tmp)
    return cwds

def flatten_data(srcs, tgts, vots, cwds):
    flat_srcs = []
    flat_tgts = []
    flat_vots = []
    flat_cwds = []
    for i in range(len(srcs)):
        n = len(cwds[i])
        flat_srcs += [srcs[i],]*n
        flat_tgts += [tgts[i],]*n
        flat_vots += [vots[i],]*n
        flat_cwds += cwds[i]
    return flat_srcs, flat_tgts, flat_vots, flat_cwds

def prepare_data(path2data, lmtz = WordNetLemmatizer(), tknz = TweetTokenizer(), threshold = 100, train_ratio = 0.7, valid_ratio = 0.05, return_dicts = False, seed = 0):
    srcs, tgts, vots, cmts, user_dic = parse_data(path2data, lmtz, tknz)
    size = len(srcs)
    train_size = int(size*train_ratio)
    valid_size = int(size*valid_ratio)
    index = np.arange(len(srcs))
    np.random.seed(seed)
    np.random.shuffle(index)
    train_index = index[:train_size]
    valid_index = index[train_size:(train_size+valid_size)]
    test_index = index[(train_size+valid_size):]
    train_srcs = [srcs[i] for i in train_index]
    train_tgts = [tgts[i] for i in train_index]
    train_vots = [vots[i] for i in train_index]
    train_cmts = [cmts[i] for i in train_index]
    ct = count_word(train_cmts)
    word2index = make_word2index(ct, threshold)
    train_cwds = word_encoding(train_cmts, word2index)
    valid_srcs = [srcs[i] for i in valid_index]
    valid_tgts = [tgts[i] for i in valid_index]
    valid_vots = [vots[i] for i in valid_index]
    valid_cmts = [cmts[i] for i in valid_index]
    valid_cwds = word_encoding(valid_cmts, word2index)
    valid = [valid_srcs, valid_tgts, valid_vots, valid_cwds]
    test_srcs = [srcs[i] for i in test_index]
    test_tgts = [tgts[i] for i in test_index]
    test_vots = [vots[i] for i in test_index]
    test_cmts = [cmts[i] for i in test_index]
    test_cwds = word_encoding(test_cmts, word2index)
    test = [test_srcs, test_tgts, test_vots, test_cwds]

    ftrain_srcs, ftrain_tgts, ftrain_vots, ftrain_cwds = flatten_data(train_srcs, train_tgts, train_vots, train_cwds)
    flat_train = np.vstack((ftrain_srcs, ftrain_tgts, ftrain_vots, ftrain_cwds)).T
    fvalid_srcs, fvalid_tgts, fvalid_vots, fvalid_cwds = flatten_data(valid_srcs, valid_tgts, valid_vots, valid_cwds)
    flat_valid = np.vstack((fvalid_srcs, fvalid_tgts, fvalid_vots, fvalid_cwds)).T
    if return_dicts:
        return flat_train, flat_valid, valid, test, user_dic, word2index
    else:
        return flat_train, flat_valid, valid, test
