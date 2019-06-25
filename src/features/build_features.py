# return features for the datasets we're working with 
import gc 
import pandas as pd 
import re 

from sklearn.preprocessing import LabelEncoder



###### SNLI ########


### HELPER Functions ###
def all_vocab(txt):
    vocab = set()
    for index, row in txt.iterrows():
        try:
            s1 = row['sentence1']
            s2 = row['sentence2']
            vocab.update(tokenize(s1).split(' '))
            vocab.update(tokenize(s2).split(' '))
        except Exception as e:
            print('exception processing line {}'.format(index))
            #print(e, s1, s2)
    return vocab


def tokenize(sent):
    #print(sent)
    return ' '.join([x.strip() for x in re.split('(\W+)?', sent) if x.strip()])


def preprocess(X, train=False): 
    lbls = []
    data = []
    pids = []
    diffs = []
    error_count = 0            

    #for i in range(len(X.gold_label)):
    for index, row in X.iterrows():
        try:
            label_set = ['entailment', 'contradiction', 'neutral']
            lbl = row.gold_label
            s1 = tokenize(row.sentence1).split(' ')
            s2 = tokenize(row.sentence2).split(' ')
            if train:
                pair_diff = row.difficulty 
            else:
                pair_diff = 0 

            if lbl != '-':
                sents = [s1, s2]

                data.append(sents)
                lbls.append(lbl)
                diffs.append(pair_diff)

        except Exception as e:
            print('Exception on line {}'.format(index))
            error_count += 1

    print('error count: {}'.format(error_count))
    
    print(len(data), len(lbls))
    result = {'sents': data, 'labels': lbls, 'pairIDs': pids, 'difficulty': diffs}
    return result


def load_snli(data_dir):
    # get data
    '''
    Load some subset of the SNLI dataset, based on diff threshold 
    '''
    # load data files
    print('loading SNLI data')
    # train_size = 10000
    trainfile = 'snli_1.0_train_diff.txt'
    devfile = 'snli_1.0_dev.txt'
    testfile = 'snli_1.0_test.txt'
    train = pd.read_csv(data_dir +'/processed/' + trainfile, sep='\t',
                        usecols=['gold_label', 'sentence1', 'sentence2', 'difficulty'])
    dev = pd.read_csv(data_dir + '/raw/' + devfile, sep='\t',
                      usecols=['gold_label', 'sentence1', 'sentence2'])
    test = pd.read_csv(data_dir + '/raw/' + testfile, sep='\t',
                            usecols=['gold_label', 'sentence1', 'sentence2'])

    print(train['sentence1'][0])

    # preprocess them as necessary
    # create the vocab for all of the data for consistency
    print('generating vocab...')
    vocab = all_vocab(train) | all_vocab(dev) | all_vocab(test)
    vocab_size = len(vocab)

    print('preprocessing data...')
    out_train = preprocess(train)
    out_dev = preprocess(dev)
    out_test = preprocess(test)
    gc.collect()

    le = LabelEncoder()
    le.fit(out_train['labels'])
    out_train['lbls'] = le.transform(out_train['labels'])
    out_dev['lbls'] = le.transform(out_dev['labels'])
    out_test['lbls'] = le.transform(out_test['labels'])

    print('training set size: {}'.format(len(out_train['lbls'])))

    train, dev, test, vocab = out_train, out_dev, out_test, vocab

    # build embedding table
    print('loading word vectors...')
    print('vocab size: {}'.format(len(vocab)))

    vectors = []
    w2i = {}
    i2w = {}

    i = 0
    
    with open(data_dir + '/raw/' + 'glove.840B.300d.txt', 'r', encoding='utf-8') as glovefile:
        for j, line in enumerate(glovefile):
            vals = line.rstrip().split(' ')
            if vals[0] in vocab:
                w2i[vals[0]] = i
                i2w[i] = vals[0]
                vectors.append(list(map(float, vals[1:])))
                i += 1

    # now I need to look at vocab words that aren't in glove
    next_i = len(vectors)
    dict_keys = w2i.keys()
    for w in vocab:
        if w not in dict_keys:
            w2i[w] = next_i
            i2w[next_i] = w
            next_i += 1
    w2i['<PAD>'] = next_i
    i2w[next_i] = '<PAD>'

    gc.collect()
    print('dict size: {}'.format(len(w2i)))
    return train, dev, test, w2i, i2w, vectors

