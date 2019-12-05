# return features for the datasets we're working with 
import copy 
import gc 
import numpy as np 
import pandas as pd 
import re 
import time 

import torch 

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
            continue
            #print('exception processing line {}'.format(index))
            #print(e, s1, s2)
    return vocab


def tokenize(sent):
    #print(sent)
    return ' '.join([x.strip() for x in re.split('(\W+)?', sent) if x.strip()])


def preprocess(X, train=False, bert=False): 
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
            if not bert:
                s1 = tokenize(row.sentence1).split(' ')
                s2 = tokenize(row.sentence2).split(' ')
            else:
                s1 = row.sentence1 
                s2 = row.sentence2 
            pid = row.pairID 
            if train:
                pair_diff = row.difficulty 
            else:
                pair_diff = 0 

            if lbl != '-':
                sents = [s1, s2]

                data.append(sents)
                lbls.append(lbl)
                diffs.append(pair_diff)
                pids.append(pid) 

        except Exception as e:
            print('Exception on line {}'.format(index))
            error_count += 1

    #print('error count: {}'.format(error_count))
    
    #print(len(data), len(lbls))
    result = {'phrase': data, 'labels': lbls, 'pairID': pids, 'difficulty': diffs}
    return result


def load_snli(data_dir):
    # get data
    '''
    Load some subset of the SNLI dataset, based on diff threshold 
    '''
    # load data files
    #print('loading SNLI data')
    # train_size = 10000
    trainfile = 'snli_1.0_train_diff.txt'
    devfile = 'snli_1.0_dev.txt'
    testfile = 'snli_1.0_test.txt'
    train = pd.read_csv(data_dir +'/processed/' + trainfile, sep='\t',
                        usecols=['gold_label', 'sentence1', 'sentence2', 'pairID', 'difficulty'])
    dev = pd.read_csv(data_dir + '/raw/' + devfile, sep='\t',
                      usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
    test = pd.read_csv(data_dir + '/raw/' + testfile, sep='\t',
                            usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])

    #print(train['sentence1'][0])

    # preprocess them as necessary
    # create the vocab for all of the data for consistency
    #print('generating vocab...')
    vocab = all_vocab(train) | all_vocab(dev) | all_vocab(test)
    vocab_size = len(vocab)

    #print('preprocessing data...')
    out_train = preprocess(train, True)
    out_dev = preprocess(dev)
    out_test = preprocess(test)
    gc.collect()

    le = LabelEncoder()
    le.fit(out_train['labels'])
    out_train['lbls'] = le.transform(out_train['labels'])
    out_dev['lbls'] = le.transform(out_dev['labels'])
    out_test['lbls'] = le.transform(out_test['labels'])

    #print('training set size: {}'.format(len(out_train['lbls'])))

    train, dev, test, vocab = out_train, out_dev, out_test, vocab

    # build embedding table
    #print('loading word vectors...')
    #print('vocab size: {}'.format(len(vocab)))

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
    #print('dict size: {}'.format(len(w2i)))
    return train, dev, test, w2i, i2w, vectors


def load_snli_bert(data_dir):
    # get data
    '''
    Load some subset of the SNLI dataset, based on diff threshold 
    '''
    # load data files
    #print('loading SNLI data')
    # train_size = 10000
    trainfile = 'snli_1.0_train_diff.txt'
    devfile = 'snli_1.0_dev.txt'
    testfile = 'snli_1.0_test.txt'
    train = pd.read_csv(data_dir +'/processed/' + trainfile, sep='\t',
                        usecols=['gold_label', 'sentence1', 'sentence2', 'pairID', 'difficulty'])
    dev = pd.read_csv(data_dir + '/raw/' + devfile, sep='\t',
                      usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
    test = pd.read_csv(data_dir + '/raw/' + testfile, sep='\t',
                            usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])

    #print(train['sentence1'][0])

    # preprocess them as necessary
    # create the vocab for all of the data for consistency
    #print('generating vocab...')
    #vocab = all_vocab(train) | all_vocab(dev) | all_vocab(test)
    #vocab_size = len(vocab)

    #print('preprocessing data...')
    out_train = preprocess(train, train=True, bert=True)
    out_dev = preprocess(dev, bert=True)
    out_test = preprocess(test, bert=True)
    gc.collect()

    return out_train, out_dev, out_test


############# SSTB ################

def load_sstb(data_dir):
    # load SSTB data
    trainfile = 'sstb_train_diff.tsv'
    devfile = 'sstb_dev.tsv'
    testfile = 'sstb_test.labeled.tsv'
    label_set_MASTER = [0, 1]

    with open(data_dir + '/processed/' + trainfile, 'r') as infile:
        training_data = infile.readlines()[1:]
        training_data = [t.split('\t') for t in training_data]
        TRAIN_SIZE = len(training_data)
        idx = list(range(TRAIN_SIZE))  # generate labels for each item 
         
        # load training data 
        train = {}
        train['lbls'] = [] 
        train['phrase'] = [] 
        train['difficulty'] = []
        train['pairID'] = []
        for i in range(TRAIN_SIZE):
            train['lbls'].append(eval(training_data[i][1]))
            train['phrase'].append(tokenize(training_data[i][0].strip()).split(' ')) 
            train['difficulty'].append(eval(training_data[i][3])) 
            train['pairID'].append(eval(training_data[i][2]))

    with open(data_dir + '/raw/' + devfile, 'r') as infile:
        dev_data = infile.readlines()[1:]
        dev = {}
        dev['lbls'] = [eval(l.split('\t')[1]) for l in dev_data]
        dev['phrase'] = [tokenize(l.split('\t')[0].strip()).split(' ') for l in dev_data]
        dev['pairID'] = list(range(len(dev['phrase'])))

    with open(data_dir + '/raw/' + testfile, 'r') as infile:
        test_data = infile.readlines()[1:]
        test = {}
        test['lbls'] = [eval(l[0]) for l in test_data]
        #test['lbls'] = [0] * len(test_data)  # for now
        test['phrase'] = [tokenize(l[1:].strip()).split(' ') for l in test_data]
        test['pairID'] = list(range(len(test['phrase']))) 

    # build vocab
    vocab = set()
    for dataset in [train, dev, test]:
        for r in dataset['phrase']:
            vocab.update(r)

    # build embeddings
    #print('loading word vectors...')
    #print('vocab size: {}'.format(len(vocab)))

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

    out_train = train
    out_dev = dev

    out_test = test

    gc.collect()
    #print('dict size: {}'.format(len(w2i)))
    return out_train, out_dev, out_test, w2i, i2w, vectors



def load_sstb_bert(data_dir):
    # load SSTB data
    trainfile = 'sstb_train_diff.tsv'
    devfile = 'sstb_dev.tsv'
    testfile = 'sstb_test.labeled.tsv'
    label_set_MASTER = [0, 1]

    with open(data_dir + '/processed/' + trainfile, 'r') as infile:
        training_data = infile.readlines()[1:]
        training_data = [t.split('\t') for t in training_data]
        TRAIN_SIZE = len(training_data)
        idx = list(range(TRAIN_SIZE))  # generate labels for each item 
         
        # load training data 
        train = {}
        train['lbls'] = [] 
        train['phrase'] = [] 
        train['difficulty'] = []
        train['pairID'] = []
        for i in range(TRAIN_SIZE):
            train['lbls'].append(eval(training_data[i][1]))
            train['phrase'].append(training_data[i][0]) 
            train['difficulty'].append(eval(training_data[i][3])) 
            train['pairID'].append(eval(training_data[i][2]))

    with open(data_dir + '/raw/' + devfile, 'r') as infile:
        dev_data = infile.readlines()[1:]
        dev = {}
        dev['lbls'] = [eval(l.split('\t')[1]) for l in dev_data]
        dev['phrase'] = [l.split('\t')[0] for l in dev_data]
        dev['pairID'] = list(range(len(dev['phrase'])))

    with open(data_dir + '/raw/' + testfile, 'r') as infile:
        test_data = infile.readlines()[1:]
        test = {}
        test['lbls'] = [eval(l[0]) for l in test_data]
        #test['lbls'] = [0] * len(test_data)  # for now
        test['phrase'] = [l[1:] for l in test_data]
        test['pairID'] = list(range(len(test['phrase']))) 

    out_train = train
    out_dev = dev

    out_test = test

    gc.collect()
    #print('dict size: {}'.format(len(w2i)))
    return out_train, out_dev, out_test


####### Get CL Data per epoch ########
def get_epoch_training_data(ts, args, epoch, task, theta_hat=None, diffs_sorted_idx=None):
    if args.strategy == 'baseline':
        return ts 
    if args.strategy == 'theta':
        assert theta_hat is not None and args.ordering == 'easiest' 
    if args.strategy == 'theta-hard':
        assert theta_hat is not None and args.ordering == 'hardest' 

    training_set = copy.deepcopy(ts) 
    c_init = 0.01  # as per naacl19 paper 
 
    # set up CL
    train_2 = {
        'pairID': [],
        'lbls':[],
        'phrase':[ ],
        'difficulty': []
    }
    train = {
        'pairID': [],
        'lbls':[],
        'phrase':[ ],
        'difficulty': []
    }
    
    # how will we order the data before building curriculum?
    # difficulty baseline: use the length of the text as a proxy
    if args.use_length:
        if task == 'sstb':
            training_set['difficulty'] = [len(p) for p in training_set['phrase']]
        else:  # snli
            training_set['difficulty'] = [len(p[0]) for p in training_set['phrase']] 

    if diffs_sorted_idx is None:
        difficulties = [img[3] for img in training_set]
        diffs_sorted_idx = np.argsort(difficulties) 

    if args.ordering == 'easiest':
        diffs_sorted_idx = np.argsort(training_set['difficulty']) 
    elif args.ordering == 'hardest':
        diffs_sorted_idx = np.argsort(training_set['difficulty'])[::-1]
    elif args.ordering == 'middleout':  # middle out
        diffs_sorted_idx = np.argsort(np.abs(training_set['difficulty'])) 
    else:  # random baseline 
        raise NotImplementedError
    #if args.k > 0:
    #    diffs_sorted_idx = k_sort(training_set['difficulty'], args.k) 

    # determine if we want balanced per-label or not
    if not args.balanced:
        train_2['phrase'] = [training_set['phrase'][d] for d in diffs_sorted_idx] 
        train_2['lbls'] = [training_set['lbls'][d] for d in diffs_sorted_idx] 
        train_2['difficulty'] = [training_set['difficulty'][d] for d in diffs_sorted_idx]
        train_2['pairID'] = [training_set['pairID'][d] for d in diffs_sorted_idx]
    else:
        per_label_lists = {
            0:[], 1:[]
        }
        if task == 'snli':
            per_label_lists[2] = [] 
        for d in diffs_sorted_idx:
            eg = training_set['lbls'][d] 
            per_label_lists[eg].append(d) 

        #max_length = max(len(per_label_lists[0]), len(per_label_lists[1]))
        max_length = max([len(v) for k,v in per_label_lists.items()])
        train_2_idx = []
        for l in range(max_length):
            for k, v in per_label_lists.items():
                if l < len(v):
                    train_2_idx.append(v[l])

        train_2['phrase'] = [training_set['phrase'][z] for z in train_2_idx]
        train_2['lbls'] = [training_set['lbls'][z] for z in train_2_idx]
        train_2['difficulty'] = [training_set['difficulty'][z] for z in train_2_idx]
        train_2['pairID'] = [training_set['pairID'][z] for z in train_2_idx]

    # based on strategy, select our training set for this epoch
    if args.strategy == 'ordered':
        return train_2 
    elif args.strategy == 'simple':
        # how many examples do we want this epoch? 
        # CL for first half, full data for 2nd half 
        data_per_epoch = len(training_set['phrase']) / (args.num_epochs / 2.)
        if epoch % 2 == 0:
            num_train = min(int(data_per_epoch * (epoch + 1)), len(training_set['phrase'])) 
        else:
            num_train = min(int(data_per_epoch * (epoch)), len(training_set['phrase'])) 
        train['lbls'] = [train_2['lbls'][i] for i in range(num_train)] 
        train['phrase'] = [train_2['phrase'][i] for i in range(num_train)] 
        train['difficulty'] = [train_2['difficulty'][i] for i in range(num_train)]
        train['pairID'] = [train_2['pairID'][i] for i in range(num_train)]
        return train 
    elif args.strategy == 'theta':
        train_idx = [i for i in range(len(training_set['phrase'])) if train_2['difficulty'][i] <= theta_hat]
        if len(train_idx) < args.min_train_length:
            train_idx = [i for i in range(args.min_train_length)] 
        train['lbls'] = [train_2['lbls'][i] for i in train_idx] 
        train['phrase'] = [train_2['phrase'][i] for i in train_idx] 
        train['difficulty'] = [train_2['difficulty'][i] for i in train_idx]
        train['pairID'] = [train_2['pairID'][i] for i in train_idx]
        return train 
    elif args.strategy == 'theta-hard':
        train_idx = [i for i in range(len(training_set['phrase'])) if train_2['difficulty'][i] >= theta_hat]
        if len(train_idx) < args.min_train_length:
            train_idx = [i for i in range(args.min_train_length)] 
        train['lbls'] = [train_2['lbls'][i] for i in train_idx] 
        train['phrase'] = [train_2['phrase'][i] for i in train_idx] 
        train['difficulty'] = [train_2['difficulty'][i] for i in train_idx]
        train['pairID'] = [train_2['pairID'][i] for i in train_idx]
        return train 
    elif args.strategy == 'naacl-linear':
        epoch_competency = np.min([1, epoch * ((1 - c_init)/args.competency) + c_init])
        train['lbls'] = [train_2['lbls'][i] for i in range(int(epoch_competency * len(training_set['phrase'])))]
        train['phrase'] = [train_2['phrase'][i] for i in range(int(epoch_competency * len(training_set['phrase'])))]
        train['difficulty'] = [train_2['difficulty'][i] for i in range(int(epoch_competency * len(training_set['phrase'])))]
        train['pairID'] = [train_2['pairID'][i] for i in range(int(epoch_competency * len(training_set['phrase'])))]
        return train 
    elif args.strategy == 'naacl-root':
        epoch_competency = np.min([1,np.sqrt(epoch * ((1 - c_init**2)/args.competency) + c_init**2)])
        train['lbls'] = [train_2['lbls'][i] for i in range(int(epoch_competency * len(training_set['phrase'])))]
        train['phrase'] = [train_2['phrase'][i] for i in range(int(epoch_competency * len(training_set['phrase'])))]
        train['difficulty'] = [train_2['difficulty'][i] for i in range(int(epoch_competency * len(training_set['phrase'])))]
        train['pairID'] = [train_2['pairID'][i] for i in range(int(epoch_competency * len(training_set['phrase'])))]
        return train 
    else:
        raise NotImplementedError


### CL for vision data sets (since they are built slightly differently)
def get_epoch_training_data_vision(ts, args, epoch, theta_hat=None, diffs_sorted_idx=None):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    c_init = 0.01  # as per naacl19 paper 

    if args.strategy == 'baseline':
        return torch.utils.data.DataLoader(ts,
                batch_size=args.batch_size, shuffle=True, **kwargs)  
    if args.strategy == 'theta':
        assert theta_hat is not None and args.ordering == 'easiest' 
    if args.strategy == 'theta-hard':
        assert theta_hat is not None and args.ordering == 'hardest' 

    training_set = copy.deepcopy(ts)
 
    # set up CL #     
    # how will we order the data before building curriculum?
    if diffs_sorted_idx is None:
        difficulties = [img[3] for img in training_set]
        diffs_sorted_idx = np.argsort(difficulties) 

    if args.ordering == 'easiest':
        diffs_sorted_idx = diffs_sorted_idx
    elif args.ordering == 'hardest':
        diffs_sorted_idx = diffs_sorted_idx[::-1]
    elif args.ordering == 'middleout':  # middle out
        diffs_sorted_idx = np.argsort(np.abs(difficulties))  
    else:  # random baseline 
        raise NotImplementedError

    # determine if we want balanced per-label or not
    if not args.balanced:
        train_2 = [training_set[d] for d in diffs_sorted_idx]
    else:
        per_label_lists = {
                0:[], 1:[], 2:[], 3:[], 4:[], 
                5:[], 6:[], 7:[], 8:[], 9:[]
            }
        for d in diffs_sorted_idx:
            eg = training_set[d]
            try:
                per_label_lists[eg[1].item()].append(d) 
            except:
                per_label_lists[eg[1]].append(d) 

        max_length = max([len(v) for k,v in per_label_lists.items()])
        train_2_idx = []
        for l in range(max_length):
            for k, v in per_label_lists.items():
                if l < len(v):
                    train_2_idx.append(v[l])
        train_2 = [training_set[j] for j in train_2_idx]

    # based on strategy, select our training set for this epoch
    if args.strategy == 'ordered':
        return torch.utils.data.DataLoader(train_2,
                batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.strategy == 'simple':
        # how many examples do we want this epoch? 
        # CL for first half, full data for 2nd half 
        data_per_epoch = len(training_set) / (args.num_epochs / 2.)
        if epoch % 2 == 0:
            num_train = min(int(data_per_epoch * (epoch + 1)), len(training_set))
        else:
            num_train = min(int(data_per_epoch * (epoch)), len(training_set))
        train = [train_2[i] for i in range(num_train)] 
        return torch.utils.data.DataLoader(train,
                batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.strategy == 'theta':
        train = [train_2[i] for i in range(len(training_set)) if train_2[i][3] <= theta_hat]
        if len(train) < args.min_train_length:
            train = [train_2[i] for i in range(args.min_train_length)] 
        return torch.utils.data.DataLoader(train,
                batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.strategy == 'theta-hard':
        train = [train_2[i] for i in range(len(training_set)) if train_2[i][3] >= theta_hat]
        if len(train) < args.min_train_length:
            train = [train_2[i] for i in range(args.min_train_length)] 
        return torch.utils.data.DataLoader(train,
                batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.strategy == 'naacl-linear':
        epoch_competency = np.min([1, epoch * ((1 - c_init)/args.competency) + c_init])
        train = [train_2[i] for i in range(int(epoch_competency * len(training_set)))]
        return torch.utils.data.DataLoader(train,
                batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.strategy == 'naacl-root':
        epoch_competency = np.min([1,np.sqrt(epoch * ((1 - c_init**2)/args.competency) + c_init**2)])
        train = [train_2[i] for i in range(int(epoch_competency * len(training_set)))]
        return torch.utils.data.DataLoader(train,
                batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        raise NotImplementedError

    
def k_sort(D, k):
    '''
    take a fully sorted input and return a random shuffling that is k-sorted (each element is at most k away from the right spot)
    '''
    D_shuffled_idx = list(np.argsort(D)) 

    for i in range(0, len(D_shuffled_idx), k):
        lst_slice = D_shuffled_idx[i:i+k]
        np.random.shuffle(lst_slice)
        D_shuffled_idx[i:i+k] = lst_slice
    
    '''
    k_sorted = False 
    for i in range(len(D_shuffled) - 1):
        for j in range(len(D_shuffled) - i - 1):
            if D[j] > D[j+1]:
                D[j], D[j+1] = D[j+1], D[j]
                D_idx[j], D_idx[j+1] = D_idx[j+1], D_idx[j] 
        k_sorted = True  
        for j in range(len(D_shuffled)):
            if np.abs(j - D_shuffled_idx.index(D_idx[j])) <= k:
                continue 
            else:
                k_sorted = False 
                break 
        if k_sorted:
            break 
    ''' 

    result = D_shuffled_idx  

    return result 

