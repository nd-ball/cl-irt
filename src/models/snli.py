# lstm rnn for snli

import numpy as np
import dynet as dy
import argparse
import random
import gc
import csv
import pandas as pd

import datetime 
import time 
import os 

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 

from features.build_features import load_snli, get_epoch_training_data, k_sort, load_glue_task, tokenize
from features.irt_scoring import calculate_theta, calculate_diff_threshold 

GLUETASKS = ['MRPC', 'MNLI', 'QNLI', 'RTE', 'QQP']

parser = argparse.ArgumentParser()
parser.add_argument('--dynet-autobatch', help='DyNet requirement for autobatching')
parser.add_argument('--dynet-gpus', help='DyNet requirement to trigger gpu')
parser.add_argument('--dynet-gpu', help='DyNet requirement to trigger gpu')
parser.add_argument('--dynet-mem', help='DyNet requirement to allocate memory')
parser.add_argument('--dynet-weight-decay', help='DyNet requirement for regularization')

parser.add_argument('--gpu', type=int, default=-1, help='use GPU?')
parser.add_argument('--num-units', type=int, default=300, help='number of units per layer')
parser.add_argument('--data-dir') 
parser.add_argument('--balanced', action='store_true') 
parser.add_argument('--strategy', choices=['baseline', 'ordered', 'simple', 'theta', 'naacl-linear', 'naacl-root', 'theta-hard'],
                    help='CL data policy', default='simple')
parser.add_argument('--ordering', choices=['easiest', 'hardest', 'middleout'], default='easiest') 
parser.add_argument('--num-epochs', type=int, default=100) 
parser.add_argument('--random', action='store_true') 
parser.add_argument('--use-length', action='store_true')
parser.add_argument('--use-word-rarity', action='store_true')
parser.add_argument('--min-train-length', default=100, type=int)
parser.add_argument('--k', default=0, type=int) 
parser.add_argument('--competency', default=50, type=int) 
parser.add_argument('--p-correct', default=0.5, type=float, help="P(correct) to filter training data for IRT")
parser.add_argument('--diff-dir', help='path to SNLI dataset')
parser.add_argument('--task', choices=GLUETASKS, help='GLUE task for fine-tuning')
args = parser.parse_args()

print(args)

VOCAB_SIZE = 0
INPUT_DIM = 100

outdir = 'results/lstm-{}/{}-{}-len-{}-wordrarity-{}/{}/'.format(
        args.balanced,
        args.task,
        args.strategy,
        args.use_length,
        args.use_word_rarity,
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )


#preds_file = '{}processed/test_predictions/snli_{}_{}_{}_{}_{}.csv'.format(args.data_dir, args.strategy, args.balanced, args.ordering, args.random, args.k) 
#outfile = open(preds_file, 'w') 
#outwriter = csv.writer(outfile, delimiter=',')
#outwriter.writerow(['epoch', 'itemID', 'correct', 'pred'])

# first define the LSTM model that we'll be using 
# TODO: rip this out and replace with a more advanced model (in PyTorch) 

class RNNBuilder:
    '''
    This model is going to roughly look like the following:
    premise (300d) -> tanh layer (100d) -> LSTM (100d)
    hypothesis (300d) -> tanh layer (100d) -> LSTM (100d)

    concat the two above 100D vectors (200D)
    tanh -> tanh -> tanh -> softmax
    '''

    def __init__(self, model, out_dim, WORD_EMBS, w2i, i2w, lstm_dim):
        self.m = model
        #self.trainer = dy.AdadeltaTrainer(model, eps=1e-7, rho=0.95)
        self.trainer = dy.SimpleSGDTrainer(self.m)
        self.LSTM_DIM = lstm_dim
        self.TANH_DIM = 2 * lstm_dim
        self.EMB_IN_DIM = 300
        self.EMB_OUT_DIM = 100
        self.p_first = 0.10
        self.p_second = 0.25

        # parameters
        self.W_premise = self.m.add_parameters((self.EMB_IN_DIM, self.EMB_OUT_DIM))
        self.W_hypothesis = self.m.add_parameters((self.EMB_IN_DIM, self.EMB_OUT_DIM))
        self.b_premise = self.m.add_parameters(self.EMB_OUT_DIM)
        self.b_hypothesis = self.m.add_parameters(self.EMB_OUT_DIM)

        self.W_tanh1 = self.m.add_parameters((self.TANH_DIM, self.TANH_DIM))
        self.W_tanh2 = self.m.add_parameters((self.TANH_DIM, self.TANH_DIM))
        self.W_tanh3 = self.m.add_parameters((self.TANH_DIM, self.TANH_DIM))
        self.b_tanh1 = self.m.add_parameters(self.TANH_DIM)
        self.b_tanh2 = self.m.add_parameters(self.TANH_DIM)
        self.b_tanh3 = self.m.add_parameters(self.TANH_DIM)
        self.W_out = self.m.add_parameters((self.TANH_DIM, out_dim))
        self.b_out = self.m.add_parameters(out_dim)
        
        self.p_builder = dy.CoupledLSTMBuilder(1, self.EMB_IN_DIM, self.LSTM_DIM, self.m)
        self.h_builder = dy.CoupledLSTMBuilder(1, self.EMB_IN_DIM, self.LSTM_DIM, self.m)
        self.lookup = self.m.add_lookup_parameters((len(w2i), self.EMB_IN_DIM))

        for i in range(len(WORD_EMBS)):
            self.lookup.init_row(i, WORD_EMBS[i])

        self.w2i = w2i
        self.i2w = i2w

    def forward(self, p, h, label, train=True):
        p_wembs = [self.lookup[self.w2i[w]] for w in p]
        h_wembs = [self.lookup[self.w2i[w]] for w in h]

        p_state = self.p_builder.initial_state()
        h_state = self.h_builder.initial_state()

        lstm_input_p = p_wembs
        lstm_input_h = h_wembs
        output_p = p_state.transduce(lstm_input_p)[-1]
        output_h = h_state.transduce(lstm_input_h)[-1]

        if train:
            output_p = dy.dropout(output_p, self.p_second)
            output_h = dy.dropout(output_h, self.p_second)

        emb_out = dy.concatenate([output_p, output_h])

        # 3 tanh nonlinearities
        t1 = dy.tanh((self.W_tanh1 * emb_out) + self.b_tanh1)
        t2 = dy.tanh((self.W_tanh2 * t1) + self.b_tanh2)
        t3 = dy.tanh((self.W_tanh3 * t2) + self.b_tanh3)

        # output
        out = (dy.transpose(self.W_out) * t3) + self.b_out
        return out


def run():

    # variables
    num_epoch = args.num_epochs
    batch_size = 64
    pre_trained_embs = True
    out_dim = 3

    #print('num_epoch: {}\nbatch_size: {}'.format(num_epoch, batch_size))
    exp_label = '{}_{}_{}_{}'.format(args.strategy, args.balanced, args.ordering, args.random)

    # save training data set size to disk for bookkeeping
    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    progress_file = open(outdir + 'tracker.csv', 'w')
    progress_writer = csv.writer(progress_file)
    progress_writer.writerow(["epoch","num_training_examples", "dev_acc", "test_acc"])

    #train, dev, test, w2i, i2w, vectors = load_snli(args.data_dir)  
    train, dev, test = load_glue_task(args.data_dir, args.diff_dir, args.task)  
    if args.random:
        random.shuffle(train['difficulty'])
    # build embeddings
    vectors = []
    w2i = {}
    i2w = {}

    # build vocab
    vocab = set()
    for dataset in [train, dev, test]:
        for s1, s2 in dataset['phrase']:
            vocab.update(tokenize(s1).split(' '))
            vocab.update(tokenize(s2).split(' '))

    i = 0
    with open(args.data_dir + '/glove/' + 'glove.840B.300d.txt', 'r', encoding='utf-8') as glovefile:
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

    # if k is set, sort once
    if args.k > 0:
        diffs = train['difficulty'] 
        diffs_sorted_idx = k_sort(diffs, args.k) 
    else:
        diffs_sorted_idx = None 

    # load model and train
    #print('initialize model...')
    model = dy.Model()
    dnnmodel = RNNBuilder(model, out_dim, vectors, w2i, i2w, args.num_units)
    
    max_epoch = 0
    max_train = 0
    max_dev = 0
    max_test = 0
    num_train = len(train['phrase'])
    num_dev = len(dev['phrase'])
    num_test = len(test['phrase'])
    top_dev = 0.0
    top_dev_epoch = 0
    top_dev_test = 0.0

    # build the label encoder
    le = LabelEncoder()
    le.fit(train['lbls']) 
    train['lbls'] = le.transform(train['lbls'])
    dev['lbls'] = le.transform(dev['lbls']) 
    test['lbls'] = le.transform(test['lbls']) 

    #print('Training model {}'.format(model))
    #print('training')
    for i in range(num_epoch):
        # estimate theta for current model parameters
        if args.strategy in ['theta', 'theta-hard']:
            labels = []
            predictions = []
            correct = []
            pairIDs = []
            outs = []
            k = 0
            for j in range(num_train):
                if k % batch_size == 0:
                    dy.renew_cg()
                    outs = []

                sent1, sent2 = train['phrase'][j]
                sent1 = tokenize(sent1).split(' ')
                sent2 = tokenize(sent2).split(' ')
                lbl = train['lbls'][j]
                correct.append(lbl)
                out = dy.softmax(dnnmodel.forward(sent1, sent2, lbl, False))
                outs.append(out)

                if k % batch_size == batch_size - 1:
                    dy.forward(outs)
                    predictions.extend([o.npvalue() for o in outs])
                    outs = []
                k += 1

            # final pass if there are unbatched values
            if len(outs) > 0:
                dy.forward(outs)
                predictions.extend([o.npvalue() for o in outs])
            preds = np.argmax(np.array(predictions), axis=1)
            rps = [int(p == c) for p, c in zip(preds, correct)] 
            rps = [j if j==1 else -1 for j in rps] 
            #print(rps) 
            #print(train['difficulty']) 
            theta_hat = calculate_theta(train['difficulty'], rps)[0] 
            #print('estimated theta: {}'.format(theta_hat))     
            # calculate the difficulty value required for such that 
            # we only include items where p_correct >= args.p_correct
            theta_hat = calculate_diff_threshold(args.p_correct, theta_hat)
        else:
            theta_hat=0

        loss = 0.0
        #print('train epoch {}'.format(i))
        epoch_training_data = get_epoch_training_data(train, args, i, 'snli', theta_hat, diffs_sorted_idx)  
        num_train_epoch = len(epoch_training_data['phrase'])
        #print('training set size: {}'.format(num_train_epoch))

        # shuffle training data
        examples = list(range(num_train_epoch))
        if args.strategy != 'ordered':
            random.shuffle(examples)
        labels = []
        predictions = []
        correct = []
        losses = []
        pairIDs = []
        outs = []
        k = 0

        for j in examples:
            if k % batch_size == 0:
                dy.renew_cg()
                losses = []
                outs = []
            sent1, sent2 = epoch_training_data['phrase'][j]
            sent1 = tokenize(sent1).split(' ')
            sent2 = tokenize(sent2).split(' ')    

            #label = train['labels'][j]
            lbl = epoch_training_data['lbls'][j]
            
            #labels.append(label)
            correct.append(epoch_training_data['lbls'][j])
            out = dnnmodel.forward(sent1, sent2, lbl)
            outs.append(out)
            loss = dy.pickneglogsoftmax(out, lbl)
            
            losses.append(loss)

            if k % batch_size == batch_size - 1:
                batch_loss = dy.esum(losses) / len(losses)
                loss_bucket = batch_loss.npvalue()  # dont need the value, just the call
                predictions.extend([o.npvalue() for o in outs])
                batch_loss.backward()
                dnnmodel.trainer.update()
                losses = []
                outs = []
            k += 1

        # need to do one last batch in case there are elements left over in a "last batch"
        # final pass if there are unbatched values
        if len(outs) > 0:
            batch_loss = dy.esum(losses) / len(losses)
            loss_bucket = batch_loss.npvalue()  # dont need the value, just the call
            predictions.extend([o.npvalue() for o in outs])
            batch_loss.backward()
            dnnmodel.trainer.update()

        preds = np.argmax(np.array(predictions), axis=1)
        acc_train = accuracy_score(correct, preds)
        #print("Training accuracy: {}, epoch: {}, num examples: {}".format(acc_train, i, len(preds)))

        # Dev
        labels = []
        predictions = []
        correct = []
        pairIDs = []
        outs = []
        k = 0
        for j in range(num_dev):
            if k % batch_size == 0:
                dy.renew_cg()
                outs = []

            sent1, sent2 = dev['phrase'][j]
            sent1 = tokenize(sent1).split(' ')
            sent2 = tokenize(sent2).split(' ')
                
            lbl = dev['lbls'][j]
            #label = dev['labels'][j]
            #labels.append(label)
            correct.append(lbl)

            out = dy.softmax(dnnmodel.forward(sent1, sent2, lbl, False))
            outs.append(out)

            if k % batch_size == batch_size - 1:
                dy.forward(outs)
                predictions.extend([o.npvalue() for o in outs])
                outs = []
            k += 1

        # final pass if there are unbatched values
        if len(outs) > 0:
            dy.forward(outs)
            predictions.extend([o.npvalue() for o in outs])

        preds = np.argmax(np.array(predictions), axis=1)
        acc_dev = accuracy_score(correct, preds)
        #print('Dev accuracy: {}'.format(acc_dev))
        
        # Test (SNLI)
        labels = []
        predictions = []
        correct = []
        pairIDs = []
        outs = []
        k = 0
        for j in range(num_test):
            if k % batch_size == 0:
                dy.renew_cg()
                outs = []

            sent1, sent2 = test['phrase'][j]
            sent1 = tokenize(sent1).split(' ')
            sent2 = tokenize(sent2).split(' ')
                
            lbl = test['lbls'][j]
            #label = test['labels'][j]
            #pairIDs.append(test['pairIDs'][j])
            #labels.append(label)
            correct.append(lbl)

            out = dy.softmax(dnnmodel.forward(sent1, sent2, lbl, False))
            outs.append(out)

            if k % batch_size == batch_size - 1:
                dy.forward(outs)
                predictions.extend([o.npvalue() for o in outs])
                outs = []
            k += 1

        # final pass if there are unbatched values
        if len(outs) > 0:
            dy.forward(outs)
            predictions.extend([o.npvalue() for o in outs])

        preds = np.argmax(np.array(predictions), axis=1)
        acc_test_snli = accuracy_score(correct, preds)
        #print('Test accuracy (SNLI): {}'.format(acc_test_snli))        
        # write test predictions to file
        #for j in range(len(predictions)):
        #    row = [i, pairIDs[j], correct[j], preds[j]]
        #    outwriter.writerow(row) 


        # write num_examples to tracker file
        progress_writer.writerow([i, num_train_epoch, acc_dev, acc_test_snli])

        if acc_dev > top_dev:
            top_dev = acc_dev
            top_dev_epoch = i
            top_dev_test = acc_test_snli

            #os.makedirs(os.path.dirname(outdir), exist_ok=True)
            with open(outdir+'preds.csv', "w") as f:
                outwriter = csv.writer(f)
                outwriter.writerow(['epoch','idx','correct','prediction'])
                for j in range(len(preds)):
                    row = [i, j, correct[j], preds[j]]
                    outwriter.writerow(row) 

            # save model to disk
            dnnmodel.m.save(outdir + 'model.pt') 
        
        #print('{},{},{},{},{},{},{}'.format(exp_label,i,num_train_epoch, acc_train, acc_dev, acc_test_snli, theta_hat))
        #print('Best so far (by dev dev): D: {}, T; {}, epoch {}'.format(top_dev, top_dev_test, top_dev_epoch))
    progress_file.close()
    

if __name__ == '__main__':
    start_time = time.time()
    run()
    end_time = time.time()
    print(end_time - start_time)



