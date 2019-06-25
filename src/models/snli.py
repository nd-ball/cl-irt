# lstm rnn for snli

import numpy as np
import dynet as dy
import argparse
import random
import gc
import csv
import pandas as pd

from sklearn.metrics import accuracy_score

from features.build_features import load_snli 

parser = argparse.ArgumentParser()
parser.add_argument('--dynet-autobatch', help='DyNet requirement for autobatching')
parser.add_argument('--dynet-gpus', help='DyNet requirement to trigger gpu')
parser.add_argument('--dynet-gpu', help='DyNet requirement to trigger gpu')
parser.add_argument('--dynet-mem', help='DyNet requirement to allocate memory')
parser.add_argument('--dynet-weight-decay', help='DyNet requirement for regularization')

parser.add_argument('--gpu', type=int, default=-1, help='use GPU?')
parser.add_argument('--num-units', type=int, default=300, help='number of units per layer')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--balanced', action='store_true') 
args = parser.parse_args()

print(args)

VOCAB_SIZE = 0
INPUT_DIM = 100


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
        W_tanh1 = dy.parameter(self.W_tanh1)
        W_tanh2 = dy.parameter(self.W_tanh2)
        W_tanh3 = dy.parameter(self.W_tanh3)
        b_tanh1 = dy.parameter(self.b_tanh1)
        b_tanh2 = dy.parameter(self.b_tanh2)
        b_tanh3 = dy.parameter(self.b_tanh3)
        W_out = dy.parameter(self.W_out)
        b_out = dy.parameter(self.b_out)

        t1 = dy.tanh((W_tanh1 * emb_out) + b_tanh1)
        t2 = dy.tanh((W_tanh2 * t1) + b_tanh2)
        t3 = dy.tanh((W_tanh3 * t2) + b_tanh3)

        # output
        out = (dy.transpose(W_out) * t3) + b_out
        return out


def run():

    # variables
    num_epoch = 100
    batch_size = 64
    pre_trained_embs = True
    out_dim = 3

    print('num_epoch: {}\nbatch_size: {}'.format(num_epoch, batch_size))

    train, dev, test, w2i, i2w, vectors = load_snli()

    # set up CL
    if not args.baseline:
        train_2 = {
            'lbls':[],
            'sents':[ ],
            'difficulty': []
        }
        
        diffs_sorted = np.sort(train['difficulty'])
        diffs_sorted_idx = np.argsort(train['difficulty']) 
        if not args.balanced:
            train_2['sents'] = [train['sents'][d] for d in diffs_sorted_idx] 
            train_2['lbls'] = [train['lbls'][d] for d in diffs_sorted_idx] 
            train_2['difficulty'] = [train['difficulty'][d] for d in diffs_sorted_idx]
        else:
            per_label_lists = {
                0:[], 1:[], 2:[]
            }
            for d in diffs_sorted_idx:
                eg = train['lbls'][d] 
                per_label_lists[eg].append(d) 
            zipped_egs = zip(
                per_label_lists[0], per_label_lists[1], per_label_lists[2]
            )
            train_2_idx = [j for i in zipped_egs for j in i]
            print(train_2_idx)
            train_2['sents'] = [train_2['sents'][z] for z in train_2_idx]
            train_2['lbls'] = [train_2['lbls'][z] for z in train_2_idx]
            train_2['difficulty'] = [train_2['difficulty'][z] for z in train_2_idx]
        train = train_2 

    # load model and train
    print('initialize model...')
    model = dy.Model()
    dnnmodel = RNNBuilder(model, out_dim, vectors, w2i, i2w, args.num_units)
    
    max_epoch = 0
    max_train = 0
    max_dev = 0
    max_test = 0
    num_train = len(train['sents'])
    num_dev = len(dev['sents'])
    num_test = len(test['sents'])
    top_dev = 0.0
    top_dev_epoch = 0
    top_dev_test = 0.0

    print('Training model {}'.format(model))
    print('training')
    for i in range(num_epoch):
        loss = 0.0
        print('train epoch {}'.format(i))
        # shuffle training data

        examples = list(range(num_train))
        if args.baseline:
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
            sent1, sent2 = train['sents'][j]

            label = train['labels'][j]
            lbl = train['lbls'][j]
            
            labels.append(label)
            correct.append(train['lbls'][j])
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
        print("Training accuracy: {}, epoch: {}, num examples: {}".format(acc_train, i, len(preds)))

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

            sent1, sent2 = dev['sents'][j]
            lbl = dev['lbls'][j]
            label = dev['labels'][j]
            labels.append(label)
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
        print('Dev accuracy: {}'.format(acc_dev))
        
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

            sent1, sent2 = test['sents'][j]
            lbl = test['lbls'][j]
            label = test['labels'][j]
            pairIDs.append(test['pairIDs'][j])
            labels.append(label)
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
        print('Test accuracy (SNLI): {}'.format(acc_test_snli))        
        
        if acc_dev > top_dev:
            top_dev = acc_dev
            top_dev_epoch = i
            top_dev_test = acc_test_snli

        print('Best so far (by dev dev): D: {}, T; {}, epoch {}'.format(top_dev, top_dev_test, top_dev_epoch))

        

if __name__ == '__main__':
    run()


