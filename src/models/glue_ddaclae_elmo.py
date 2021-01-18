import numpy as np
import argparse
import random
import gc
import csv
import pandas as pd

import os 
import datetime 
import time 

from sklearn.metrics import accuracy_score

from features.build_features import load_glue_task, get_epoch_training_data, k_sort, tokenize
from features.irt_scoring import calculate_theta, calculate_diff_threshold

# import required elmo libraries

import torch 
#from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.tokenizers import Token 
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance 
from allennlp.data.dataset_readers import AllennlpDataset 
from allennlp.data.dataloader import PyTorchDataLoader 
from allennlp.data.vocabulary import Vocabulary 
from allennlp.data.samplers import SequentialSampler, RandomSampler 
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer 

from transformers import (WEIGHTS_NAME, BertConfig,
                            BertForSequenceClassification, BertTokenizer) 
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule 
from transformers.data.processors import utils  
from transformers import glue_convert_examples_to_features as convert_examples_to_features


options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo_token_indexer = ELMoTokenCharactersIndexer()

class CLF(torch.nn.Module):
    def __init__(self, num_classes, num_sentences=2):
        super(CLF, self).__init__()
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.lstm_layer = 100
        self.num_classes = num_classes
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0.5)
        self.lstm = torch.nn.LSTM(1024, self.lstm_layer, 1)
        self.tanh = torch.nn.Tanh() 
        self.num_sentences = num_sentences
        self.linear_layer = self.lstm_layer * self.num_sentences 
        self.softmax = torch.nn.Softmax()

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(self.linear_layer, self.linear_layer),
            self.tanh,
            torch.nn.Linear(self.linear_layer, self.linear_layer),
            self.tanh,
            torch.nn.Linear(self.linear_layer, self.linear_layer),
            self.tanh,
            torch.nn.Linear(self.linear_layer, self.num_classes),
            self.softmax
        )

    def forward(self, s1, s2=None):
        character_ids_s1 = batch_to_ids(s1['tokens']['elmo_tokens'])
        s1_emb = self.elmo(character_ids_s1)['elmo_representations'][0]
        if s2 is not None:
            character_ids_s2 = batch_to_ids(s2['tokens']['elmo_tokens'])
            s2_emb = self.elmo(character_ids_s2)['elmo_representations'][0] 
            lstm_s1, _ = self.lstm(s1_emb)
            lstm_s2, _ = self.lstm(s2_emb)
            print(lstm_s2.shape)
            embs = torch.cat((lstm_s1, lstm_s2), 2)
        else:
            embs, _ = self.lstm(s1_emb)
        print(embs)
        return self.feed_forward(embs) 
            


def text_to_instance(examples):

    dataset = []
    for i in range(len(examples)):
        fields = {}
        fields['t1'] = TextField(
            [Token(w) for w in tokenize(examples['phrase'][i][0]).split(' ')],
            token_indexers={'tokens': elmo_token_indexer}
        )

        try:
            fields['t2'] = TextField(
                [Token(w) for w in tokenize(examples['phrase'][i][1]).split(' ')],
                token_indexers={'tokens': elmo_token_indexer}
            )
        except:
            fields['t2'] = None  # single sent
        fields['label'] = LabelField(examples['lbls'][i])
        dataset.append(Instance(fields)) 
    
    dataset = AllennlpDataset(dataset)
    return dataset


def train(args, outfile): 

    # variables
    num_epoch = args.num_epochs
    batch_size = 8
    max_grad_norm = 1.0

    device = torch.device('cuda' if args.gpu >= 0 else 'cpu') 

    train, dev, test = load_glue_task(args.data_dir, args.diff_dir, args.task)
    label_list = list(set(train['lbls']))
    out_dim = len(label_list)
    

    if args.task == 'SSTB':
        num_sent = 1
    else:
        num_sent = 2
    num_classes = len(set(train['lbls']))
    model = CLF(num_classes, num_sent)
    model.to(device) 

    # construct the exp label

    exp_label = 'elmo_{}_{}_{}_{}'.format(args.strategy, args.balanced, args.ordering, args.random)

    # save training data set size to disk for bookkeeping
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    progress_file = open(outfile + 'tracker.csv', 'w')
    progress_writer = csv.writer(progress_file)
    progress_writer.writerow(["epoch","num_training_examples", "dev_acc", "test_acc"])


    full_train_diffs = train['difficulty'] 
    full_train_examples = []
    try:
        single_sentence = np.isnan(train['phrase'][0][1]) 
    except:
        single_sentence = False  # isnan will throw an error if type is str
    print(single_sentence)
    full_train_examples = text_to_instance(train)
    vocab = Vocabulary.from_instances(full_train_examples) 
    full_train_examples = AllennlpDataset(full_train_examples, vocab=vocab)

    if args.num_obs < len(full_train_examples):
        theta_sample = np.random.randint(0, len(full_train_examples), args.num_obs) 
        theta_diffs = [full_train_diffs[z] for z in theta_sample]
        theta_train = [full_train_examples[z] for z in theta_sample]
        features_train = theta_train
    else:
        features_train = full_train_examples
        theta_diffs = full_train_diffs 

    dev_examples = text_to_instance(dev)
    features_dev = AllennlpDataset(dev_examples, vocab=vocab) 
    
    test_examples = text_to_instance(test)
    features_test = AllennlpDataset(test_examples, vocab=vocab)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, correct_bias=False)
    scheduler = get_constant_schedule(optimizer) 

    if args.random:
        random.shuffle(train['difficulty'])

    # if k is set, sort once
    #if args.k > 0:
    #    diffs = train['difficulty'] 
    #    diffs_sorted_idx = k_sort(diffs, args.k) 
    #else:
    diffs_sorted_idx = None 

    num_train = len(train['phrase'])
    top_dev = 0.0
    top_dev_test = 0.0

    print('Training model {}'.format(model))
    print('training')
    model.zero_grad()
    for i in range(num_epoch):
        loss = 0.0
        #print('train epoch {}'.format(i))
        # load training data for this epoch

        # estimate theta_hat 
        if args.strategy in ['theta', 'theta-hat']:
            theta_sampler = SequentialSampler(features_train)
            theta_dataloader = PyTorchDataLoader(features_train, sampler=theta_sampler, batch_size=batch_size)
            preds = None 
            for batch in theta_dataloader:
                model.eval() 
                batch = tuple(t.to(device) for t in batch) 
                with torch.no_grad():
                    outputs = model(batch['t1'], batch['t2']) 
                    tmp_eval_loss, logits = outputs[:2]

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = batch['lbls'].detach().cpu().numpy() 
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0) 
                    out_label_ids = np.append(out_label_ids, batch['lbls'].detach().cpu().numpy(), axis=0) 

            preds = np.argmax(preds, axis=1) 
            
            rps = [int(p == c) for p, c in zip(preds, out_label_ids)] 
            rps = [j if j==1 else -1 for j in rps] 
            theta_hat = calculate_theta(theta_diffs, rps)[0] 
            # calculate the difficulty value required for such that 
            # we only include items where p_correct >= args.p_correct
            theta_hat = calculate_diff_threshold(args.p_correct, theta_hat)
        else:
            theta_hat=0

        epoch_training_data = get_epoch_training_data(train, args, i, 'glue', theta_hat, diffs_sorted_idx) 
        num_train_epoch = len(epoch_training_data['phrase'])

        # shuffle training data
        # per epoch training set
        features_train_epoch = text_to_instance(epoch_training_data)
        features_train_epoch = AllennlpDataset(features_train_epoch, vocab=vocab)
        features_train_epoch.index_with(vocab)
        train_sampler = RandomSampler(features_train_epoch)
        train_dataloader = PyTorchDataLoader(features_train_epoch, sampler=train_sampler, batch_size=batch_size) 

        #model.zero_grad()
        model.train() 
        for j, batch in enumerate(train_dataloader):
            outputs = model(batch['t1'], batch['t2']) 
            print(outputs)
            print(batch['label'])
            loss = criterion(outputs, batch['label'])
            
            loss.backward() 
            optimizer.step() 
            scheduler.step()
            model.zero_grad() 

        #print("Training accuracy: {}, epoch: {}, num examples: {}".format(acc_train, i, len(preds)))

        # Dev
        dev_sampler = SequentialSampler(features_dev)
        dev_dataloader = PyTorchDataLoader(features_dev, sampler=dev_sampler, batch_size=batch_size)
        preds = None 
        dev_loss = 0
        for batch in dev_dataloader:
            model.eval() 
            batch = tuple(t.to(device) for t in batch) 
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3]
                }
                outputs = model(**inputs) 
                tmp_eval_loss, logits = outputs[:2]
                dev_loss += tmp_eval_loss.item()

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy() 
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0) 
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0) 

        preds = np.argmax(preds, axis=1) 
        print('dev loss:{}'.format(dev_loss))
        
        rps = [int(p == c) for p, c in zip(preds, out_label_ids)] 
        print('dev acc:{}'.format(np.mean(rps)))
        dev_acc = np.mean(rps) 
        
        # Test 
        test_sampler = SequentialSampler(features_test)
        test_dataloader = PyTorchDataLoader(features_test, sampler=test_sampler, batch_size=batch_size)
        preds = None 
        for batch in test_dataloader:
            model.eval() 
            batch = tuple(t.to(device) for t in batch) 
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3]
                }
                outputs = model(**inputs) 
                tmp_eval_loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy() 
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0) 
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0) 

        preds = np.argmax(preds, axis=1) 
        
        rps = [int(p == c) for p, c in zip(preds, out_label_ids)] 
        print('test acc:{}'.format(np.mean(rps)))
        
        test_acc = np.mean(rps) 
        print('{},{},{},{},{},{}'.format(exp_label,i,num_train_epoch, dev_acc, test_acc, theta_hat))

        # write num_examples to tracker file
        progress_writer.writerow([i, num_train_epoch, dev_acc, test_acc])

        # write test predictions to file
        if dev_acc > top_dev:
            top_dev = dev_acc
            top_dev_epoch = i
            top_dev_test = test_acc 
            #os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile+'preds.csv', "w") as f:
                outwriter = csv.writer(f)
                outwriter.writerow(['epoch','idx','correct','prediction'])
                for j in range(len(preds)):
                    row = [i, j, out_label_ids[j], preds[j]]
                    outwriter.writerow(row) 

            # save model to disk
            torch.save(model.state_dict(), outfile + 'model.pt') 


        print('Best so far (dev): {}, epoch {}'.format(top_dev, top_dev_epoch))
        print('Best so far (test): {}, epoch {}'.format(top_dev_test, top_dev_epoch))

    progress_file.close()
        
    return top_dev_test, num_train 


def run():
    GLUETASKS = ['CoLA', 'SST-2', 'MRPC', 'MNLI', 'QNLI', 'RTE', 'WNLI', 'QQP']
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='use GPU?')
    parser.add_argument('--data-dir', help='path to SNLI dataset')
    parser.add_argument('--diff-dir', help='path to SNLI dataset')
    parser.add_argument('--num-units', type=int, default=300, help='number of units per layer')
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
    parser.add_argument('--cache-dir', help='cache dir for bert models')
    parser.add_argument('--num-obs', help='num obs for learning theta', default=1000, type=int)
    parser.add_argument('--task', choices=GLUETASKS, help='GLUE task for fine-tuning')
    args = parser.parse_args()

    # create output directory-file
    outdir = 'results/bert-{}/{}-{}-len-{}-wordrarity-{}/{}/'.format(
        args.balanced,
        args.task,
        args.strategy,
        args.use_length,
        args.use_word_rarity,
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )

    #print(args)
    start_time = time.time()
    test_acc, training_set_size = train(args, outdir)   
    end_time = time.time()
    print(end_time - start_time)
    #print(test_acc) 


if __name__ == '__main__':
    run()


