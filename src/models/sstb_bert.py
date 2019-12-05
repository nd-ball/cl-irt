import numpy as np
#import dynet as dy
import argparse
import random
import gc
import csv
import pandas as pd

from sklearn.metrics import accuracy_score

from features.build_features import load_sstb_bert, get_epoch_training_data, k_sort
from features.irt_scoring import calculate_theta, calculate_diff_threshold

# import required bert libraries

import torch 
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from transformers import (WEIGHTS_NAME, BertConfig,
                            BertForSequenceClassification, BertTokenizer) 
from transformers import AdamW, get_linear_schedule_with_warmup 

from transformers.data.processors import utils  

from transformers import glue_convert_examples_to_features as convert_examples_to_features



def generate_features(examples, tokenizer):
    label_list = [0, 1] 
    max_seq_len = 128
    output_mode = 'classification'
    pad_on_left=False 
    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id=0

    features = convert_examples_to_features(
        examples, tokenizer,
        label_list=label_list, 
        max_length=max_seq_len,
        output_mode=output_mode,
        pad_on_left=pad_on_left,
        pad_token=pad_token,
        pad_token_segment_id=pad_token_segment_id
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
 
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def train(args): #, outwriter): 

    # variables
    num_epoch = args.num_epochs
    batch_size = 8
    out_dim = 2
    max_grad_norm = 1.0

    device = torch.device('cuda' if args.gpu >= 0 else 'cpu') 

    config_class = BertConfig
    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer 

    config = config_class.from_pretrained('bert-base-uncased',
                                            num_labels=out_dim,
                                            cache_dir=args.cache_dir)
    tokenizer = tokenizer_class.from_pretrained(
                                            'bert-base-uncased',
                                            do_lower_case=True,
                                            cache_dir=args.cache_dir
    )
    model = model_class.from_pretrained(
                                    'bert-base-uncased',
                                    config=config,
                                    cache_dir=args.cache_dir
    )

    model.to(device) 

    #print('num_epoch: {}\nbatch_size: {}'.format(num_epoch, batch_size))
    # construct the exp label

    exp_label = 'bert_{}_{}_{}_{}'.format(args.strategy, args.balanced, args.ordering, args.random)

    train, dev, test = load_sstb_bert(args.data_dir)

    full_train_diffs = train['difficulty'] 
    full_train_examples = []
    for i in range(len(train['phrase'])):
        next_example = utils.InputExample(
            train['pairID'][i],
            train['phrase'][i],
            label=train['lbls'][i]
        )
        full_train_examples.append(next_example) 
    features_train = generate_features(full_train_examples, tokenizer)

    dev_examples = []
    for i in range(len(dev['phrase'])):
        next_example = utils.InputExample(
            dev['pairID'][i],
            dev['phrase'][i],
            label=dev['lbls'][i]
        )
        dev_examples.append(next_example) 
    features_dev = generate_features(dev_examples, tokenizer) 

    test_examples = []
    for i in range(len(test['phrase'])):
        next_example = utils.InputExample(
            test['pairID'][i],
            test['phrase'][i],
            label=test['lbls'][i]
        )
        test_examples.append(next_example) 
    features_test = generate_features(test_examples, tokenizer)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_epoch)


    if args.random:
        random.shuffle(train['difficulty'])

    # if k is set, sort once
    #if args.k > 0:
    #    diffs = train['difficulty'] 
    #    diffs_sorted_idx = k_sort(diffs, args.k) 
    #else:
    diffs_sorted_idx = None 

    
    num_train = len(train['phrase'])
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
            theta_dataloader = DataLoader(features_train, sampler=theta_sampler, batch_size=batch_size)
            preds = None 
            for batch in theta_dataloader:
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
            rps = [j if j==1 else -1 for j in rps] 
            #print(rps) 
            #print(train['difficulty']) 
            theta_hat = calculate_theta(train['difficulty'], rps, num_obs=args.num_obs)[0] 
            #print('estimated theta: {}'.format(theta_hat))     
            # calculate the difficulty value required for such that 
            # we only include items where p_correct >= args.p_correct
            theta_hat = calculate_diff_threshold(args.p_correct, theta_hat)
        else:
            theta_hat=0

        epoch_training_data = get_epoch_training_data(train, args, i, 'sstb', theta_hat, diffs_sorted_idx) 
        num_train_epoch = len(epoch_training_data['phrase'])
        #print('training set size: {}'.format(num_train_epoch))
        # shuffle training data
        # per epoch training set
        train_examples = []
        for j in range(num_train_epoch):
            next_example = utils.InputExample(
                epoch_training_data['pairID'][i],
                epoch_training_data['phrase'][j],
                label=epoch_training_data['lbls'][j]
            )
            train_examples.append(next_example) 
        features_train_epoch = generate_features(train_examples, tokenizer)

        train_sampler = RandomSampler(features_train_epoch)
        train_dataloader = DataLoader(features_train_epoch, sampler=train_sampler, batch_size=batch_size) 

        #model.zero_grad()
        for j, batch in enumerate(train_dataloader):
            model.train() 
            batch = tuple(t.to(device) for t in batch) 
            inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels': batch[3]
                    }
            #print(inputs) 
            outputs = model(**inputs) 
            loss = outputs[0]
            print(loss.item()) 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
            optimizer.step() 
            scheduler.step()
            model.zero_grad() 

        #print("Training accuracy: {}, epoch: {}, num examples: {}".format(acc_train, i, len(preds)))

        # Dev
        dev_sampler = SequentialSampler(features_dev)
        dev_dataloader = DataLoader(features_dev, sampler=dev_sampler, batch_size=batch_size)
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
        print(rps) 
        print('dev acc:{}'.format(np.mean(rps)))
        dev_acc = np.mean(rps) 
        
        # Test (SNLI)
        test_sampler = SequentialSampler(features_test)
        test_dataloader = DataLoader(features_test, sampler=test_sampler, batch_size=batch_size)
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
        print(rps) 
        print('test acc:{}'.format(np.mean(rps)))

        test_acc = np.mean(rps) 

        print('{},{},{},{},{},{}'.format(exp_label,i,num_train_epoch, dev_acc, test_acc, theta_hat))
                
        # write test predictions to file
        #for j in range(len(predictions)):
        #    row = [i, itemIDs[j], correct[j], preds[j]]
        #    outwriter.writerow(row) 

        #if acc_dev > top_dev:
        #    top_dev = acc_dev
        #    top_dev_epoch = i
        #    top_dev_test = acc_test 

        #print('Best so far (dev): {}, epoch {}'.format(top_dev, top_dev_epoch))
        #print('Best so far (test): {}, epoch {}'.format(top_dev_test, top_dev_epoch))
        
    return top_dev_test, num_train 


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='use GPU?')
    parser.add_argument('--data-dir', help='path to SNLI dataset')
    parser.add_argument('--num-units', type=int, default=300, help='number of units per layer')
    parser.add_argument('--balanced', action='store_true') 
    parser.add_argument('--strategy', choices=['baseline', 'ordered', 'simple', 'theta', 'naacl-linear', 'naacl-root', 'theta-hard'],
                        help='CL data policy', default='simple')
    parser.add_argument('--ordering', choices=['easiest', 'hardest', 'middleout'], default='easiest') 
    parser.add_argument('--num-epochs', type=int, default=100) 
    parser.add_argument('--random', action='store_true') 
    parser.add_argument('--use-length', action='store_true')
    parser.add_argument('--min-train-length', default=100, type=int)
    parser.add_argument('--k', default=0, type=int) 
    parser.add_argument('--competency', default=50, type=int) 
    parser.add_argument('--p-correct', default=0.5, type=float, help="P(correct) to filter training data for IRT")
    parser.add_argument('--cache-dir', help='cache dir for bert models')
    parser.add_argument('--num-obs', help='num obs for learning theta', default=1000)
    args = parser.parse_args()

    #preds_file = '{}processed/test_predictions/sstb_{}_{}_{}_{}_{}.csv'.format(args.data_dir, args.strategy, args.balanced, args.ordering, args.random, args.k) 
    #outfile = open(preds_file, 'w') 
    #outwriter = csv.writer(outfile, delimiter=',')
    #outwriter.writerow(['epoch', 'itemID', 'correct', 'pred'])

    #print(args)
    test_acc, training_set_size = train(args) #, outwriter)   
    #print(test_acc) 


if __name__ == '__main__':
    run()





