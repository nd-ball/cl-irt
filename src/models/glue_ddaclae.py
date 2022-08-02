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

from features.build_features import load_glue_task, get_epoch_training_data, k_sort
from features.irt_scoring import calculate_theta, calculate_diff_threshold

# import required bert libraries

import torch 
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import (WEIGHTS_NAME, BertConfig,
                            BertForSequenceClassification, BertTokenizer) 
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule 
from transformers.data.processors import utils  
from transformers import glue_convert_examples_to_features as convert_examples_to_features


def generate_features(examples, tokenizer, label_list):
    max_seq_len = 128
    output_mode = 'classification'
    pad_on_left=False 
    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id=0

    features = convert_examples_to_features(
        examples, tokenizer,
        label_list=label_list, 
        max_length=max_seq_len,
        output_mode=output_mode
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
 
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def train(args, outfile): 

    # variables
    num_epoch = args.num_epochs
    batch_size = 8
    max_grad_norm = 1.0

    device = torch.device('cuda' if args.gpu >= 0 else 'cpu') 

    config_class = BertConfig
    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer 

    train, dev, test = load_glue_task(args.data_dir, args.diff_dir, args.task)
    label_list = list(set(train['lbls']))
    out_dim = len(label_list)
    

    config = config_class.from_pretrained('bert-large-uncased',
                                            num_labels=out_dim,
                                            cache_dir=args.cache_dir)
    tokenizer = tokenizer_class.from_pretrained(
                                            'bert-large-uncased',
                                            do_lower_case=True,
                                            cache_dir=args.cache_dir
    )
    model = model_class.from_pretrained(
                                    'bert-large-uncased',
                                    config=config,
                                    cache_dir=args.cache_dir
    )

    model.to(device) 

    #print('num_epoch: {}\nbatch_size: {}'.format(num_epoch, batch_size))
    # construct the exp label

    exp_label = 'bert_large_{}_{}_{}_{}_{}_{}'.format(args.strategy, args.balanced, args.ordering, args.random, args.lower_bound, args.upper_bound)

    # save training data set size to disk for bookkeeping
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    progress_file = open(outfile + 'tracker.csv', 'w')
    progress_writer = csv.writer(progress_file)
    progress_writer.writerow(["epoch","num_training_examples", "dev_acc", "test_acc","theta_hat"])


    full_train_diffs = train['difficulty'] 
    full_train_examples = []
    try:
        single_sentence = np.isnan(train['phrase'][0][1]) 
    except:
        single_sentence = False  # isnan will throw an error if type is str
    print(single_sentence)
    for i in range(len(train['phrase'])):
        if single_sentence:
            next_example = utils.InputExample(
                train['pairID'][i],
                train['phrase'][i][0],
                label=train['lbls'][i]
            )
        else:
            next_example = utils.InputExample(
                train['pairID'][i],
                train['phrase'][i][0],
                train['phrase'][i][1],
                train['lbls'][i]
            )
        full_train_examples.append(next_example) 

    if args.num_obs < len(full_train_examples):
        theta_sample = np.random.randint(0, len(full_train_examples), args.num_obs) 
        theta_diffs = [full_train_diffs[z] for z in theta_sample]
        theta_train = [full_train_examples[z] for z in theta_sample]
        features_train = generate_features(theta_train, tokenizer, label_list)
    else:
        features_train = generate_features(full_train_examples, tokenizer, label_list) 
        theta_diffs = full_train_diffs 

    dev_examples = []
    for i in range(len(dev['phrase'])):
        if single_sentence:
            next_example = utils.InputExample(
                dev['pairID'][i],
                dev['phrase'][i][0],
                label=dev['lbls'][i]
            )
        else:
            next_example = utils.InputExample(
                dev['pairID'][i],
                dev['phrase'][i][0],
                dev['phrase'][i][1],
                dev['lbls'][i]
            )

        dev_examples.append(next_example) 
    features_dev = generate_features(dev_examples, tokenizer, label_list) 
    
    test_examples = []
    for i in range(len(test['phrase'])):
        if single_sentence:
            next_example = utils.InputExample(
                test['pairID'][i],
                test['phrase'][i][0],
                label=test['lbls'][i]
            )
        else:
            next_example = utils.InputExample(
                test['pairID'][i],
                test['phrase'][i][0],
                test['phrase'][i][1],
                test['lbls'][i]
            )
        test_examples.append(next_example) 
    features_test = generate_features(test_examples, tokenizer, label_list)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, correct_bias=False)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_epoch)
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
        if args.strategy in ['theta', 'theta-hat','baseline']:
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
            theta_hat = calculate_theta(theta_diffs, rps)[0] 
            # calculate the difficulty value required for such that 
            # we only include items where p_correct >= args.p_correct
            theta_hat = calculate_diff_threshold(args.p_correct, theta_hat)
        else:
            theta_hat=0

        epoch_training_data = get_epoch_training_data(train, args, i, 'glue', theta_hat, diffs_sorted_idx, lower_offset=args.lower_bound, upper_offset=args.upper_bound) 
        num_train_epoch = len(epoch_training_data['phrase'])
        #print('training set size: {}'.format(num_train_epoch))
        # shuffle training data
        # per epoch training set
        train_examples = []
        for j in range(num_train_epoch):
            if single_sentence:
                next_example = utils.InputExample(
                    epoch_training_data['pairID'][j],
                    epoch_training_data['phrase'][j][0],
                    label=epoch_training_data['lbls'][j]
                )
            else:
                next_example = utils.InputExample(
                    epoch_training_data['pairID'][j],
                    epoch_training_data['phrase'][j][0],
                    epoch_training_data['phrase'][j][1],
                    epoch_training_data['lbls'][j]
                )
            train_examples.append(next_example) 
        features_train_epoch = generate_features(train_examples, tokenizer, label_list)

        train_sampler = RandomSampler(features_train_epoch)
        train_dataloader = DataLoader(features_train_epoch, sampler=train_sampler, batch_size=batch_size) 

        #model.zero_grad()
        model.train() 
        for j, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch) 
            inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels': batch[3]
                    }
            outputs = model(**inputs) 
            loss = outputs[0]
            loss.backward() 
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
        print('dev acc:{}'.format(np.mean(rps)))
        dev_acc = np.mean(rps) 
        
        # Test 
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
        print('test acc:{}'.format(np.mean(rps)))
        
        test_acc = np.mean(rps) 
        print('{},{},{},{},{},{}'.format(exp_label,i,num_train_epoch, dev_acc, test_acc, theta_hat))

        # write num_examples to tracker file
        progress_writer.writerow([i, num_train_epoch, dev_acc, test_acc, theta_hat])

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
    parser.add_argument('--lower-bound', default=np.NINF, type=float)
    parser.add_argument('--upper-bound', default=0, type=float)
    args = parser.parse_args()

    # create output directory-file
    outdir = 'results/bert-large-{}/{}-{}-len-{}-wordrarity-{}-{}-{}/{}/'.format(
        args.balanced,
        args.task,
        args.strategy,
        args.use_length,
        args.use_word_rarity,
        args.lower_bound,
        args.upper_bound,
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


