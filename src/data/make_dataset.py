# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd 


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
#def main(input_filepath, output_filepath):
def main(data_path): 
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # for snli, write premise, hypothesis, label, and difficulty
    trainfile = 'snli_1.0_train.txt'
    devfile = 'snli_1.0_dev.txt'
    testfile_snli = 'snli_1.0_test.txt'
    #raw_data_path = '/mnt/nfs/work1/hongyu/lalor/data/cl-data/raw/'
    #processed_data_path = '/mnt/nfs/work1/hongyu/lalor/data/cl-data/processed/'
    raw_data_path = data_path + '/raw/'
    processed_data_path = data_path + '/processed/' 

    train = pd.read_csv(raw_data_path + trainfile, sep='\t',
                        usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
    dev = pd.read_csv(raw_data_path + devfile, sep='\t',
                      usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
    test = pd.read_csv(raw_data_path + testfile_snli, sep='\t',
                            usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
    train_difficulties = pd.read_csv(raw_data_path + 'snli_train_diffs.csv', sep=',',
                            header=None, names=['pairID', 'difficulty'])

    new_train = pd.merge(train, train_difficulties, on='pairID')
    new_train.to_csv(processed_data_path + 'snli_1.0_train_diff.txt', sep='\t', index=False) 


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
