import pickle
import os
from search_utils import Voc
import argparse

parser = argparse.ArgumentParser(description='UCT based on the language model')
parser.add_argument('--save_dir', type=str, default='../data/save',
                    help='directory of the save place')

args = parser.parse_args()

def indexing(pair_batch, EOS_token):

    new_batch = []
    for pair in pair_batch:
        temp = []
        for sen in pair:
            if sen[-1] != EOS_token:
                sen.append(EOS_token)
            temp.append(sen)
        new_batch.append(temp)


    return new_batch

def load_data(args):
    voc = pickle.load(open(os.path.join(args.save_dir, 'processed_voc.p'), 'rb'))
    train_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'processed_train_sen_2000000.p'), 'rb'))
    valid_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'processed_valid_sen_2000000.p'), 'rb'))
    neg_train_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_processed_train.p'), 'rb'))
    neg_valid_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_processed_valid.p'), 'rb'))

    return voc, train_pos_pairs, valid_pos_pairs, neg_train_pairs, neg_valid_pairs

voc, train_pos_pairs, valid_pos_pairs, neg_train_pairs, neg_valid_pairs = load_data(args)
EOS_token = voc.word2index['<EOS>']


train_pos_pairs = indexing(train_pos_pairs, EOS_token)
valid_pos_pairs = indexing(valid_pos_pairs, EOS_token)
neg_train_pairs = indexing(neg_train_pairs, EOS_token)
neg_valid_pairs = indexing(neg_valid_pairs, EOS_token)

pickle.dump(train_pos_pairs, open('indexed_processed_train_sen_2000000.p', 'wb'))
pickle.dump(valid_pos_pairs, open('indexed_processed_valid_sen_2000000.p', 'wb'))
pickle.dump(neg_train_pairs, open('indexed_neg_train_sen_2000000.p', 'wb'))
pickle.dump(neg_valid_pairs, open('indexed_neg_valid_sen_2000000.p', 'wb'))