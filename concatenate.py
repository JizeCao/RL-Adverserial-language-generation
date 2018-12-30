import pickle
import argparse
import os

parser = argparse.ArgumentParser(description='concatenate files')
parser.add_argument('--save_dir', type=str, default='.')
parser.add_argument('--RL_index', type=int, default=341, help="RL index")
parser.add_argument('--train_batch_size', type=int, default=20000, help="train batch_size")
parser.add_argument('--val_batch_size', type=int, default=2000, help="valid batch_size")
parser.add_argument('--train_sen_length', type=int, default=120000, help="train_length")
parser.add_argument('--valid_sen_length', type=int, default=12000, help="valid_length")
parser.add_argument('--train_begin', type=int, default=0, help="begin of the train")
parser.add_argument('--valid_begin', type=int, default=0, help="begin of the val")

args = parser.parse_args()


def concatenate_files(prefix, begin, batch_size, total_length, args):
    text = pickle.load(open(os.path.join(args.save_dir, prefix + '_' + str(begin) + '_' + str(begin + batch_size)), 'rb'))

    for i in range(begin + batch_size, total_length, batch_size):
        text += pickle.load(open(os.path.join(args.save_dir, prefix + '_' + str(i) + '_' + str(i + batch_size)), 'rb'))

    pickle.dump(text, open(os.path.join(args.save_dir, prefix), 'wb'))


prefix_train = 'Generated_sentences_' + str(args.RL_index)
prefix_valid = 'Generated_sentences_valid_' + str(args.RL_index)

concatenate_files(prefix_train, args.train_begin, args.train_batch_size, args.train_sen_length, args)
concatenate_files(prefix_valid, args.valid_begin, args.val_batch_size, args.valid_sen_length, args)
