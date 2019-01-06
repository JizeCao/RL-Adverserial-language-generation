import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminator import batch2TrainData
import time
import numpy as np
from torch import optim
import pickle
import os
import argparse
from model import hierEncoder_frequency_batchwise, hierEncoder_frequency
from search_utils import logging, Voc

class critic(nn.Module):
    def __init__(self, vocab_size, embedding_size):

        super(critic, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = self.embedding_size  # the hidden size of GRU equals embedding size by default

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size)
        self.gru2 = nn.GRU(self.embedding_size, 128)
        self.linear1 = nn.Linear(128, 32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self, pair=None, sources=None, targets=None, sources_length=None, targets_length=None, targets_order=None, to_device=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if pair is not None:
            # pair为对话 {x, y} 类型为torch.tensor()
            x_length = pair[0].size(0)
            y_length = pair[1].size(0)

            if to_device:
                hidden = self.initHidden().to(device)
            else:
                hidden = self.initHidden()

            for i in range(x_length):
                embedded_x = self.embedding(pair[0][i]).view(1, 1, -1)
                _, hidden = self.gru1(embedded_x, hidden)
            hidden_x = hidden  # x句的编码结果

            if to_device:
                hidden = self.initHidden().to(device)
            else:
                hidden = self.initHidden()

            for i in range(y_length):
                embedded_y = self.embedding(pair[1][i]).view(1, 1, -1)
                _, hidden = self.gru1(embedded_y, hidden)
            hidden_y = hidden  # y句的编码结果

            if to_device:
                hidden = torch.zeros(1, 1, 128).to(device)
            else:
                hidden = torch.zeros(1, 1, 128)

            _, hidden = self.gru2(hidden_x, hidden)
            _, hidden = self.gru2(hidden_y, hidden)
            hidden_xy = hidden  # 得到{x，y}编码结果

            max_frequency = self.count_max_frequency(pair[1]).to(device)
            output = F.relu(self.linear1(hidden_xy.squeeze()))

            output = self.linear2(output)


            return output

        sources = sources.to(device)
        targets = targets.to(device)
        # pair为对话 {x, y} 类型为torch.tensor()
        embedded_sources = self.embedding(sources)
        embedded_targets = self.embedding(targets)


        packed_sources = torch.nn.utils.rnn.pack_padded_sequence(embedded_sources, sources_length)
        packed_targets = torch.nn.utils.rnn.pack_padded_sequence(embedded_targets, targets_length)

        # Source level GRU
        _, hidden_sources = self.gru1(packed_sources)
        _, hidden_targets = self.gru1(packed_targets)

        # Change the hidden state to correct order
        hidden_targets = hidden_targets[:, targets_order, :]
        # Sentence level GRU, Check whether the dimension is correct!
        _, hidden = self.gru2(hidden_sources, None)
        _, hidden = self.gru2(hidden_targets, hidden)

        output = F.relu(self.linear1(hidden.squeeze()))

        output = self.linear2(output)

        return output

def generate_reward(pos_data, neg_data, modelD, vocab, batch_size=512):

    # prepare data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pos_data_batches = [batch2TrainData(vocab, pos_data[i * batch_size: i * batch_size + batch_size]) for i in
                        range(len(pos_data) // batch_size)]
    neg_data_batches = [batch2TrainData(vocab, neg_data[i * batch_size: i * batch_size + batch_size]) for i in
                        range(len(neg_data) // batch_size)]

    data_batches = pos_data_batches + neg_data_batches

    loss = torch.Tensor([0]).to(device)
    start_time = time.time()
    criterion = nn.NLLLoss()
    missclassification = 0
    num_sen = 0

    rewards = []

    with torch.no_grad():

        for batch in data_batches:
            input_data, input_length, output_data, output_length, output_order, input_order = batch

            #  Get the order that is used to retrive the original order
            retrive_order = np.argsort(output_order)

            output = modelD(sources=input_data, targets=output_data, sources_length=input_length,
                            targets_length=output_length, targets_order=retrive_order)

            batch_rewards = torch.exp(output[:, 0])
            rewards.append(batch_rewards)

    return rewards, data_batches


def critic_train(critic_model, modelD, pos_train, neg_train, pos_valid, neg_valid, EOS_token, vocab, learning_rate=0.001, batch_size=128,
                 n_iteration=5000, to_device=True):

    critic_model.train()
    critic_optimizer = optim.SGD(critic_model.parameters(), lr=learning_rate, momentum=0.8)
    modelD.eval()

    # prepare data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    total_loss = torch.Tensor([0]).to(device)
    start_time = time.time()
    criterion = nn.NLLLoss()


    rewards, data_batches = generate_reward(pos_train, neg_train, modelD, vocab, batch_size=batch_size)

    permute = np.random.permutation(len(data_batches))
    data_batches = [data_batches[i] for i in permute]
    rewards = [rewards[i] for i in permute]
    num_batch = 0
    val_loss = 1000000000
    best_iteration = 0

    try:

        for iteration in n_iteration:
            for batch in data_batches:

                reward = rewards[num_batch]

                input_data, input_length, output_data, output_length, output_order, input_order = batch

                #  Get the order that is used to retrive the original order
                retrive_order = np.argsort(output_order)

                output = critic_model(sources=input_data, targets=output_data, sources_length=input_length,
                                targets_length=output_length, targets_order=retrive_order)

                loss = criterion(output, reward)
                total_loss += loss

                loss.backward()
                critic_optimizer.step()

                num_batch += 1

                print('The train loss for iteration {}, at batches {} / {}'.format(iteration, num_batch, len(data_batches)))

            curr_val_loss = critic_evaluation(critic_model, modelD, pos_valid, neg_valid, EOS_token, vocab, batch_size=512)

            if curr_val_loss < val_loss:
                best_iteration = iteration
                torch.save({
                    'iteration': iteration,
                    'val_loss': val_loss,
                    'disc': critic_model.state_dict()
                }, 'critic_model_' + str(iteration) + '.pt')
    except KeyboardInterrupt:
        print('The best val_loss is {:2f}, at iterations {}'.format(val_loss, best_iteration))


def critic_evaluation(critic_model, modelD, pos_valid, neg_valid, EOS_token, vocab, batch_size=512):

    critic_model.eval()
    total_loss = 0
    criterion =  nn.NLLLoss()

    rewards, data_batches = generate_reward(pos_valid, neg_valid, modelD, vocab, batch_size=batch_size)


    with torch.no_grad():

        batch_index = 0
        for batch in data_batches:

            reward = rewards[batch_index]
            input_data, input_length, output_data, output_length, output_order, input_order = batch

            #  Get the order that is used to retrive the original order
            retrive_order = np.argsort(output_order)

            output = critic_model(sources=input_data, targets=output_data, sources_length=input_length,
                                  targets_length=output_length, targets_order=retrive_order)

            loss = criterion(output, reward)
            total_loss += loss
            batch_index += 1

    print('Average MSE loss', str(total_loss / len(data_batches)))

    return total_loss / len(data_batches)

def load_data(args):
    #voc = pickle.load(open(os.path.join(args.save_dir, 'processed_voc.p'), 'rb'))
    #train_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'processed_train_sen_2000000.p'), 'rb'))
    #valid_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'processed_valid_sen_2000000.p'), 'rb'))
    #neg_train_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_processed_train.p'), 'rb'))
    #neg_valid_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_processed_valid.p'), 'rb'))

    voc = pickle.load(open(os.path.join(args.save_dir, 'whole_data_voc.p'), 'rb'))
    train_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'small_train_2000000.p'), 'rb'))
    valid_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'small_valid_2000000.p'), 'rb'))
    neg_train_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_train.p'), 'rb'))
    neg_valid_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_valid.p'), 'rb'))

    model_save_dir = os.path.join(args.save_dir, 'cb_model/Open_subtitles/3-3_512')

    modelD = hierEncoder_frequency_batchwise(len(voc.index2word), 500)
    #modelD = hierEncoder_frequency(len(voc.index2word), 500)

    if args.cuda:
        dis_checkpoint = torch.load(os.path.join(model_save_dir, 'disc_params_beam_frquency.pt'))
        modelD.cuda()
    else:
        dis_checkpoint = torch.load(os.path.join(model_save_dir, 'disc_params_beam_frquency.pt'),
                                    map_location=lambda storage, loc:storage)

    modelD.load_state_dict(dis_checkpoint['disc'])

    return voc, modelD, train_pos_pairs, valid_pos_pairs, neg_train_pairs, neg_valid_pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UCT based on the language model')
    parser.add_argument('--save_dir', type=str, default='./data/save',
                        help='directory of the save place')
    parser.add_argument('--retrain', action='store_true',
                        help='retrain from an existing checkpoint')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda')

    args = parser.parse_args()

    voc, modelD, train_pos_pairs, valid_pos_pairs, neg_train_pairs, neg_valid_pairs = load_data(args)

    EOS_token = voc.word2index['<EOS>']

    critic_model = critic(len(voc.index2word), 500)

    critic_train(critic_model, modelD, train_pos_pairs, neg_train_pairs, valid_pos_pairs, neg_valid_pairs, EOS_token, voc)











