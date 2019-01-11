from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import pickle
import numpy as np
from model import LuongAttnDecoderRNN, EncoderRNN
from search_utils import Voc
import argparse

def indexesFromSentence(voc, sentence):
    EOS_token = voc.word2index['<EOS>']
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# Do zero padding given a list of either source or target sens
# Return: a list contains #max_length of lists, each list contain #batch_size elements
def zeroPadding(l, fillvalue):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    # indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    indexes_batch = l
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, voc.word2index['<PAD>'])
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = l
    #indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, voc.word2index['<PAD>'])
    mask = binaryMatrix(padList, voc.word2index['<PAD>'])
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    # pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


def beam_search(decoder_hidden, beam_search_k, decoder, max_length, decoder_input, encoder_outputs, EOS_Token,
                to_device=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 用来记录概率最大的k个选择的隐藏层, type: [torch.tensor()]
    hidden_log = [decoder_hidden for _ in range(beam_search_k)]

    # 用来记录最大的k个概率, type: [float]
    prob_log = [0 for _ in range(beam_search_k)]

    # 用来记录概率最大的k个选择的解码输出, type: [[int]]
    decoder_outputs = np.empty([beam_search_k, 1]).tolist()

    # 先进行第一步解码
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden,
                                                                  encoder_outputs)
    # 选择概率最大的k个选项
    topv, topi = decoder_output.topk(beam_search_k)

    for k in range(beam_search_k):
        # 记录隐藏层, type: [torch.tensor()]
        hidden_log[k] = decoder_hidden

        # 记录概率（默认降序排列）, type: [float]
        prob_log[k] += topv.squeeze()[k].item()

        # 记录输出（与prob_log的概率对应）, type: [int]
        decoder_outputs[k].append(topi.squeeze()[k].item())
        decoder_outputs[k].pop(0)  # 删除初始化时存入的元素

    # beam search
    for ei in range(max_length - 1):
        # 用以暂时存储概率在后续进行比较
        if to_device:
            temp_prob_log = torch.tensor([]).to(device)
            temp_output_log = torch.tensor([], dtype=torch.long).to(device)
            temp_hidden_log = []
        else:
            temp_prob_log = torch.tensor([])
            temp_hidden_log = []
            temp_output_log = torch.tensor([], dtype=torch.long)

        for k in range(beam_search_k):
            if to_device:
                decoder_input = torch.tensor([decoder_outputs[k][-1]], dtype=torch.long).to(device)

            else:
                decoder_input = torch.tensor([decoder_outputs[k][-1]], dtype=torch.long)

            if decoder_input.item() != EOS_Token:
                decoder_hidden = hidden_log[k]
                decoder_output, decoder_hidden = decoder(decoder_input.unsqueeze(0), decoder_hidden,
                                                              encoder_outputs)
                # 初步比较
                topv, topi = decoder_output.topk(beam_search_k)
                topv += prob_log[k]

                temp_prob_log = torch.cat([temp_prob_log, topv], dim=1)
                temp_hidden_log.append(decoder_hidden)
                temp_output_log = torch.cat([temp_output_log, topi], dim=1)

            else:  # 如果已达到 <EOS>
                if to_device:
                    topv = torch.ones(1, beam_search_k).to(device) * prob_log[k]
                    topi = torch.ones(1, beam_search_k).to(device) * EOS_Token
                    # Index should be long
                    topi = topi.long()
                else:
                    topv = torch.ones(1, beam_search_k) * prob_log[k]
                    topi = torch.ones(1, beam_search_k, dtype=torch.long) * EOS_Token

                temp_prob_log = torch.cat([temp_prob_log, topv], dim=1)
                temp_hidden_log.append(None)
                temp_output_log = torch.cat([temp_output_log, topi], dim=1)

        # 最终比较（在 k*k 个候选项中选出概率最大的 k 个选项）
        temp_topv, temp_topi = temp_prob_log.topk(beam_search_k)

        temp_decoder_outputs = decoder_outputs.copy()

        # 记录结果(保持概率降序排列)
        for k in range(beam_search_k):
            ith = int(temp_topi.squeeze()[k].item() / beam_search_k)

            hidden_log[k] = temp_hidden_log[ith]

            prob_log[k] = temp_topv.squeeze()[k].item()

            decoder_outputs[k] = temp_decoder_outputs[ith].copy()
            if temp_output_log.squeeze()[temp_topi.squeeze()[k].item()].item() != EOS_Token \
                    and decoder_outputs[k][-1] != EOS_Token:
                decoder_outputs[k].append(temp_output_log.squeeze()[temp_topi.squeeze()[k].item()].item())

    # Optimistic return
    return decoder_outputs[0]


def evaluate(encoder, decoder, voc, sentence, beam=10, max_length=20):

    # generate sentence per source
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_token, EOS_token, PAD_token = voc.word2index['<SOS>'], voc.word2index['<EOS>'], voc.word2index['<PAD>']
    ### Format input sentence as a batch
    # words -> indexes
    # indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(sentence) + 1])
    # Transpose dimensions of batch to match models' expectations
    sentence.append(EOS_token)
    input_batch = torch.LongTensor([sentence]).transpose(0, 1)

    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)

    # Forward input through encoder model
    encoder_outputs, encoder_hidden = encoder(input_batch, lengths)
    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Initialize decoder input with SOS_token
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

    # Only return the beam with highest last probability
    decoder_word = beam_search(decoder_hidden, beam, decoder, max_length, decoder_input, encoder_outputs, EOS_token)


    # Decode sentence with searcher
    #tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    #decoded_words = [voc.index2word[token.item()] for token in tokens]

    return decoder_word


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference Pre-train sentences')
    parser.add_argument('--division', type=int, default=2,
                    help="the number of divisions one used to generate text")
    parser.add_argument('--partition', type=int, default=0,
                        help="the partion used to generate")
    parser.add_argument('--beam_size', type=int, default=5,
                        help="the beam size")

    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    corpus_name = 'Open_subtitles'
    corpus = os.path.join('data', corpus_name)

    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 512
    encoder_n_layers = 3
    decoder_n_layers = 3
    dropout = 0.2
    batch_size = 128
    clip = 5.0
    teacher_forcing_ratio = 0.7
    learning_rate = 1
    decoder_learning_ratio = 1.0
    n_iteration = 4000
    print_every = 1
    save_every = 500
    epoch = 100
    weight_decay = 0.00001


    # Load/Assemble voc and pairs
    save_dir = os.path.join("data", "save")
    voc_dir = os.path.join(save_dir, 'whole_data_voc.p')
    train_data_dir = os.path.join(save_dir, 'small_train_2000000.p')
    valid_data_dir = os.path.join(save_dir, 'small_valid_2000000.p')


    voc = pickle.load(open(voc_dir, 'rb'))

    pairs = pickle.load(open(train_data_dir, 'rb'))
    valid = pickle.load(open(valid_data_dir, 'rb'))

    # Default word tokens
    PAD_token = voc.word2index['<PAD>']  # Used for padding short sentences
    SOS_token = voc.word2index['<SOS>']  # Start-of-sentence token
    EOS_token = voc.word2index['<EOS>']  # End-of-sentence token



    # Initialize search module
    #searcher = GreedySearchDecoder(encoder, decoder)

    # Load the best model

    loadFilename = './data/save/cb_model/Open_subtitles/3-3_512/best_model_checkpoint_original_setting_no_valid.pt'

    # If loading on same machine the model was trained on
    if USE_CUDA:
        checkpoint = torch.load(loadFilename)
    else:
        checkpoint = torch.load(loadFilename, map_location=lambda storage, loc: storage)


    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    embedding.to(device)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    encoder.to(device)
    decoder.to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    encoder.eval()
    decoder.eval()

    partition = str(args.partition) + 'th'

    print('Start generating sentences (Choose the highest prob along beam search: Do not stop!!!')
    sources = []
    counter = 0
    # Generate sentences in blocks
    length_of_division = len(pairs) // args.division
    start_generation = length_of_division * args.partition
    end_generation = length_of_division * args.partition + length_of_division

    for i in range(start_generation, end_generation):
    #for i in range(len(pairs) // 2 + 1, len(pairs)):
    #for i in range(200000):
       temp = [pairs[i][0]]
       ai_response = evaluate(encoder, decoder, voc, sentence=pairs[i][0], beam=args.beam_size)
       temp.append(ai_response)

       sources.append(temp)
       if counter % 1000 == 0:
           print('Now generated {} sentence pairs in train_data'.format(str(counter)))
           pickle.dump(sources, open('Generated_data_beam_search_no_valid_' + partition + '_half.p', 'wb'))
       counter += 1
    pickle.dump(sources, open('Generated_data_beam_search_no_valid_'+ partition + '_half.p', 'wb'))


    sources = []
    counter = 0

    # Generate sentences in blocks
    length_of_division = len(valid) // args.division
    start_generation = length_of_division * args.partition
    end_generation = length_of_division * args.partition + length_of_division

    for i in range(start_generation, end_generation):
        temp = [valid[i][0]]

        ai_response = evaluate(encoder, decoder, voc, sentence=valid[i][0], beam=args.beam_size)

        temp.append(ai_response)

        sources.append(temp)
        if counter % 1000 == 0:
            print('Now generated {} sentence pairs in valid data'.format(str(counter)))
            pickle.dump(sources, open('Generated_valid_data_beam_search_no_valid_' + partition + '_half.p', 'wb'))
        counter += 1
    pickle.dump(sources, open('Generated_valid_data_beam_search_no_valid_' + partition + '_half.p', 'wb'))
