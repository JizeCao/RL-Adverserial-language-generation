import torch
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import shutil
import pickle
import time
import random
import math
from model import EncoderRNN, LuongAttnDecoderRNN, hierEncoder, hierEncoder_frequency


def load_model_dictionary_pairs(args, dis_model=True, only_data=False):
    hidden_size = 512
    encoder_n_layers = 3
    decoder_n_layers = 3
    dropout = 0.2
    attn_model = 'dot'
    learning_rate = args.gen_lr
    decoder_learning_ratio = 1.0
    dis_learning_rate = args.dis_lr
    voc = pickle.load(open(os.path.join(args.save_dir, 'Vocabulary'), 'rb'))
    #train_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'small_train_2000000.p'), 'rb'))
    train_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'train_remain_pairs'), 'rb'))
    valid_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'small_valid_2000000.p'), 'rb'))
    neg_train_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_train.p'), 'rb'))
    neg_valid_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_valid.p'), 'rb'))

    #  The set up of the experiment
    model_save_dir = os.path.join(args.save_dir, 'cb_model/Open_subtitles/3-3_512')
    #checkpoint = torch.load(os.path.join(model_save_dir, '4000_checkpoint.pt'), map_location=lambda storage, loc: storage)
    loadFilename = os.path.join(model_save_dir, 'best_model_checkpoint_original_setting_no_valid.pt')
    print(args)
    if args.cuda:
        checkpoint = torch.load(loadFilename)
        print('check_point')
    else:
        checkpoint = torch.load(loadFilename, map_location=lambda storage, loc: storage)

    if only_data:
        return train_pos_pairs, valid_pos_pairs, neg_train_pairs, neg_valid_pairs

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
    embedding.to(args.device)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    encoder.to(args.device)
    decoder.to(args.device)

    if args.adam:
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.gen_lr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.gen_lr)
    else:
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.gen_lr, momentum=0.8)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.gen_lr, momentum=0.8)
    if dis_model:
        if args.cuda:
            #dis_checkpoint = torch.load(os.path.join(model_save_dir, 'disc_params_correct_beam.pt'))
            dis_checkpoint = torch.load(os.path.join(model_save_dir, 'disc_params_beam_frquency.pt'))
        else:
            dis_checkpoint = torch.load(os.path.join(model_save_dir, 'disc_params_beam_frquency.pt'), map_location=lambda storage, loc:storage)
        #dis_model = hierEncoder(len(voc.index2word), 500)
        dis_model = hierEncoder_frequency(len(voc.index2word), 500)
        dis_model.load_state_dict(dis_checkpoint['disc'])
        dis_model.to(args.device)
        dis_model_optimizer = optim.SGD(dis_model.parameters(), lr=args.dis_lr, momentum=0.8)
        return encoder, decoder, dis_model, encoder_optimizer, decoder_optimizer, dis_model_optimizer, voc, train_pos_pairs,\
               valid_pos_pairs, neg_train_pairs, neg_valid_pairs, embedding
    else:
        return voc, train_pos_pairs, valid_pos_pairs, neg_train_pairs, neg_valid_pairs


def trim_dummy_sen(SenSet, vocab, EOS_id=25001, PAD_id=25002):
    processed_list = []
    for pair_id in range(len(SenSet)):
        temp_list = []
        for j in range(2):
            if len(SenSet[pair_id][j]) == 0:
                break
            else:
                if type(SenSet[pair_id][j]) is list:
                    temp_list.append([word for word in SenSet[pair_id][j]
                                           if word != EOS_id and word != PAD_id])
                else:
                    temp_list.append([word for word in SenSet[pair_id][j].split()
                                              if word != EOS_id and word != PAD_id])
                if j == 1:
                    processed_list.append(temp_list)
    return processed_list


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


## Just for writing the log
def logging(s, log_name, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join('./data/save', log_name), 'a+') as f_log:
            f_log.write(s + '\n')

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def indexesFromSentence(voc, sentence, EOS_token=2):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# pos_data and neg_data should both are tensors
def build_mega_data(pos_data, neg_data, args):
    if not (len(pos_data) == len(neg_data)):
        print(len(pos_data), len(neg_data))
        print("Warning: the size positive data is not equivalent to the negative one")
    # Satisfy the seq_length requirement
    mega_data = torch.cat([pos_data, neg_data]).numpy().reshape(-1, args.seq_len)
    labels = np.append(np.ones(len(pos_data) // args.seq_len), np.zeros(len(neg_data) // args.seq_len)).reshape((-1, 1))
    mega_data = np.append(mega_data, labels, 1)
    return mega_data


def tensorFromPair(pair, EOS_Token, to_device=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_tensor = pair[0]
    target_tensor = pair[1]

    if to_device:

        if input_tensor[-1] != EOS_Token:
            if type(input_tensor) is list:
                input_tensor.append(EOS_Token)
            else:
                input_tensor = torch.cat((input_tensor, torch.LongTensor([EOS_Token])), 0)
        
        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(-1,1).to(device)

        if target_tensor[-1] != EOS_Token:
            if type(target_tensor) is list:
                target_tensor.append(EOS_Token)     
            else:
                target_tensor = torch.cat((target_tensor, torch.LongTensor([EOS_Token])), 0)
        target_tensor = torch.tensor(target_tensor, dtype=torch.long).view(-1,1).to(device)

    else:
        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(-1, 1)

        if target_tensor[-1] != EOS_Token:
            target_tensor.append(EOS_Token)
        target_tensor = torch.tensor(target_tensor, dtype=torch.long).view(-1, 1)

    return input_tensor, target_tensor

def tensorFromPairEval(pair, EOS_Token, to_device=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_tensor = pair[0]
    target_tensor = pair[1]

    if to_device:
        if input_tensor[-1] != EOS_Token:
            if type(input_tensor) is list:
                input_tensor.append(EOS_Token)
            else:
                input_tensor = torch.cat((input_tensor, torch.LongTensor([EOS_Token])), 0)

        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(-1,1).to(device)

        if target_tensor[-1] != EOS_Token:
            if type(target_tensor) is list:
                target_tensor.append(EOS_Token)
            else:
                target_tensor = torch.cat((target_tensor, torch.LongTensor([EOS_Token])), 0)
        target_tensor = torch.tensor(target_tensor, dtype=torch.long).view(-1,1).to(device)

    else:
        input_tensor = torch.tensor(input_tensor, dtype=torch.long).view(-1, 1)

        if target_tensor[-1] != EOS_Token:
            target_tensor = torch.cat((target_tensor, torch.LongTensor([EOS_Token])), 0)
        target_tensor = torch.tensor(target_tensor, dtype=torch.long).view(-1, 1)

    return input_tensor, target_tensor

def evaluate_word(decoder, encoder_outputs, input, decoder_hidden, args):
    # Unsqueeze input to satisfy dimension
    if args.cuda:
        decoder_output, decoder_hidden = decoder(input.unsqueeze(0).to(args.device), decoder_hidden, encoder_outputs)
    else:
        decoder_output, decoder_hidden = decoder(input.unsqueeze(0), decoder_hidden, encoder_outputs)

    return decoder_output.detach().squeeze(), decoder_hidden.detach()


def evaluate_sen(encoder, decoder, sentence, device, encoder_outputs=None, decoder_input=None, decoder_hidden=None,
                 SOS_token=0, EOS_token=25001):
    ### Format input sentence as a batch
    # words -> indexes
    # if type(sentence) is str:
    #     indexes_batch = [indexesFromSentence(voc, sentence)]
    # else:
    if sentence[-1] != EOS_token:
        indexes_batch = torch.cat((sentence, torch.LongTensor([EOS_token])), 0)
    else:
        indexes_batch = sentence
    # Create lengths tensor
    # lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    lengths = torch.LongTensor([len(indexes_batch)])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).unsqueeze(1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    if encoder_outputs is None:
        encoder_outputs, encoder_hidden = encoder(input_batch, lengths)
    if decoder_hidden is None:
        decoder_hidden = encoder_hidden[:decoder.n_layers]
    if decoder_input is None:
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
    return encoder_outputs, decoder_output, decoder_hidden


# Data source is a pair
def dis_evaluate_sen(data_source, model, args):
    with torch.no_grad():
        data = data_source
        # print(type(data), data.shape)
        log_prob = model(data, to_device=True)
        # Only evaluate last element is important
        prob = torch.exp(log_prob)[0][0].item()
        # sys.exit()
    return prob


# Temporary solution
def evaluateD(modelD, pos_valid, neg_valid, EOS_token):
    # prepare data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    to_device = True
    pos_data = [tensorFromPair(pos_sen, EOS_Token=EOS_token, to_device=to_device) for pos_sen in pos_valid]
    neg_data = [tensorFromPair(neg_sen, EOS_Token=EOS_token, to_device=to_device) for neg_sen in neg_valid]
    posTag = torch.tensor([0]).to(device)
    negTag = torch.tensor([1]).to(device)
    loss = 0
    start_time = time.time()
    criterion = nn.NLLLoss()

    modelD.eval()
    with torch.no_grad():
        for pos_pair in pos_data:
            output = modelD(pos_pair, to_device=True)
            loss += criterion(output, posTag)
        for neg_pair in neg_data:
            output = modelD(neg_pair, to_device=True)
            loss += criterion(output, negTag)

    print("Time consumed: {} Batch loss: {:.2f} ".format((time.time()-start_time),
                                                         loss.item() / (len(pos_data) + len(neg_data))))
    return loss.item() / (len(pos_data) + len(neg_data))


def dis_retrain(dis_model, args, train_data, labels, ix_to_word=None, dis_lr=0.001, validation=False,
                pos_valid_pairs=None, neg_valid_pairs=None, current_val_loss=0):

    dis_val_loss = 0

    EOS_token = args.EOS_id
    if validation:
        dis_model.eval()
        dis_val_loss = evaluateD(dis_model, pos_valid=pos_valid_pairs[:20000], neg_valid=neg_valid_pairs[:20000], EOS_token=EOS_token)
        #print('The dis_val_loss for 20 iteration is', dis_val_loss)
        if dis_val_loss >= current_val_loss:
            dis_lr /= 2
            current_val_loss = dis_val_loss

    dis_model.train()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dis_optimizer = optim.SGD(dis_model.parameters(), lr=dis_lr)
    print('The discriminator learning rate is', str(dis_lr))
    criterion = nn.NLLLoss()
    dis_optimizer.zero_grad()    
    labels = torch.LongTensor(labels).to(device)
    Accuracy = 0
    loss = 0
    total_pos_prob = 0
    for i in range(len(labels)):
        train_data[i] = tensorFromPairEval(train_data[i], EOS_token)
        output = dis_model(train_data[i], to_device=True)
        pos_prob = torch.exp(output)[0][0].item()
        total_pos_prob += pos_prob
        outputTag = torch.argmax(torch.exp(output))
        if outputTag == labels[i].long():
            Accuracy += 1
        current_loss = criterion(output, labels[i])
        loss += current_loss
    loss.backward()
    
    _ = torch.nn.utils.clip_grad_norm_(dis_model.parameters(), args.clip)
    dis_optimizer.step()
    print('Accuracy: ', Accuracy / len(labels), 'loss: ', loss)
    print('Average pos prob: ', total_pos_prob / len(labels))
    # current_val_loss is the largest validation loss, loss is 
    # the actual 'current' one !!!
    return loss, dis_lr, current_val_loss, dis_val_loss


