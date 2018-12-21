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


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
corpus_name = 'Open_subtitles'
corpus = os.path.join('data', corpus_name)

# Default word tokens
PAD_token = 25002  # Used for padding short sentences
SOS_token = 0  # Start-of-sentence token
EOS_token = 25001  # End-of-sentence token

class Voc:
    def __init__(self, name, PAD_token=25002, SOS_token=0, EOS_token=25001):
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

    def addDictionary(self, dictionary):
        self.index2word = copy.deepcopy(dictionary)
        for index in dictionary.keys():
            self.word2index[dictionary[index]] = index
        self.num_words = len(self.index2word.keys())

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
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)




######################################################################
# Now we can assemble our vocabulary and query/response sentence pairs.
# Before we are ready to use this data, we must perform some
# preprocessing.
#
# First, we must convert the Unicode strings to ASCII using
# ``unicodeToAscii``. Next, we should convert all letters to lowercase and
# trim all non-letter characters except for basic punctuation
# (``normalizeString``). Finally, to aid in training convergence, we will
# filter out sentences with length greater than the ``MAX_LENGTH``
# threshold (``filterPairs``).
#

MAX_LENGTH = 12  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    pickle.dump(voc, open(os.path.join(save_dir, 'Vocabulary'), 'wb'))
    pickle.dump(pairs, open(os.path.join(save_dir, 'Preprocessed_data'), 'wb'))
    return voc, pairs


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc_dir = os.path.join(save_dir, 'whole_data_voc.p')
train_data_dir = os.path.join(save_dir, 'small_train_2000000.p')
valid_data_dir = os.path.join(save_dir, 'small_valid_2000000.p')

if os.path.exists(voc_dir) and os.path.exists(train_data_dir):
    voc = pickle.load(open(voc_dir, 'rb'))

    pairs = pickle.load(open(train_data_dir, 'rb'))
    valid = pickle.load(open(valid_data_dir, 'rb'))
else:
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


######################################################################
# Another tactic that is beneficial to achieving faster convergence during
# training is trimming rarely used words out of our vocabulary. Decreasing
# the feature space will also soften the difficulty of the function that
# the model must learn to approximate. We will do this as a two-step
# process:
#
# 1) Trim words used under ``MIN_COUNT`` threshold using the ``voc.trim``
#    function.
#
# 2) Filter out pairs with trimmed words.
#

MIN_COUNT = 3    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# Do zero padding given a list of either source or target sens
# Return: a list contains #max_length of lists, each list contain #batch_size elements
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
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
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = l
    #indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
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


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


######################################################################
# Define Training Procedure
# -------------------------
#
# Masked loss
# ~~~~~~~~~~~
#
# Since we are dealing with batches of padded sequences, we cannot simply
# consider all elements of the tensor when calculating loss. We define
# ``maskNLLLoss`` to calculate our loss based on our decoder’s output
# tensor, the target tensor, and a binary mask tensor describing the
# padding of the target tensor. This loss function calculates the average
# negative log likelihood of the elements that correspond to a *1* in the
# mask tensor.
#
# Change NLLloss to batch_wise !!!!!!
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    encoder.train()
    decoder.train()

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename, val_pairs,
               epoch):

    val_loss = 1000000000
    for current_epoch in range(epoch):
        # Load batches for each iteration
        # Use random choices in batches
        # training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
        #                   for _ in range(n_iteration)]
        
        if epoch > 5:
            for param_group in encoder_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
            for param_group in decoder_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2

        n_iteration = len(pairs) // batch_size - 1
        # Use random permutation in batches
        permutation = np.random.permutation(len(pairs))
        pairs = [pairs[index] for index in permutation]
        training_batches = [batch2TrainData(voc, [pairs[i + j * batch_size] for i in range(batch_size)])
                            for j in range(n_iteration)]
        check_point_name = 'best_model_small_checkpoint'
        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if loadFilename:
            start_iteration = checkpoint['iteration'] + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, n_iteration + 1):
        #for iteration in range(0):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                         decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
            print_loss += loss

            # Print progress
            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Epoch: {}; Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(current_epoch, iteration, iteration / n_iteration * 100, print_loss_avg))
                print_loss = 0

        # Save checkpoint
        input_variable, lengths, target_variable, mask, max_target_len = batch2TrainData(voc, [pair for pair in val_pairs])
        val_current_loss = validation(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                                      decoder)
        if val_current_loss < val_loss:
            print(val_loss)
            val_loss = val_current_loss
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            print('Epoch: {}; Average validation loss: {:.4f}'.format(epoch, val_current_loss))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'epoch': current_epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict(),
                'val_loss': val_loss
            }, os.path.join(directory, '{}.pt'.format(check_point_name)))
            val_loss = val_current_loss


def validation(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder):

    loss = 0
    print_losses = []
    n_totals = 0
    batch_size = 128
    #n_iteration = len(input_variable[0]) // 256
    n_iteration = len(input_variable[0]) // batch_size
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for iteration in range(n_iteration):
            # Forward pass through encoder
            encoder_outputs, encoder_hidden = encoder(input_variable[:, iteration * batch_size: iteration * batch_size + batch_size], lengths[iteration * batch_size: iteration * batch_size + batch_size])
            #encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

            # Create initial decoder input (start with SOS tokens for each sentence)
            decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # Set initial decoder hidden state to the encoder's final hidden state
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t,iteration * batch_size: iteration * batch_size + batch_size].to(device), mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

    return sum(print_losses) / n_totals



######################################################################
# Define Evaluation
# -----------------
#
# After training a model, we want to be able to talk to the bot ourselves.
# First, we must define how we want the model to decode the encoded input.
#
# Greedy decoding
# ~~~~~~~~~~~~~~~
#
# Greedy decoding is the decoding method that we use during training when
# we are **NOT** using teacher forcing. In other words, for each time
# step, we simply choose the word from ``decoder_output`` with the highest
# softmax value. This decoding method is optimal on a single time-step
# level.
#
# To facilite the greedy decoding operation, we define a
# ``GreedySearchDecoder`` class. When run, an object of this class takes
# an input sequence (``input_seq``) of shape *(input_seq length, 1)*, a
# scalar input length (``input_length``) tensor, and a ``max_length`` to
# bound the response sentence length. The input sentence is evaluated
# using the following computational graph:
#
# **Computation Graph:**
#
#    1) Forward input through encoder model.
#    2) Prepare encoder's final hidden layer to be first hidden input to the decoder.
#    3) Initialize decoder's first input as SOS_token.
#    4) Initialize tensors to append decoded words to.
#    5) Iteratively decode one word token at a time:
#        a) Forward pass through decoder.
#        b) Obtain most likely word token and its softmax score.
#        c) Record token and score.
#        d) Prepare current token to be next decoder input.
#    6) Return collections of word tokens and scores.
#

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            if decoder_input == EOS_token:
                break
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


######################################################################
# Evaluate my text
# ~~~~~~~~~~~~~~~~
#
# Now that we have our decoding method defined, we can write functions for
# evaluating a string input sentence. The ``evaluate`` function manages
# the low-level process of handling the input sentence. We first format
# the sentence as an input batch of word indexes with *batch_size==1*. We
# do this by converting the words of the sentence to their corresponding
# indexes, and transposing the dimensions to prepare the tensor for our
# models. We also create a ``lengths`` tensor which contains the length of
# our input sentence. In this case, ``lengths`` is scalar because we are
# only evaluating one sentence at a time (batch_size==1). Next, we obtain
# the decoded response sentence tensor using our ``GreedySearchDecoder``
# object (``searcher``). Finally, we convert the response’s indexes to
# words and return the list of decoded words.
#
# ``evaluateInput`` acts as the user interface for our chatbot. When
# called, an input text field will spawn in which we can enter our query
# sentence. After typing our input sentence and pressing *Enter*, our text
# is normalized in the same way as our training data, and is ultimately
# fed to the ``evaluate`` function to obtain a decoded output sentence. We
# loop this process, so we can keep chatting with our bot until we enter
# either “q” or “quit”.
#
# Finally, if a sentence is entered that contains a word that is not in
# the vocabulary, we handle this gracefully by printing an error message
# and prompting the user to enter another sentence.
#

def beam_search(decoder_hidden, beam_search_k, decoder, max_length, decoder_input, encoder_outputs, EOS_Token,
                to_device=True):

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

def evaluate(encoder, decoder, voc, sentence, max_length=MAX_LENGTH, SOS_token=0, EOS_token=25001, PAD_token=25002):
    # generate sentence per source

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
    decoder_word = beam_search(decoder_hidden, 3, decoder, max_length, decoder_input, encoder_outputs, EOS_token)


    # Decode sentence with searcher
    #tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    #decoded_words = [voc.index2word[token.item()] for token in tokens]

    return decoder_word




######################################################################
# Run Model
# ---------
#
# Finally, it is time to run our model!
#
# Regardless of whether we want to train or test the chatbot model, we
# must initialize the individual encoder and decoder models. In the
# following block, we set our desired configurations, choose to start from
# scratch or set a checkpoint to load from, and build and initialize the
# models. Feel free to play with different model configurations to
# optimize performance.
#

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

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = '/Users/TONY/Downloads/machine_learning/nlp/A-chatbot-tutorial/data/save/cb_model/cornell movie-dialogs corpus/2-2_500/4000_checkpoint.pt'
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename, map_location=lambda storage, loc: storage)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']





print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

print('Models built and ready to go!')


######################################################################
# Run Training
# ~~~~~~~~~~~~
#
# Run the following block if you want to train the model.
#
# First we set training parameters, then we initialize our optimizers, and
# finally we call the ``trainIters`` function to run our training
# iterations.
#

# Configure training/optimization
clip = 5.0
teacher_forcing_ratio = 0.7
learning_rate = 1
decoder_learning_ratio = 1.0
n_iteration = 4000
print_every = 1
save_every = 500
epoch = 100
weight_decay = 0.00001

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
# encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate * decoder_learning_ratio, weight_decay=weight_decay)
# if loadFilename:
#     encoder_optimizer.load_state_dict(encoder_optimizer_sd)
#     decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
# trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
#            embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
#            print_every, save_every, clip, corpus_name, loadFilename, valid, epoch)



######################################################################
# Run Evaluation
# ~~~~~~~~~~~~~~
#
# To chat with your model, run the following block.
#

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
#searcher = GreedySearchDecoder(encoder, decoder)

# Load the best model

loadFilename = './data/save/cb_model/Open_subtitles/3-3_512/best_model_checkpoint_original_setting_no_valid.pt'

# If loading on same machine the model was trained on
if device == 'cpu':
    checkpoint = torch.load(loadFilename, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(loadFilename)

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

#partition = '1st'
partition = '2nd'

print('Start generating sentencei (Choose the highest prob along beam search: Do not stop!!!')
sources = []
counter = 0
#for i in range(len(pairs) // 2):
##for i in range(len(pairs) // 2 + 1, len(pairs)):
##for i in range(200000):
#    temp = [pairs[i][0]]
#    ai_response = evaluate(encoder, decoder, voc, sentence=pairs[i][0])
#    temp.append(ai_response)
#
#    sources.append(temp)
#    if counter % 1000 == 0:
#        print('Now generated {} sentence pairs in train_data'.format(str(counter)))
#        pickle.dump(sources, open('Generated_data_beam_search_no_valid_' + partition + '_half.p', 'wb'))
#    counter += 1
#pickle.dump(sources, open('Generated_data_beam_search_no_valid_'+ partition + '_half.p', 'wb'))


sources = []
counter = 0

#for i in range(len(valid) // 2):
for i in range(len(valid) // 2 + 1, len(valid)):
    temp = [valid[i][0]]
     
    ai_response = evaluate(encoder, decoder, voc, sentence=valid[i][0])
    
    temp.append(ai_response)

    sources.append(temp)
    if counter % 1000 == 0:
        print('Now generated {} sentence pairs in valid data'.format(str(counter))) 
        pickle.dump(sources, open('Generated_valid_data_beam_search_no_valid_' + partition + '_half.p', 'wb'))
    counter += 1
pickle.dump(sources, open('Generated_valid_data_beam_search_no_valid_' + partition +  '_half.p', 'wb'))

# Begin chatting (uncomment and run the following line to begin)
# evaluateInput(encoder, decoder, searcher, voc)
