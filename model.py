import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim


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
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
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


class hierEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size):

        super(hierEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = self.embedding_size  # the hidden size of GRU equals embedding size by default

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size)
        self.gru2 = nn.GRU(self.embedding_size, 128)
        self.linear1 = nn.Linear(128, 32)
        self.linear2 = nn.Linear(32, 2)

    def forward(self, pair, to_device=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        hidden_x = hidden     # x句的编码结果

        if to_device:
            hidden = self.initHidden().to(device)
        else:
            hidden = self.initHidden()

        for i in range(y_length):
            embedded_y = self.embedding(pair[1][i]).view(1, 1, -1)
            _, hidden = self.gru1(embedded_y, hidden)
        hidden_y = hidden     # y句的编码结果

        if to_device:
            hidden = torch.zeros(1, 1, 128).to(device)
        else:
            hidden = torch.zeros(1, 1, 128)

        _, hidden = self.gru2(hidden_x, hidden)
        _, hidden = self.gru2(hidden_y, hidden)
        hidden_xy = hidden    # 得到{x，y}编码结果

        output = F.relu(self.linear1(hidden_xy.squeeze()))
        output = F.relu(self.linear2(output)).view(1, -1)
        output = F.log_softmax(output, dim=1)        ## 注意此处的输出为 log_softmax

        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.embedding_size)

class hierEncoder_frequency(nn.Module):
    def __init__(self, vocab_size, embedding_size):

        super(hierEncoder_frequency, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = self.embedding_size  # the hidden size of GRU equals embedding size by default

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size)
        self.gru2 = nn.GRU(self.embedding_size, 128)
        self.linear1 = nn.Linear(128, 32)
        self.linear2 = nn.Linear(33, 2)

    def count_max_frequency(self, sen):
        counts = torch.zeros(torch.max(sen).item() + 1)
        for i in sen:
            counts[i] += 1
        return torch.max(counts).view(1)


    def forward(self, pair, to_device=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        hidden_x = hidden     # x句的编码结果

        if to_device:
            hidden = self.initHidden().to(device)
        else:
            hidden = self.initHidden()

        for i in range(y_length):
            embedded_y = self.embedding(pair[1][i]).view(1, 1, -1)
            _, hidden = self.gru1(embedded_y, hidden)
        hidden_y = hidden     # y句的编码结果

        if to_device:
            hidden = torch.zeros(1, 1, 128).to(device)
        else:
            hidden = torch.zeros(1, 1, 128)

        _, hidden = self.gru2(hidden_x, hidden)
        _, hidden = self.gru2(hidden_y, hidden)
        hidden_xy = hidden    # 得到{x，y}编码结果

        max_frequency = self.count_max_frequency(pair[1]).to(device)
        output = F.relu(self.linear1(hidden_xy.squeeze()))
        output = torch.cat((output, max_frequency), dim=0)
        output = F.relu(self.linear2(output)).view(1, -1)
        output = F.log_softmax(output, dim=1)        ## 注意此处的输出为 log_softmax

        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.embedding_size)


class hierEncoder_frequency_batchwise(nn.Module):
    def __init__(self, vocab_size, embedding_size):

        super(hierEncoder_frequency_batchwise, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = self.embedding_size  # the hidden size of GRU equals embedding size by default

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size)
        self.gru2 = nn.GRU(self.embedding_size, 128)
        self.linear1 = nn.Linear(128, 32)
        self.linear2 = nn.Linear(33, 2)

    def count_max_frequency(self, sen):
        counts = torch.zeros(torch.max(sen).item() + 1)
        for i in sen:
            counts[i] += 1
        return torch.max(counts).view(1)

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
            output = torch.cat((output, max_frequency), dim=0)
            output = F.relu(self.linear2(output)).view(1, -1)
            output = F.log_softmax(output, dim=1)  ## 注意此处的输出为 log_softmax

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


        max_frequency = torch.Tensor([self.count_max_frequency(targets[:, i]) for i in range(len(targets[0]))]).unsqueeze(dim=1).to(device)
        output = F.relu(self.linear1(hidden.squeeze()))
        output = torch.cat((output, max_frequency), dim=1)
        output = F.relu(self.linear2(output))
        output = F.log_softmax(output, dim=1)  ## 注意此处的输出为 log_softmax

        return output


if __name__ == '__main__':
    sources = torch.round(torch.rand(4, 10)).long()
    targets = torch.round(torch.rand(4, 10)).long()
    length = torch.LongTensor([4] * 10)
    dist = hierEncoder_frequency_batchwise(10, 100)
    result = dist(sources=sources, targets=targets, sources_length=length, targets_length=length)
    print()

