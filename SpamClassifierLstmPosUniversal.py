import torch
import torch.nn as nn
from UniversalTagger import UniversalTagger


class SpamClassifierLstmPosUniversal(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device, index_mapper, drop_prob):
        super(SpamClassifierLstmPosUniversal, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm_cell_empty = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_adj = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_adp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_adv = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_conj = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_det = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_noun = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_num = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_prt = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_pron = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_verb = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_other = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_x = nn.LSTMCell(embedding_size, hidden_dim)


        self.dropout = nn.Dropout(drop_prob)
        # dense layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # activation function
        self.sigmoid = nn.Sigmoid()

        self.tagger = UniversalTagger()
        self.indexMapper = index_mapper

        self.tag_counter = dict()
        for tag in self.tagger.possible_tags():
            self.tag_counter[tag] = 0

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        indices_list = x.tolist()[0]
        list_of_words = self.indexMapper.indices_to_words(indices_list)
        list_of_tags = self.tagger.map_sentence(list_of_words)

        embeds = self.embedding(x)

        for i in range(0, self.embedding_size):
            tag = list_of_tags[i]
            self.tag_counter[tag] += 1
            cell_input = embeds[0][i].view(batch_size, self.embedding_size)

            if tag == 'EMPTY':
                hidden = self.lstm_cell_empty(cell_input, hidden)
            elif tag == 'ADJ':
                hidden = self.lstm_cell_adj(cell_input, hidden)
            elif tag == 'ADP':
                hidden = self.lstm_cell_adp(cell_input, hidden)
            elif tag == 'ADV':
                hidden = self.lstm_cell_adv(cell_input, hidden)
            elif tag == 'CONJ':
                hidden = self.lstm_cell_conj(cell_input, hidden)
            elif tag == 'DET':
                hidden = self.lstm_cell_det(cell_input, hidden)
            elif tag == 'NOUN':
                hidden = self.lstm_cell_noun(cell_input, hidden)
            elif tag == 'NUM':
                hidden = self.lstm_cell_num(cell_input, hidden)
            elif tag == 'PRT':
                hidden = self.lstm_cell_prt(cell_input, hidden)
            elif tag == 'PRON':
                hidden = self.lstm_cell_pron(cell_input, hidden)
            elif tag == 'VERB':
                hidden = self.lstm_cell_verb(cell_input, hidden)
            elif tag == '.':
                hidden = self.lstm_cell_other(cell_input, hidden)
            elif tag == 'X':
                hidden = self.lstm_cell_x(cell_input, hidden)
            else:
                print("Unexpected tag:", tag)
                raise NotImplementedError('Unexpected tag!')

        lstm_out = hidden[0].contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden
