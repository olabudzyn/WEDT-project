import torch
import torch.nn as nn
from Tagger import FullTagger


class SpamClassifierLstmCellPos(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device, index_mapper, drop_prob):
        super(SpamClassifierLstmCellPos, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm_cell_vbn = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_vbz = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_vbg = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_vbp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_vbd = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_md = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_nn = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_nnps = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_nnp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_nns = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_jjs = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_jjr = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_jj = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_rb = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_rbr = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_rbs = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_EMPTY = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_cd = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_in = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_pdt = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_cc = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_ex = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_pos = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_rp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_fw = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_dt = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_uh = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_to = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_prp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_prp_dollar = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_dollar = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_wp = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_wp_dollar = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_wdt = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_wrb = nn.LSTMCell(embedding_size, hidden_dim)
        self.lstm_cell_other = nn.LSTMCell(embedding_size, hidden_dim)

        self.dropout = nn.Dropout(drop_prob)
        # dense layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # activation function
        self.sigmoid = nn.Sigmoid()

        self.tagger = FullTagger()
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

            if tag == 'VBN':
                hidden = self.lstm_cell_vbn(cell_input, hidden)
            elif tag == 'VBZ':
                hidden = self.lstm_cell_vbz(cell_input, hidden)
            elif tag == 'VBG':
                hidden = self.lstm_cell_vbg(cell_input, hidden)
            elif tag == 'VBP':
                hidden = self.lstm_cell_vbp(cell_input, hidden)
            elif tag == 'VBD':
                hidden = self.lstm_cell_vbd(cell_input, hidden)
            elif tag == 'MD':
                hidden = self.lstm_cell_md(cell_input, hidden)
            elif tag == 'NN':
                hidden = self.lstm_cell_nn(cell_input, hidden)
            elif tag == 'NNPS':
                hidden = self.lstm_cell_nnps(cell_input, hidden)
            elif tag == 'NNP':
                hidden = self.lstm_cell_nnp(cell_input, hidden)
            elif tag == 'NNS':
                hidden = self.lstm_cell_nns(cell_input, hidden)
            elif tag == 'JJS':
                hidden = self.lstm_cell_jjs(cell_input, hidden)
            elif tag == 'JJR':
                hidden = self.lstm_cell_jjr(cell_input, hidden)
            elif tag == 'JJ':
                hidden = self.lstm_cell_jj(cell_input, hidden)
            elif tag == 'RB':
                hidden = self.lstm_cell_rb(cell_input, hidden)
            elif tag == 'RBR':
                hidden = self.lstm_cell_rbr(cell_input, hidden)
            elif tag == 'RBS':
                hidden = self.lstm_cell_rbs(cell_input, hidden)
            elif tag == 'EMPTY':
                hidden = self.lstm_cell_EMPTY(cell_input, hidden)
            elif tag == 'CD':
                hidden = self.lstm_cell_cd(cell_input, hidden)
            elif tag == 'IN':
                hidden = self.lstm_cell_in(cell_input, hidden)
            elif tag == 'PDT':
                hidden = self.lstm_cell_pdt(cell_input, hidden)
            elif tag == 'CC':
                hidden = self.lstm_cell_cc(cell_input, hidden)
            elif tag == 'EX':
                hidden = self.lstm_cell_ex(cell_input, hidden)
            elif tag == 'POS':
                hidden = self.lstm_cell_pos(cell_input, hidden)
            elif tag == 'RP':
                hidden = self.lstm_cell_rp(cell_input, hidden)
            elif tag == 'FW':
                hidden = self.lstm_cell_fw(cell_input, hidden)
            elif tag == 'DT':
                hidden = self.lstm_cell_dt(cell_input, hidden)
            elif tag == 'UH':
                hidden = self.lstm_cell_uh(cell_input, hidden)
            elif tag == 'TO':
                hidden = self.lstm_cell_to(cell_input, hidden)
            elif tag == 'PRP':
                hidden = self.lstm_cell_prp(cell_input, hidden)
            elif tag == 'PRP$':
                hidden = self.lstm_cell_prp_dollar(cell_input, hidden)
            elif tag == '$':
                hidden = self.lstm_cell_dollar(cell_input, hidden)
            elif tag == 'WP':
                hidden = self.lstm_cell_wp(cell_input, hidden)
            elif tag == 'WP$':
                hidden = self.lstm_cell_wp_dollar(cell_input, hidden)
            elif tag == 'WDT':
                hidden = self.lstm_cell_wdt(cell_input, hidden)
            elif tag == 'WRB':
                hidden = self.lstm_cell_wrb(cell_input, hidden)
            elif tag == 'OTHER':
                hidden = self.lstm_cell_other(cell_input, hidden)
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
