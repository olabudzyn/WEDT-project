import torch
import torch.nn as nn


class SpamClassifierSingleLstmCell(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device, drop_prob):
        super(SpamClassifierSingleLstmCell, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm_cell = nn.LSTMCell(embedding_size, hidden_dim)

        self.dropout = nn.Dropout(drop_prob)
        # dense layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        embeds = self.embedding(x)

        for i in range(0, self.embedding_size):
            cell_input = embeds[0][i].view(batch_size, self.embedding_size)
            hidden = self.lstm_cell(cell_input, hidden)

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
