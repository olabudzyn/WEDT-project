import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import nltk
import numpy as np
import tqdm
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split


SEQUENCE_LENGTH = 100  # the length of all sequences (number of words per sample)
EMBEDDING_SIZE = 100   # Using 100-Dimensional GloVe embedding vectors
TEST_SIZE = 0.20       # ratio of testing set
OUTPUT_SIZE = 1
#N_ITERS = 5
#EPOCHS = int(N_ITERS / (len(X_train) / BATCH_SIZE))
EPOCHS = 3
HIDDEN_DIM = 100
N_LAYERS = 2
LEARNING_RATE = 0.005


# to convert labels to integers and vice-versa
label2int = {"ham": 0, "spam": 1}
int2label = {0: "ham", 1: "spam"}


def load_data():
    texts, labels = [], []
    with open("data/SMSSpamCollection", encoding="utf8") as f:
        for line in f:
            split = line.split()
            labels.append(split[0].strip())
            texts.append(' '.join(split[1:]).strip())
    return texts, labels


# load the data
num = 5574
X, y = load_data()
X = X[:num]
y = y[:num]

stop_words = set(stopwords.words('english'))

dataWithoutStopWords = []
# filter sentences stop words
for j in range(0, len(X)):
    tokenized = sent_tokenize(X[j])
    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        wordsList = [w for w in wordsList if not w in stop_words]
        sentence = ' '.join(wordsList)
        dataWithoutStopWords.append(sentence)

# Text tokenization
# vectorizing text, turning each text into sequence of integers
# tokenizer1 = Tokenizer(lower=False)
# tokenizer1.fit_on_texts(dataWithoutStopWords)
# convert to sequence of integers

# tokenizedSentences = tokenizer1.texts_to_sequences(dataWithoutStopWords)
# X = tokenizer1.texts_to_sequences(dataWithoutStopWords)

taggedWordsList = []

for j in range(0, len(dataWithoutStopWords)):
    tokenized = sent_tokenize(dataWithoutStopWords[j])
    for i in tokenized:
        # Word tokenizers is used to find the words
        # and punctuation in a string
        wordsList = nltk.word_tokenize(i)
        #  Using a Tagger. Which is part-of-speech
        # tagger or POS-tagger.
        tagged = nltk.pos_tag(wordsList)
        taggedWordsList.append(tagged)

taggedWordsList2 = []

for j in range(0, 10):
    tokenized = sent_tokenize(X[j])
    for i in tokenized:
        # Word tokenizers is used to find the words
        # and punctuation in a string
        wordsList = nltk.word_tokenize(i)
        #  Using a Tagger. Which is part-of-speech
        # tagger or POS-tagger.
        tagged = nltk.pos_tag(wordsList)
        taggedWordsList2.append(tagged)

taggedWordsList3 = []

for j in range(0, 10):
    tokenized = sent_tokenize(X[j])
    for i in tokenized:
        # Word tokenizers is used to find the words
        # and punctuation in a string
        wordsList = nltk.word_tokenize(i)

        # removing stop words from wordList
        wordsList = [w for w in wordsList if not w in stop_words]

        #  Using a Tagger. Which is part-of-speech
        # tagger or POS-tagger.
        tagged = nltk.pos_tag(wordsList)

        taggedWordsList3.append(tagged)


# Text tokenization
# vectorizing text, turning each text into sequence of integers
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(X)
# convert to sequence of integers
X = tokenizer.texts_to_sequences(X)
# convertomg to numpy arrays
X = np.array(X)
y = np.array(y)

# padding sequences at the beginning of each sequence with 0's to SEQUENCE_LENGTH
X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

y = [label2int[label] for label in y]
y = np.asarray(y, dtype=np.float32)

import json

# split and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=7)


dictionary = json.loads(tokenizer.get_config()['index_word'])


split_frac = 0.5 # 50% validation, 50% test
split_id = int(split_frac * len(X_test))
X_val, X_test = X_test[:split_id], X_test[split_id:]
y_val, y_test = y_test[:split_id], y_test[split_id:]

train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

BATCH_SIZE = int(1)   # it must be a divisor X_train and X_val
# BATCH_SIZE = int(len(X_val)/1)   # it must be a divisor X_train and X_val

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)



def get_embedding_vectors(tokenizer, dim=100):
    embedding_index = {}
    with open(f"data/glove.6B.{dim}d.txt", encoding='utf8') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

embedding_matrix = get_embedding_vectors(tokenizer)



# initialize our ModelCheckpoint and TensorBoard callbacks
# model checkpoint for saving best weights
model_checkpoint = ModelCheckpoint("results/spam_classifier_{val_loss:.2f}", save_best_only=True,
                                   verbose=1)



# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    print(sample_x.shape, sample_y.shape)


def indexToWord(index):
    if index == 0:
        return " "
    word = dictionary[str(index)]
    return word

from nltk.data import load

possibleTagList = load('help/tagsets/upenn_tagset.pickle').keys()

def mapTag(tag):
    if tag in ['VBN','VBZ','VBG','VBP','VBD','MD']:
        return 'V'
    elif tag in ['NN', 'NNPS', 'NNP', 'NNS']:
        return 'N'
    elif tag in ['JJS', 'JJR', 'JJ']:
        return 'A'
    elif tag in ['WP', 'WP$', 'WDT', 'WRB']:
        return 'W'
    elif tag in ['RB', 'RBR', 'RBS']:
        return 'ADV'
    elif tag in ['$']:
        return 'DOLLAR'
    elif tag in ['CD']:
        return 'CD'
    elif tag in ['IN', 'PDT', 'CC', 'EX', 'POS', 'RP', 'FW', 'DT', 'UH', 'TO', 'PRP', 'PRP$']:
        return 'OTHER'
    else:
        return 'OTHER_OTHER'

newTagDictionary = list(dict.fromkeys(map(mapTag, possibleTagList)))

tagCounter = dict()
for tag in newTagDictionary:
    tagCounter[tag] = 0



class SpamClassifier(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SpamClassifier, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = 100

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstmCellOne = nn.LSTMCell(100, hidden_dim)
        self.lstmCells = dict()
        for tag in newTagDictionary:
            newlstmCell = nn.LSTMCell(100, hidden_dim)
            self.lstmCells[tag] = newlstmCell

        self.lstm = nn.LSTM(EMBEDDING_SIZE, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        # self.lstmUnits = dict()
        # for tag in possibleTagList:
        #     newlstm = nn.LSTM(EMBEDDING_SIZE, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        #     self.lstmUnits[tag] = newlstm


        self.dropout = nn.Dropout(0.2)
        # dense layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # activation function
        self.softmax = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        listOfWords = []
        indexList = x.tolist()[0]
        for i in range(0, len(indexList)):
            index = indexList[i]
            word = indexToWord(index)
            listOfWords.append(word)

        listOfPOSTags = nltk.pos_tag(listOfWords)
        listOfTags = []
        for i in range(0, len(listOfPOSTags)):
            tuple = listOfPOSTags[i]
            tag = mapTag(tuple[1])
            listOfTags.append(tag)

        embeds = self.embedding(x)
        for i in range(0, 100):
            tag = listOfTags[i]
            tagCounter[tag] += 1
            input = embeds[0][i].view(1,100)
            hidden = self.lstmCells[tag](input, hidden)
            # hidden = self.lstmCellOne(input, hidden)

        # lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = hidden[0].contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.softmax(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(1, self.hidden_dim).zero_().to(device),
                  weight.new(1, self.hidden_dim).zero_().to(device))
        return hidden


VOCAB_SIZE = len(tokenizer.word_index) + 1


model = SpamClassifier(VOCAB_SIZE, OUTPUT_SIZE, EMBEDDING_SIZE, HIDDEN_DIM, N_LAYERS)
model.to(device)
print(model)


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

counter = 0
print_every = 1000
clip = 5
valid_loss_min = np.Inf


######################## TRAINING ###########################
# Set model to train configuration
model.train()

for i in range(EPOCHS):
    h = model.init_hidden(BATCH_SIZE)
    start_time = time.time()
    avg_loss = 0

    for inputs, labels in train_loader:
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

        # For every (print_every) checking checking output of the model against the validation dataset
        # and saving the model if it performed better than the previous time
        if counter % print_every == 0:
            val_h = model.init_hidden(BATCH_SIZE)
            val_losses = []
            # Set model to validation configuration - Doesn't get trained here
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(i + 1, EPOCHS),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state/state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)



######################## TESTING ###########################
# Loading the best model
model.load_state_dict(torch.load('./state/state_dict.pt'))

test_losses = []
num_correct = 0
h = model.init_hidden(BATCH_SIZE)
model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))



def get_predictions(text):
    model.load_state_dict(torch.load('./state/state_dict.pt'))
    model.eval()
    h = model.init_hidden(1)
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    for inputs in sequence:
        inputs =  np.reshape(inputs, (1,len(inputs)))
        inputs = torch.from_numpy(inputs)
        h = tuple([each.data for each in h])
        output, h = model(inputs, h)
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    if(pred==0):
        return "ham"
    else:
        return "spam"


text = "Congratulations! you have won 100,000$ this week, click here to claim fast"
print(get_predictions(text))

text = "Hi man, I was wondering if we can meet tomorrow."
print(get_predictions(text))

text = "Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged"
print(get_predictions(text))