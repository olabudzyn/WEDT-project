import matplotlib.pyplot as plt
import nltk
import numpy as np
import scipy.special
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from torch.utils import data

SEQUENCE_LENGTH = 100                  # the length of all sequences (number of words per sample)
EMBEDDING_SIZE = 100                   # Using 100-Dimensional GloVe embedding vectors
TEST_SIZE = 0.25                       # ratio of testing set
BATCH_SIZE = 1
EPOCHS = 7                             # number of epochs
HIDDEN_SIZE = 50                       # Number of dimensions in the hidden state
INPUT_SIZE = 1                         # Size of the vocabulary used (each word represented as one number)
Z_SIZE = HIDDEN_SIZE + INPUT_SIZE      # Size of concatenated hidden + input vector
OUTPUT_SIZE = 1
ETA = 0.001                            # learning rate

# to convert labels to integers and vice-versa
label2int = {"ham": 0, "spam": 1}
int2label = {0: "ham", 1: "spam"}


def load_data():
    """
    Loads SMS Spam Collection dataset
    """
    texts, labels = [], []
    with open("data/SMSSpamCollection", encoding="utf8") as f:
        for line in f:
            split = line.split()
            labels.append(split[0].strip())
            texts.append(' '.join(split[1:]).strip())
    return texts, labels


# load the data
X, y = load_data()
X = X[:300]
y = y[:300]


stop_words = set(stopwords.words('english'))

# sent_tokenize is one of instances of
# PunktSentenceTokenizer from the nltk.tokenize.punkt module

taggedWordsList = []

for j in range(0, len(X)):
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

        taggedWordsList.append(tagged)

        # print(tagged)

# Text tokenization
# vectorizing text, turning each text into sequence of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
# convert to sequence of integers
X = tokenizer.texts_to_sequences(X)


# convert to numpy arrays
X = np.array(X)
y = np.array(y)
# pad sequences at the beginning of each sequence with 0's
# for example if SEQUENCE_LENGTH=4:
# [[5, 3, 2], [5, 1, 2, 3], [3, 4]]
# will be transformed to:
# [[0, 5, 3, 2], [5, 1, 2, 3], [0, 0, 3, 4]]
X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

# One Hot encoding labels
# [spam, ham, spam, ham, ham] will be converted to:
# [1, 0, 1, 0, 1] and then to:
# [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
y2 = [label2int[label] for label in y]
print(y2)
y = [label2int[label] for label in y]
y = to_categorical(y)

y = y2      #temporarily


# split and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=7)

# divide test dataset to test and validation
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=7)



class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]

        return X, y


def create_datasets( dataset_class):
    inputs_train = X_train
    targets_train = y_train
    inputs_val = X_test
    targets_val = y_test
    inputs_test = X_test
    targets_test = y_test

    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set

training_set, validation_set, test_set = create_datasets(Dataset)

print(f'We have {len(training_set)} samples in the training set.')
print(f'We have {len(validation_set)} samples in the validation set.')
print(f'We have {len(test_set)} samples in the test set.')




def init_orthogonal(param):
    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported.")

    rows, cols = param.shape
    new_param = np.random.randn(rows, cols)

    if rows < cols:
        new_param = new_param.T

    # Compute QR factorization
    q, r = np.linalg.qr(new_param)

    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T

    new_param = q

    return new_param


def sigmoid(x, derivative=False):
    """
    Computes the element-wise sigmoid activation function for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = 1 / (1 + np.exp(-x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        return f * (1 - f)
    else:  # Return the forward pass of the function at x
        return f


def tanh(x, derivative=False):
    """
    Computes the element-wise tanh activation function for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = (np.exp(x_safe) - np.exp(-x_safe)) / (np.exp(x_safe) + np.exp(-x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        return 1 - f ** 2
    else:  # Return the forward pass of the function at x
        return f


def softmax(x, derivative=False):
    """
    Computes the softmax for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        pass  # We will not need this one
    else:  # Return the forward pass of the function at x
        return f



def clip_gradient_norm(grads, max_norm=0.25):
    # Set the maximum of the norm to be of type float
    max_norm = float(max_norm)
    total_norm = 0

    # Calculate the L2 norm squared for each gradient and add them to the total norm
    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # If the total norm is larger than the maximum allowable norm, then clip the gradient
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef

    return grads



def update_parameters(params, grads, lr=1e-3):
    # Take a step
    for param, grad in zip(params, grads):
        param -= lr * grad

    return params



def init_lstm(hidden_size, input_size, z_size):

    # Weight matrix (forget gate)
    W_forget = np.random.randn(hidden_size, z_size)

    # Bias for forget gate
    b_forget = np.zeros((hidden_size, 1))

    # Weight matrix (input gate)
    W_input = np.random.randn(hidden_size, z_size)

    # Bias for input gate
    b_input = np.zeros((hidden_size, 1))

    # Weight matrix (candidate)
    W_g = np.random.randn(hidden_size, z_size)

    # Bias for candidate
    b_g = np.zeros((hidden_size, 1))

    # Weight matrix of the output gate
    W_output = np.random.randn(hidden_size, z_size)
    b_output = np.zeros((hidden_size, 1))

    # Weight matrix relating the hidden-state to the output
    W_v = np.random.randn(input_size, hidden_size)
    b_v = np.zeros((input_size, 1))

    # Fully connected layer Weights and Bias
    W_fc = np.random.randn(100, OUTPUT_SIZE)  # first size should be hidden size, but errors
    b_fc = np.zeros((1, OUTPUT_SIZE))

    # Initialize weights
    W_forget = init_orthogonal(W_forget)
    W_input = init_orthogonal(W_input)
    W_g = init_orthogonal(W_g)
    W_output = init_orthogonal(W_output)
    W_v = init_orthogonal(W_v)
    W_fc = init_orthogonal(W_fc)

    return W_forget, W_input, W_g, W_output, W_v, W_fc, b_forget, b_input, b_g, b_output, b_v, b_fc



def model_output(lstm_output, W_fc, b_fc):
  '''Takes the LSTM output and transforms it to our desired
  output size using a final, fully connected layer'''
  model_output = np.dot(lstm_output,W_fc) + b_fc
  return sigmoid(model_output)



def forward(inputs, h_prev, C_prev, p):
    """
    Arguments:
    x -- your input data at timestep "t", numpy array of shape (n_x, m).
    h_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    C_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    p -- python list containing:
                        W_forget -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        b_forget -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        W_input -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        b_input -- Bias of the update gate, numpy array of shape (n_a, 1)
                        W_g -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        b_g --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        W_output -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        b_output --  Bias of the output gate, numpy array of shape (n_a, 1)
                        W_v -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_v, n_a)
                        b_v -- Bias relating the hidden-state to the output, numpy array of shape (n_v, 1)
    Returns:
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s -- lists of size m containing the computations in each forward pass
    outputs -- prediction at timestep "t", numpy array of shape (n_v, m)
    """

    assert C_prev.shape == (HIDDEN_SIZE, 1)
    assert h_prev.shape == (HIDDEN_SIZE, 1)

    # First we unpack our parameters
    W_forget, W_input, W_g, W_output, W_v, W_fc, b_forget, b_input, b_g, b_output, b_v, b_fc = p

    # Save a list of computations for each of the components in the LSTM
    x_s, z_s, f_s, i_s, = [], [], [], []
    g_s, C_s, o_s, h_s = [], [], [], []
    v_s, output_s, fc_output_s = [], [], []

    # Append the initial cell and hidden state to their respective lists
    h_s.append(h_prev)
    C_s.append(C_prev)

    # inputs is one sms
    for x in inputs:
        # Concatenate input and hidden state
        a = np.array(x)[np.newaxis]
        input = a.T
        x = np.transpose(x)
        z = np.row_stack((h_prev, input))
        z_s.append(z)


        # Calculate forget gate
        f = sigmoid(np.dot(W_forget, z) + b_forget)
        f_s.append(f)

        # Calculate input gate
        i = np.float64(sigmoid(np.dot(W_input, z) + b_input))
        i_s.append(i)

        # Calculate candidate
        g = tanh(np.dot(W_g, z) + b_g)
        g_s.append(g)

        # Calculate memory state
        C_prev = f * C_prev + i * g
        C_s.append(C_prev)

        # Calculate output gate
        o = sigmoid(np.dot(W_output, z) + b_output)
        o_s.append(o)

        # Calculate hidden state
        h_prev = o * tanh(C_prev)
        h_s.append(h_prev)

        # Calculate logits
        v = np.dot(W_v, h_prev) + b_v
        v_s.append(v)

        # Calculate softmax
        output = softmax(v)     # why it sets all values as 1?
        output_s.append(output)


    fc_output = model_output(np.concatenate(output_s, axis=0).T, W_fc, b_fc)
    fc_output_s.append(fc_output)

    return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s, np.concatenate(fc_output_s, axis=0)



def backward(z, f, i, g, C, o, h, v, outputs, fc_outputs, targets, p, error):
    """
    Arguments:
    z -- your concatenated input data  as a list of size m.
    f -- your forget gate computations as a list of size m.
    i -- your input gate computations as a list of size m.
    g -- your candidate computations as a list of size m.
    C -- your Cell states as a list of size m+1.
    o -- your output gate computations as a list of size m.
    h -- your Hidden state computations as a list of size m+1.
    v -- your logit computations as a list of size m.
    outputs -- your outputs as a list of size m.
    targets -- your targets as a list of size m.
    p -- python list containing:
                        W_forget -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        b_forget -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        W_input -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        b_input -- Bias of the update gate, numpy array of shape (n_a, 1)
                        W_g -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        b_g --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        W_output -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        b_output --  Bias of the output gate, numpy array of shape (n_a, 1)
                        W_v -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_v, n_a)
                        b_v -- Bias relating the hidden-state to the output, numpy array of shape (n_v, 1)
    Returns:
    loss -- crossentropy loss for all elements in output
    grads -- lists of gradients of every element in p
    """

    # Unpack parameters
    W_forget, W_input, W_g, W_output, W_v, W_fc, b_forget, b_input, b_g, b_output, b_v, b_fc = p

    # Initialize gradients as zero
    W_forget_d = np.zeros_like(W_forget)
    b_forget_d = np.zeros_like(b_forget)

    W_input_d = np.zeros_like(W_input)
    b_input_d = np.zeros_like(b_input)

    W_g_d = np.zeros_like(W_g)
    b_g_d = np.zeros_like(b_g)

    W_output_d = np.zeros_like(W_output)
    b_output_d = np.zeros_like(b_output)

    W_v_d = np.zeros_like(W_v)
    b_v_d = np.zeros_like(b_v)

    W_fc_d = np.zeros_like(W_fc)
    b_fc_d = np.zeros_like(b_fc)

    # Set the next cell and hidden state equal to zero
    dh_next = np.zeros_like(h[0])
    dC_next = np.zeros_like(C[0])

    #dfc = 0
    #hiddiff = np.zeros((100, 1))
    #dv = 0


    dfc = np.dot(error, sigmoid(fc_outputs, derivative=True))       # internal error of fc layer
    W_fc_d = ETA * np.concatenate(outputs, axis=0) * dfc            # gradient of the fc layer weights

    #hiddiff = np.float64(np.multiply(dfc,W_fc))
    #dv = np.float64(np.dot(hiddiff.T, sigmoid(np.concatenate(outputs, axis=0), derivative=True)))    # error layer before fc

    #poprawka1 = ETA * input ' *delta1;

    for t in reversed(range(len(outputs))):

        # Compute the cross entropy
        #loss += -np.mean(np.log(outputs[t]) * targets)

        # Get the previous hidden cell state
        C_prev = C[t - 1]

        # Compute the derivative of the relation of the hidden-state to the output gate
        dv = np.copy(outputs[t])
        dv[np.argmax(targets[t])] -= 1

        # Update the gradient of the relation of the hidden-state to the output gate
        W_v_d += np.dot(dv, h[t].T)
        b_v_d += dv

        # Compute the derivative of the hidden state and output gate
        dh = np.dot(W_v.T, dv)
        dh += dh_next
        do = dh * tanh(C[t])
        do = sigmoid(o[t], derivative=True) * do

        # Update the gradients with respect to the output gate
        W_output_d += np.dot(do, z[t].T)
        b_output_d += do

        # Compute the derivative of the cell state and candidate g
        dC = np.copy(dC_next)
        dC += dh * o[t] * tanh(tanh(C[t]), derivative=True)
        dg = dC * i[t]
        dg = tanh(g[t], derivative=True) * dg

        # Update the gradients with respect to the candidate
        W_g_d += np.dot(dg, z[t].T)
        b_g_d += dg

        # Compute the derivative of the input gate and update its gradients
        di = dC * g[t]
        di = sigmoid(i[t], True) * di
        W_input_d += np.dot(di, z[t].T)
        b_input_d += di

        # Compute the derivative of the forget gate and update its gradients
        df = dC * C_prev
        df = sigmoid(f[t]) * df
        W_forget_d += np.dot(df, z[t].T)
        b_forget_d += df

        # Compute the derivative of the input and update the gradients of the previous hidden and cell state
        dz = (np.dot(W_forget.T, df)
              + np.dot(W_input.T, di)
              + np.dot(W_g.T, dg)
              + np.dot(W_output.T, do))
        dh_prev = dz[:HIDDEN_SIZE, :]
        dC_prev = f[t] * dC


    grads = W_forget_d, W_input_d, W_g_d, W_output_d, W_v_d, W_fc_d, b_forget_d, b_input_d, b_g_d, b_output_d, b_v_d, b_fc_d

    # Clip gradients
    grads = clip_gradient_norm(grads)

    return grads




################################## TRAINING LOOP ###################################

# Initialize a new network
params = init_lstm(hidden_size=HIDDEN_SIZE, input_size=INPUT_SIZE, z_size=Z_SIZE)

# Initialize hidden state as zeros
hidden_state = np.zeros((HIDDEN_SIZE, 1))

error = 0
# Track loss
training_loss, validation_loss = [], []

# For each epoch
for i in range(EPOCHS):

    # Track loss
    epoch_training_loss = 0
    epoch_validation_loss = 0

    # For each sentence in validation set, input is one sms
    for input, target in validation_set:

        target2 = np.zeros((1, 99))
        a = np.array(target)[np.newaxis]

        long_target = np.column_stack((target2, a))
        long_target = long_target.T

        # Initialize hidden state and cell state as zeros
        h = np.zeros((HIDDEN_SIZE, 1))
        c = np.zeros((HIDDEN_SIZE, 1))

        # Forward pass
        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, fc_outputs = forward(input, h, c, params)

        # outputs are outputs for whole email (100d size)
        # fc_outputs are outputs multiplied by fully connected layer weights (2d size)

        error = np.fabs(target - fc_outputs)
        # Backward pass
        g = backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, fc_outputs, long_target, params, error)  # or fc_outputs ?


        #terr += 0.5 * np.sum((error).^ 2);

        loss = error
        # Update loss
        epoch_validation_loss += loss


    # For each sms in training set
    for input, target in training_set:

        target2 = np.zeros((1, 99))
        a = np.array(target)[np.newaxis]

        long_target = np.column_stack((target2, a))
        long_target = long_target.T

        # Initialize hidden state and cell state as zeros
        h = np.zeros((HIDDEN_SIZE, 1))
        c = np.zeros((HIDDEN_SIZE, 1))

        # Forward pass
        z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, fc_outputs = forward(input, h, c, params)

        error = np.fabs(target - fc_outputs)
        #print("fc_outputs")
        #print(fc_outputs)
        loss = error


        # Backward pass
        grads = backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, fc_outputs, long_target, params, error)

        # Update parameters
        params = update_parameters(params, grads, lr=1e-1)

        # Update loss
        epoch_training_loss += loss

    # Save loss for plot
    training_loss.append(epoch_training_loss / len(training_set))
    validation_loss.append(epoch_validation_loss / len(validation_set))

    # Print loss every 1 epoch
    if i % 1 == 0:
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')


training_loss = np.concatenate(training_loss, axis=0)
validation_loss = np.concatenate(validation_loss, axis=0)


# For each sms in test set
for input, target in test_set:

    # Initialize hidden state as zeros
    h = np.zeros((HIDDEN_SIZE, 1))
    c = np.zeros((HIDDEN_SIZE, 1))

    # Forward pass
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, fc_outputs = forward(input, h, c, params)

    print(f'Target {target}, Prediction: {fc_outputs}')



# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss', )
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()
