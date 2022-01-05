import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
import pickle

Dino = open('/home/kaleeswaran/Desktop/Lstm/Dinosaur.txt','r').read()
Dino = Dino.lower()

chars = sorted(list(set(Dino)))
mapping = {c:i for i,c in enumerate(chars)}

def subset(length):
    length = length
    sequences = []

    for i in range(length, len(Dino)):
        seq = Dino[i-length:i+1]
        sequences.append(seq)
    return(sequences)

# length of the characters you want the model to learn from
sequences = subset(4)

enc = []
for x in sequences:
    tem = [mapping[char] for char in x]
    enc.append(tem)

enc = np.array(enc)
X = enc[:,:-1]
y = enc[:,-1]

unique = len(mapping)
X = [to_categorical(x, num_classes = unique) for x in X]
X = np.array(X)
y = [to_categorical(yo, num_classes = unique) for yo in y]
y = np.array(y)

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(unique, activation='softmax'))

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X, y, epochs=100)

# third argument is the sequence length you initialised earlier and seed_text is the text from where you
# want the model to make prediction of the dinosaur name
def gen_seq(model, mapping, seq_length, seed_text):
	in_text = seed_text
	while in_text[-1] != '\n':
		encoded = [mapping[char] for char in in_text]
		encoded = pad_sequences([encoded], maxlen = seq_length, truncating = 'pre')
		encoded = to_categorical(encoded, num_classes = len(mapping))
		yhat = model.predict_classes(encoded, verbose = 0)
		out_char = ''
		for char, index  in mapping.items():
			if index == yhat:
				break
		in_text += char
		print(in_text)
	return in_text


print(gen_seq(model, mapping, 4, 'gat'))
