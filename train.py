from dataclasses import asdict
import json
from typing import Any

import numpy as np
from keras import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.models import load_model
from keras.callbacks import LearningRateScheduler

from toolkit.file_readers.tsv_file_reader import TsvFileReader
from toolkit.processors.token_dictionary_processor import TokenDictionaryProcessor

def schedule(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.5
    else:
        return lr

MAX_LINES = 10000
EPOCHS = 10

BASE_MODEL = None
INITIAL_EPOCH = 0

RESULT_MODEL = 'translation.h5'

EMBEDDING_DIM = 1024
HIDDEN_UNITS = 2048

file_reader = TsvFileReader()

token_sequences = file_reader.tokenize_file('sentences-ja-en.tsv', MAX_LINES)
dictionary = TokenDictionaryProcessor.create_dictionary(token_sequences)

dumped_dict = json.dumps({
    key: asdict(value)
    for key, value in dictionary.items()
})

with open(f'dict_{RESULT_MODEL}', 'wb') as dumpfile:
    dumpfile.write(dumped_dict.encode('utf8'))

num_tokens = len(dictionary.keys()) + 4

numerical_sequences = [
    TokenDictionaryProcessor.process_token_sequence_to_numerical(tokens, dictionary)
    for tokens in token_sequences
]

X = np.array([np.array(sequence[:-1]) for sequence in numerical_sequences])
Y = np.array([np.array(sequence[1:]) for sequence in numerical_sequences])

lr_scheduler = LearningRateScheduler(schedule)

if BASE_MODEL is not None:
    model: Any = load_model(BASE_MODEL)
else:
    model = Sequential([
        Embedding(input_dim=num_tokens, output_dim=EMBEDDING_DIM),
        Dropout(0.3),
        LSTM(HIDDEN_UNITS, return_sequences=True),
        Dropout(0.3),
        Dense(num_tokens, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

model.fit(X, Y, epochs=EPOCHS, callbacks=[lr_scheduler], batch_size=64, validation_split=0.1, initial_epoch=INITIAL_EPOCH)
model.save(RESULT_MODEL)

print(f'Training finished! Saved weights to {RESULT_MODEL}')