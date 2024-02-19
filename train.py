from dataclasses import asdict
import json
from typing import Any

import numpy as np
from keras import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.models import load_model

from toolkit.file_readers.tsv_file_reader import TsvFileReader
from toolkit.processors.token_dictionary_processor import TokenDictionaryProcessor

MAX_LINES = 20000
EPOCHS = 30
BASE_MODEL = None
RESULT_MODEL = 'translation.h5'

EMBEDDING_DIM = 128
HIDDEN_UNITS = 192

file_reader = TsvFileReader()

token_sequences = file_reader.tokenize_file('Sentence pairs in Japanese-English - 2024-02-18.tsv', MAX_LINES)
dictionary = TokenDictionaryProcessor.create_dictonary(token_sequences)

dumped_dict = json.dumps({
    key: asdict(value)
    for key, value in dictionary.items()
})

with open(f'dict_{RESULT_MODEL}', 'wb') as dumpfile:
    dumpfile.write(dumped_dict.encode('utf8'))

num_tokens = len(dictionary.keys())

numerical_sequences = [
    TokenDictionaryProcessor.process_token_sequence_to_numerical(tokens, dictionary)
    for tokens in token_sequences
]

X = [sequence[:-1] for sequence in numerical_sequences]
Y = [sequence[1:] for sequence in numerical_sequences]

if BASE_MODEL is not None:
    model: Any = load_model(BASE_MODEL)

    model.fit(np.array(X), np.array(Y), epochs=EPOCHS)
    model.save(RESULT_MODEL)

else:
    model = Sequential([
        Embedding(input_dim=num_tokens, output_dim=EMBEDDING_DIM),
        LSTM(HIDDEN_UNITS, return_sequences=True),
        Dense(num_tokens, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='Adam'
    )
    model.summary()
    a = np.array(X)
    b = np.array(Y)
    model.fit(np.array(X), np.array(Y), epochs=EPOCHS)
    model.save(RESULT_MODEL)

print(f'Training finished! Saved weights to {RESULT_MODEL}')
