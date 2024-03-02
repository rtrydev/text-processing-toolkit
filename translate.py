import json
from random import random
from typing import Any

import numpy as np
from keras.models import load_model

from toolkit.classes.token import Token
from toolkit.enums.language import Language
from toolkit.factories.tokenizer_factory import TokenizerFactory
from toolkit.processors.token_dictionary_processor import TokenDictionaryProcessor


TOKENS_TO_GENERATE = 20
TOKEN_CANDIDATES = 1
MAX_SEQ_LENGTH = 64

MODEL = 'translation.h5'

with open(f'dict_{MODEL}', 'r', encoding='utf8') as tokendump:
    dumped_dict = tokendump.read()

dict_json = json.loads(dumped_dict)
dictionary = {
    key: Token(
        body=entry.get('body'),
        data=entry.get('data')
    )
    for key, entry in dict_json.items()
}

decoding_dictionary = {
    entry.data['id']: entry.body
    for entry in dictionary.values()
}

num_tokens = len(dictionary.keys())

model: Any = load_model(MODEL)

input = '猫が好き。'

ja_tokenizer = TokenizerFactory.create_tokenizer(Language.JAPANESE)
print(ja_tokenizer.tokenize_string(input))

generated_sequence = TokenDictionaryProcessor.process_token_sequence_to_numerical(
    [
        Token('<|start|>', {'id': 1}),
        *ja_tokenizer.tokenize_string(input),
        Token('<|separator|>', {'id': 4})
    ],
    dictionary
)

result = []

for _ in range(TOKENS_TO_GENERATE):
    input_sequence = np.array([generated_sequence[-MAX_SEQ_LENGTH:]])  # Use only recent context
    predicted_token_probs = model.predict(input_sequence)[0][-1]  # Get last timestep predictions

    # Implement a different selection strategy if necessary, e.g., temperature sampling
    selected_index = np.random.choice(range(len(predicted_token_probs)), p=predicted_token_probs)

    if selected_index == 2:
        generated_sequence.append(selected_index)
        break

    generated_sequence.append(selected_index)

print([
    decoding_dictionary[token]
    for token in generated_sequence
])