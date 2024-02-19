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

input = 'これは猫ではない。'

ja_tokenizer = TokenizerFactory.create_tokenizer(Language.JAPANESE)
generated_sequence = TokenDictionaryProcessor.process_token_sequence_to_numerical(
    ja_tokenizer.tokenize_string(input),
    dictionary
)

result = []

for _ in range(TOKENS_TO_GENERATE):
    input_sequence = np.array([generated_sequence])
    predicted_token_probs = model.predict(input_sequence)

    candidate_probabilities = [
        {
            'index': idx,
            'probability': probability
        }
        for idx, probability in enumerate(predicted_token_probs[0][-1])

    ]
    candidate_probabilities.sort(key=lambda element: element['probability'], reverse=True)

    selected = candidate_probabilities[:TOKEN_CANDIDATES][int(random() * TOKEN_CANDIDATES)]

    if selected['index'] == 2:
        generated_sequence.append(selected['index'])
        break

    generated_sequence.append(selected['index'])

print([
    decoding_dictionary[token]
    for token in generated_sequence
])