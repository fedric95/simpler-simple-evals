
import tiktoken
import numpy as np

class Tokenizer:
    def __init__(self):
        self.model = tiktoken.encoding_for_model('gpt-4o')
    
    def __call__(self, text):
        return len(self.model.encode(text))

    def compute_stats(self, rows):
        tokens = []
        for row in rows:
            encoded_question = self(row['question'])
            tokens.append(encoded_question)
        return {
            'Max number of tokens in an instance': float(np.max(tokens)),
            'Average number of tokens in an instance': float(np.average(tokens)),
            'Number of tokens (M)': float(np.sum(tokens)/1000000),
        }
