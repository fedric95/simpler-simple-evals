import pandas as pd
from datasets import load_dataset
import os

from .math500 import QUERY_TEMPLATE


def load():
    dataset = load_dataset('GAIR/LIMO', token=os.environ.get("HF_TOKEN"))
    dataset = pd.DataFrame(dataset['train'].to_dict())
    tasks = []
    for _, row in dataset.iterrows():
        question = QUERY_TEMPLATE['english'].format(question=row.question)
        tasks.append({'question': question, 'y': row.answer})
    return tasks
