import os
import pandas as pd
from datasets import load_dataset

from .math500 import QUERY_TEMPLATE

def load():
    split = 'english'
    aime = load_dataset('fedric95/AIME2025-Multilingual', split, token=os.environ.get("HF_TOKEN"))
    aime_I = pd.DataFrame(aime['aime_2025_I'].to_dict())
    aime_II = pd.DataFrame(aime['aime_2025_II'].to_dict())
    
    tasks = []
    for _, row in aime_I.iterrows():
        question = QUERY_TEMPLATE[split].format(question=row.problem)
        tasks.append({'question': question, 'y': row.answer})

    for _, row in aime_II.iterrows():
        question = QUERY_TEMPLATE[split].format(question=row.problem)
        tasks.append({'question': question, 'y': row.answer})
    
    return tasks
