import pandas as pd
from datasets import load_dataset
import os
from dotenv import load_dotenv
import multiprocessing
from clients import OpenAI
from math500 import Experiment, QUERY_TEMPLATE


load_dotenv()

if __name__ == '__main__':

    sut = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
        model_name='Qwen/Qwen3-4B-fast',
        temperature=0.6,
        max_completion_tokens=38912, # based on suggestion in HuggingFace
        top_p=0.95,
        top_k=20
    )
    print(sut.get_params())

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

    #exit(0) 
    run = Experiment(sut)
    pool = multiprocessing.Pool(processes=min(len(tasks), 20))
    results = []
    for task in pool.imap(run, tasks):
        print(
            task['y_pred'],
            task['y'],
            task['match']
        )
        results.append(task)
    pool.close()

    df = pd.DataFrame(results)
    df.to_excel('ft_new.xlsx')