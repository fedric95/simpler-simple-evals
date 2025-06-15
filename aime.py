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
        model_name='deepseek-ai/DeepSeek-R1-0528',
        temperature=0.6,
        max_completion_tokens=128000
    )

    split = 'italian'
    aime = load_dataset('fedric95/AIME2025-Multilingual', split, token=os.environ.get("HF_TOKEN"))
    aime_I = pd.DataFrame(aime['aime_2025_I'].to_dict())
    aime_II = pd.DataFrame(aime['aime_2025_II'].to_dict())
    
    # 3 e 12 ok
    #ids = [1,3,4,6,12]

    tasks = []
    for _, row in aime_I.iterrows():
        #if row.id not in ids:
        #    continue
        #print(f'Row ID: {row.id}')
        #import pdb
        #pdb.set_trace()
        question = QUERY_TEMPLATE[split].format(question=row.problem)
        tasks.append({'question': question, 'y': row.answer})

    #exit(0) 
    run = Experiment(sut)
    pool = multiprocessing.Pool(processes=15)
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