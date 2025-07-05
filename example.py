import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import HfApi
import json

from simpler_simple_evals.clients import OpenAI, Mistral
from simpler_simple_evals.math500 import Experiment
from simpler_simple_evals.math500 import load as math500_load
from simpler_simple_evals.aime import load as aime_load
from simpler_simple_evals.gpqa import load as gpqa_load
from simpler_simple_evals.limo import load as limo_load
from simpler_simple_evals import Tokenizer




load_dotenv(
    '.env',
    verbose=True,
    override=True
)

if __name__ == '__main__':

    model_family = 'Qwen'
    model_name = 'Qwen3-4B'
    mode = 'fast'
    if mode == 'base':
        end_point = f'{model_family}/{model_name}'
    elif mode == 'fast':
        end_point = f'{model_family}/{model_name}-fast'
    else:
        raise Exception('Unexpected')
    del mode
    sut = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
        model_name=end_point,
        temperature=0.6,
        max_completion_tokens=38912,
        top_p=0.95,
        top_k=20
    )

    #model_family = 'mistral'
    #model_name = 'mistral-small-2506'
    #end_point = 'mistral-small-2506'
    #sut = Mistral(
    #    api_key=os.environ.get('MISTRAL_API_KEY'),
    #    model_name=model_name,
    #    temperature=0.15
    #)
    
    print(sut.get_params())
    with open(f'{model_family}_{model_name}.json', 'w') as f:
        json.dump(sut.get_params(), f)
    
    tasks = limo_load()
    for task in tasks:
        task['question'] = task['question'] + ' /no_think'
    
    
    print(Tokenizer().compute_stats(tasks))
    experiment = Experiment(sut)
    results = experiment.run(tasks, 40)
    df = pd.DataFrame(results)
    df = df[['question', 'answer', 'y_pred', 'y', 'match']]
    df_hf = Dataset.from_pandas(df)

    df_hf.push_to_hub(f'fedric95/LIMO-{model_name}', private=True)
    df.to_excel(f'{model_family}_{model_name}.xlsx')