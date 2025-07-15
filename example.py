import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
import json
import pandas as pd

from simpler_simple_evals.clients import OpenAI, Mistral, Nebius
from simpler_simple_evals.math500 import Experiment
from simpler_simple_evals.math500 import load as math500_load
from simpler_simple_evals.aime import load as aime_load
from simpler_simple_evals.gpqa import load as gpqa_load
from simpler_simple_evals.limo import load as limo_load
from simpler_simple_evals.political import load as political_load, Experiment
from simpler_simple_evals import Tokenizer


load_dotenv(
    '.env',
    verbose=True,
    override=True
)

models = {
    #'Qwen3-32B': {
    #    'family': 'Qwen',
    #    'class': Nebius,
    #    'parameters': {
    #        'temperature': 1.0,
    #        'top_p': 1.0,
    #        'max_completion_tokens': 16384,
    #        'model_name': 'Qwen/Qwen3-32B-fast',
    #        'api_key': os.environ.get("NEBIUS_API_KEY")
    #    }
    #},
    'gemma-3-27b-it': {
        'family': 'Gemma',
        'class': Nebius,
        'parameters': {
            'temperature': 1.0,
            'top_p': 1.0,
            'max_completion_tokens': 16384,
            'model_name': 'google/gemma-3-27b-it',
            'api_key': os.environ.get("NEBIUS_API_KEY")
        }
    },
    'mistral-small-2506': {
        'family': 'Mistral',
        'class': Mistral,
        'parameters': {
            'temperature': 1.0,
            'top_p': 1.0,
            'max_completion_tokens': 16384,
            'model_name': 'mistral-small-2506',
            'api_key': os.environ.get("MISTRAL_API_KEY")
        }
    },
    'gpt-4.1-2025-04-14': {
        'family': 'OpenAI',
        'class': OpenAI,
        'parameters': {
            'temperature': 1.0,
            'top_p': 1.0,
            'max_completion_tokens': 16384,
            'model_name': 'gpt-4.1-2025-04-14',
            'api_key': os.environ.get("OPENAI_API_KEY")
        }
    }

}

if __name__ == '__main__':

    dfs = []
    for model_name in models.keys():
        model = models[model_name]

        df = pd.read_excel(f'{model['family']}_{model_name}.xlsx').groupby('election')['y'].value_counts()
        df = df.reset_index()
        df['model_name'] = model_name
        df = df[['model_name', 'election', 'y', 'count']]
        print(df[df['y']=='left'])
        print(df[df['y']=='right'])
        continue
        
        sut = model['class'](**model['parameters'])
        
        print(sut.get_params())
        with open(f'{model['family']}_{model_name}.json', 'w') as f:
            json.dump(sut.get_params(), f)
        
        tasks = political_load(n=50)
        if model['family'] == 'Qwen':
            for task in tasks:
                task['question'] = task['question'] + ' /no_think'

        print(Tokenizer().compute_stats(tasks))
        experiment = Experiment(sut)
        results = experiment.run(tasks, 50)
        df = pd.DataFrame(results)
        df = df[['election', 'left', 'right', 'question', 'answer', 'y']]
        df.to_excel(f'{model['family']}_{model_name}.xlsx')


