import os

from simpler_simple_evals.clients import OpenAI
from simpler_simple_evals.math500 import Experiment
from simpler_simple_evals.aime import load


if __name__ == '__main__':
    sut = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
        model_name='Qwen/Qwen3-4B-fast-LoRa:last-wtTM',
        temperature=0.6,
        max_completion_tokens=38912, # based on suggestion in HuggingFace
        top_p=0.95,
        top_k=20
    )
    print(sut.get_params())
    tasks = load()[:3]
    experiment = Experiment(sut)
    results = experiment.run(tasks, 20)