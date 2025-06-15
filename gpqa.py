import pandas as pd
from datasets import load_dataset
import os
from dotenv import load_dotenv
import random
import multiprocessing
from math500 import Experiment
from clients import OpenAI

load_dotenv()

QUERY_TEMPLATE_MULTICHOICE = {
    'english': """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""
}


if __name__== '__main__':
    
    sut = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
        model_name='deepseek-ai/DeepSeek-R1-0528',
        temperature=0.0,
        max_completion_tokens=2048
    )

    dataset = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', token=os.environ.get("HF_TOKEN"))
    dataset = pd.DataFrame(dataset['train'])
    rng = random.Random(0)

    tasks = []
    for _, row in dataset.iterrows():
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        
        permutation = rng.sample(range(4), 4)
        choices = [choices[i] for i in permutation]
        correct_index = choices.index(row["Correct Answer"])
        correct_answer = "ABCD"[correct_index]

        question = QUERY_TEMPLATE_MULTICHOICE['english'].format(
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            question=row['Question']
        )
        tasks.append({
            'question' : question,
            'y': correct_answer
        })

    run = Experiment(sut)
    pool = multiprocessing.Pool(processes=20)
    results = []
    for task in pool.imap(run, tasks):
        print(
            task['y_pred'],
            task['y'],
            task['match']
        )
        results.append(task)
    pool.close()

    #match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    #extracted_answer = match.group(1) if match else None
    #score = 1.0 if extracted_answer == correct_answer else 0.0
    #import pdb
    #pdb.set_trace()
