import os
import random
import pandas as pd
from datasets import load_dataset

QUERY_TEMPLATE_MULTICHOICE = {
    'english': """Answer the following multiple choice problem. Please reason step by step, and put your final answer within \\boxed{{}}. The answer to the problem and is one of ABCD. Include only of these letters in the final answer.

{question}

A) {A}
B) {B}
C) {C}
D) {D}""",
}

def load():
    dataset = load_dataset(
        'Idavidrein/gpqa',
        'gpqa_diamond',
        token=os.environ.get("HF_TOKEN")
    )
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
            'question': question,
            'y': correct_answer
        })
    return tasks
