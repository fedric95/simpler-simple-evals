import os
import pandas as pd
from datasets import load_dataset
import re
import multiprocessing

from .clients import OpenAI


QUERY_TEMPLATE = {
    'english': """Answer the following problem. Please reason step by step, and put your final answer within \\boxed{{}}.

{question}"""
}
ANSWER_TEMPLATE = r"(?i)\\boxed\{\s*([^\n]+)\s*\}"


EQUALITY_TEMPLATE = """Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: {expression1}
    Expression 2: {expression2}"""

class Experiment:

    def __init__(self, sut):
        self.judge = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=None,
            model_name='gpt-4.1',
            temperature=0.0,
            max_completion_tokens=None
        )
        self.sut = sut
        
    def __call__(self, task):
        task['answer'] = self.sut(task['question'])
        match = re.findall(ANSWER_TEMPLATE, task['answer'])
        task['y_pred'] = match[-1] if len(match) else None

        if task['y_pred'] is None:
            task['match'] = 'No'
            return task
        
        if task['y_pred'] == task['y']:
            task['match'] = 'Yes'
            return task

        content = EQUALITY_TEMPLATE.format(expression1=task['y_pred'], expression2=task['y'])
        task['match'] = self.judge(content)

        ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
        task['answer'] = ILLEGAL_CHARACTERS_RE.sub(r'', task['answer'])
        return task


    def run(self, tasks, processes=20):
        pool = multiprocessing.Pool(processes=min(len(tasks), processes))
        results = []
        for task in pool.imap(self, tasks):
            print(
                task['y_pred'],
                task['y'],
                task['match']
            )
            results.append(task)
        pool.close()
        return results

def load():
    dataset = load_dataset('HuggingFaceH4/MATH-500', token=os.environ.get("HF_TOKEN"))
    dataset = pd.DataFrame(dataset['test'].to_dict())
    tasks = []
    for _, row in dataset.iterrows():
        question = QUERY_TEMPLATE['english'].format(question=row.problem)
        tasks.append({'question': question, 'y': row.answer})
    return tasks