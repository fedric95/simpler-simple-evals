import pandas as pd
from datasets import load_dataset
import os
import re
from dotenv import load_dotenv
import multiprocessing
from clients import OpenAI

load_dotenv()

QUERY_TEMPLATE = {
    'english': """Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.""",

    'italian': """Risolvi il seguente problema di matematica passo dopo passo. L’ultima riga della tua risposta dovrebbe essere nella forma Answer: $ANSWER (senza virgolette), dove $ANSWER è la risposta al problema.

{question}

Ricorda di mettere la tua risposta su una riga a parte dopo "Answer:", e non è necessario usare il comando \\boxed."""
}


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
        match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", task['answer'])
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


if __name__ == '__main__':
    
    sut = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
        model_name='deepseek-ai/DeepSeek-R1-0528',
        temperature=0.0,
        max_completion_tokens=2048
    )

    dataset = load_dataset('HuggingFaceH4/MATH-500', token=os.environ.get("HF_TOKEN"))
    dataset = pd.DataFrame(dataset['test'].to_dict())

    tasks = []
    for _, row in dataset.iterrows():
        question = QUERY_TEMPLATE['english'].format(question=row.problem)
        tasks.append({'question': question, 'y': row.answer})

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

    df = pd.DataFrame(results)
    df.to_excel('ft_new.xlsx')