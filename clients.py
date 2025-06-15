import openai

class OpenAI:
    def __init__(
        self,
        base_url,
        api_key,
        model_name,
        temperature,
        max_completion_tokens,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_completition_tokens = max_completion_tokens
    
    def get_params(self):
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_completition_tokens': self.max_completion_tokens
        }

    def __call__(self, text):
        client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        max_tries = 10
        tries = 0
        while True:
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    messages=[{ 'role': 'user', 'content': text}]
                )
                return completion.choices[0].message.content
            except Exception as e:
                print('retry')
                tries += 1
                if tries == max_tries:
                    raise e
