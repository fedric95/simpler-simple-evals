import openai

class OpenAI:
    def __init__(
        self,
        base_url,
        api_key,
        model_name,
        temperature=None,
        top_p=None,
        max_completion_tokens=None,
        top_k=None
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.top_k = top_k
    
    def get_params(self):
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_completion_tokens': self.max_completion_tokens,
            'top_p': self.top_p,
            'top_k': self.top_k
        }

    def __call__(self, text):
        client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        max_tries = 50
        tries = 0
        
        if self.top_k is not None:
            extra_body = {
                "top_k": 20
            }
        else:
            extra_body = None
        
        while True:
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    messages=[{ 'role': 'user', 'content': text}],
                    max_completion_tokens=self.max_completion_tokens,
                    max_tokens=self.max_completion_tokens,
                    top_p=self.top_p,
                    extra_body=extra_body
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(e)
                tries += 1
                if tries == max_tries:
                    raise e