import os
import openai
import mistralai

class Mistral:

    def __init__(
        self,
        api_key,
        model_name,
        temperature=None,
        top_p=None,
        max_completion_tokens=None
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens

    def get_params(self):
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_completion_tokens': self.max_completion_tokens,
            'top_p': self.top_p
        }

    def __call__(self, text):
        client = mistralai.Mistral(
            api_key=self.api_key
        )

        max_tries = 50
        tries = 0
        
        while True:
            try:
                completion = client.chat.complete(
                    model=self.model_name,
                    messages=[{ "role": "user", "content": text}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_completion_tokens
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(e)
                tries += 1
                if tries == max_tries:
                    raise e

class Nebius:
    def __init__(
        self,
        api_key,
        model_name,
        temperature=None,
        top_p=None,
        max_completion_tokens=None,
        top_k=None
    ):
        self.base_url = 'https://api.studio.nebius.com/v1/'
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


class OpenAI:
    def __init__(
        self,
        api_key,
        model_name,
        temperature=None,
        top_p=None,
        max_completion_tokens=None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
    
    def get_params(self):
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_completion_tokens': self.max_completion_tokens,
            'top_p': self.top_p
        }

    def __call__(
        self,
        text,
        response_format=None
    ):
        client = openai.OpenAI(
            api_key=self.api_key,
        )

        max_tries = 50
        tries = 0
        
        while True:
            try:
                if response_format is None:
                    completion = client.chat.completions.create(
                        model=self.model_name,
                        temperature=self.temperature,
                        messages=[{ 'role': 'user', 'content': text}],
                        max_completion_tokens=self.max_completion_tokens,
                        top_p=self.top_p
                    )
                    return completion.choices[0].message.content
                else:
                    completion = client.beta.chat.completions.parse(
                        model=self.model_name,
                        temperature=self.temperature,
                        messages=[{ 'role': 'user', 'content': text}],
                        max_completion_tokens=self.max_completion_tokens,
                        top_p=self.top_p,
                        response_format=response_format
                    )
                    return completion.choices[0].message.parsed
            except Exception as e:
                print(e)
                tries += 1
                if tries == max_tries:
                    raise e

class AzureOpenAI:
    def __init__(
        self,
        api_key,
        api_version,
        azure_endpoint,
        model_name,  # deployment
        temperature=None,
        top_p=None,
        max_completion_tokens=None,
        top_k=None
    ):
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
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
        client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
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
