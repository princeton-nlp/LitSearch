import openai
from typing import List

class OPENAIBaseEngine():
    def __init__(self, model_name: str, azure: bool = True):
        if not azure:
            raise NotImplementedError("Only Azure API is supported")
        
        self.model_name = model_name
        self.client = openai.AzureOpenAI()
    
    def safe_completion(self, messages: List[dict], max_tokens: int = 2000, temperature: float = 0, top_p: float = 1):
        args_dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_p == 1.0: 
            args_dict.pop("top_p")
        
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, **args_dict).to_dict()
        return {
            "finish_reason": response["choices"][0]["finish_reason"],
            "content": response["choices"][0]["message"]["content"]
        }
    
    def test_api(self):
        print("Testing API connection")
        messages = [{"role": "user", "content": "Why did the chicken cross the road?"}]
        response = self.safe_completion(messages=messages, max_tokens=20, temperature=0, top_p=1.0)
        content = response["content"]

        if response["finish_reason"] == 'api_error':
            print(f'Error in connecting to API: {response}')
        else:
            print(f'Successful API connection: {content}')




    
