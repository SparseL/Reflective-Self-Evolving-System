import requests


class InterfaceHF():

    def __init__(self,key,model_LLM,debug_mode):
        self.key = key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
   
    def get_response(self,prompt_content):

        API_URL = "https://api-inference.huggingface.co/models/"+self.model_LLM
        headers = {"Authorization": f"Bearer {self.key}"}
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        data = query(prompt_content)

        return data
