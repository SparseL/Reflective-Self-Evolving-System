import http.client
import json
import os


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

        # self.client = openai.OpenAI(base_url=self.api_endpoint,
        #                api_key=self.api_key)

    def sb_get_response(self, prompt_content):
        completion = self.client.chat.completions.create(
                model=self.model_LLM,
                messages=[
                    {"role": "user", "content": prompt_content},
                ]
        )

        return completion.choices[0].message.content

    def get_response(self, prompt_content, temp):
        payload_explanation = json.dumps(
                {
                    "model":    self.model_LLM,
                    "messages": [
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_content}
                    ],
                    "temperature": temp,
                }
        )

        headers = {
            "Authorization":    "Bearer " + self.api_key,
            "User-Agent":       "EOH-research-client/1.0",
            "Content-Type":     "application/json",
            "x-api2d-no-cache": 1,
        }

        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                if self.model_LLM == "local":
                    conn = http.client.HTTPConnection(self.api_endpoint)
                else:
                    conn = http.client.HTTPSConnection(self.api_endpoint)
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if self.debug_mode:
                    print("Error in API. Restarting the process...")
                continue

        return response

    def get_embedding(self, content):
        payload_explanation = json.dumps(
                {
                    "model":    "text-embedding-3-small",
                    "input": content
                }
        )

        headers = {
            "Authorization":    "Bearer " + self.api_key,
            "User-Agent":       "EOH-research-client/1.0",
            "Content-Type":     "application/json",
            "x-api2d-no-cache": 1,
        }

        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                if self.model_LLM == "local":
                    conn = http.client.HTTPConnection(self.api_endpoint)
                else:
                    conn = http.client.HTTPSConnection(self.api_endpoint)
                conn.request("POST", "/v1/embeddings", payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data['data'][0]['embedding']
                break
            except Exception as e:
                if self.debug_mode:
                    print("Error in API. Restarting the process...")
                continue

        return response

