import re
import time
from ...llm.interface_LLM import InterfaceLLM
import pickle
import json
import numpy as np

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs):

        # set prompt interface
        #getprompts = GetPrompts()
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        self.prompt_init_guidance = prompts.get_init_guidance() if hasattr(prompts, "get_init_guidance") else ""
        self.aware_prompt_profile = prompts.get_aware_profile() if hasattr(prompts, "get_aware_profile") else "none"
        self.aware_auto_enabled = self.aware_prompt_profile == "auto"
        self.active_aware_prompt_profile = "none" if self.aware_auto_enabled else self.aware_prompt_profile
        self._get_prompt_aware_guidance = prompts.get_aware_guidance if hasattr(prompts, "get_aware_guidance") else None
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking


        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

        self.embedding_values = []
        self.embedding_vectors = np.zeros((0, 1536))
        self.reflection_memory_max_chars = int(kwargs.get("reflection_memory_max_chars", 4000) or 4000)
        self.reflection_memory_max_items = int(kwargs.get("reflection_memory_max_items", 50) or 50)
        self.reflection_memory_entries = []

    def append_reflection_memory(self, text, source=None):
        if not text:
            return
        val = str(text).strip()
        if not val:
            return
        if len(val) > 1200:
            val = val[:1200]
        entry = {"source": source, "text": val}
        self.reflection_memory_entries.append(entry)
        if len(self.reflection_memory_entries) > self.reflection_memory_max_items:
            self.reflection_memory_entries = self.reflection_memory_entries[-self.reflection_memory_max_items:]
        while self._reflection_memory_text_length() > self.reflection_memory_max_chars and len(self.reflection_memory_entries) > 1:
            self.reflection_memory_entries.pop(0)

    def _reflection_memory_text_length(self):
        return sum(len(e.get("text", "")) + 8 for e in self.reflection_memory_entries)

    def _reflection_memory_block(self):
        if not self.reflection_memory_entries:
            return ""
        lines = ["Reflection Memory (apply these learnings when designing the next algorithm):"]
        for e in self.reflection_memory_entries[-10:]:
            s = e.get("source")
            t = e.get("text", "")
            if s:
                lines.append(f"- [{s}] {t}")
            else:
                lines.append(f"- {t}")
        return "\n".join(lines) + "\n\n"

    def _aware_guidance_block(self):
        if self.active_aware_prompt_profile in (None, "", "none", "auto"):
            return ""
        if not self._get_prompt_aware_guidance:
            return ""
        guidance = self._get_prompt_aware_guidance(self.active_aware_prompt_profile)
        if not guidance:
            return ""
        return "Targeted Aware Guidance:\n" + guidance.strip() + "\n\n"

    def _aware_control_request(self):
        if not self.aware_auto_enabled:
            return ""
        return (
            "Aware Control: Based on the metrics and reflection, decide whether future prompts should inject a mechanism-level hint. "
            "Output one extra line exactly as `AwareProfile: none`, `AwareProfile: ci_boundary`, or `AwareProfile: ci_reinsertion`. "
            "Use ci_boundary when progress is plateauing and the current rules need local radius-boundary dismantling ideas. "
            "Use ci_reinsertion only after a stronger boundary-like strategy exists but early removals look redundant or lcc_at_k_frac / anc_prefix_k remain high. "
            "Otherwise use none.\n"
        )

    def _normalize_aware_profile(self, value):
        val = (value or "").strip().lower()
        if val in {"none", "ci_boundary", "ci_reinsertion"}:
            return val
        return None

    def set_active_aware_profile(self, profile, reason=None):
        val = self._normalize_aware_profile(profile)
        if val is None:
            return False
        if val == self.active_aware_prompt_profile:
            return False
        self.active_aware_prompt_profile = val
        if reason:
            self.append_reflection_memory(
                f"Auto aware profile switched to {val}: {reason}",
                source="aware_auto",
            )
        return True

    def add_embedding(self, offspring):
        if offspring["objective"]:
            vec = self.interface_llm.get_embedding(offspring["algorithm"])
            if isinstance(vec, list) or isinstance(vec, np.ndarray):
                self.embedding_values.append(offspring)
                self.embedding_vectors = np.vstack([self.embedding_vectors, vec])




    def get_metrics_description(self, indiv):
        desc = []
        if 'objective' in indiv and indiv['objective'] is not None:
            desc.append(f"Objective: {indiv['objective']}")
        
        if 'time_select' in indiv and indiv['time_select'] is not None:
            desc.append(f"Time Select: {indiv['time_select']:.4f}")
            
        if 'time_anc' in indiv and indiv['time_anc'] is not None:
            desc.append(f"Time ANC: {indiv['time_anc']:.4f}")
            
        if 'other_inf' in indiv and isinstance(indiv['other_inf'], dict):
            for k, v in indiv['other_inf'].items():
                if k in ['operator', 'parent']:
                    continue
                if isinstance(v, (int, float)):
                    desc.append(f"{k}: {v:.4f}")
                else:
                    desc.append(f"{k}: {v}")
                    
        if not desc:
            return ""
        return "Performance Metrics: " + ", ".join(desc)

    def get_prompt_i1(self):

        init_guidance = (self.prompt_init_guidance + "\n") if self.prompt_init_guidance else ""
        prompt_content = self.prompt_task+"\n"+self._reflection_memory_block()+self._aware_guidance_block()+init_guidance+\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content


    def get_prompt_e1(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            metrics = self.get_metrics_description(indivs[i])
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + metrics + "\n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"+self._reflection_memory_block()+self._aware_guidance_block()+\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

    def get_prompt_e2(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            metrics = self.get_metrics_description(indivs[i])
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + metrics + "\n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        prompt_content = self.prompt_task+"\n"+self._reflection_memory_block()+self._aware_guidance_block()+\
"I have "+str(len(indivs))+" existing algorithms with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. \n"\
"Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

    def get_prompt_m1(self,indiv1):
        metrics = self.get_metrics_description(indiv1)
        prompt_content = self.prompt_task+"\n"+self._reflection_memory_block()+self._aware_guidance_block()+\
"I have one algorithm with its code as follows. \n"\
+ metrics + "\n"\
"Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

    def get_prompt_m2(self,indiv1):
        metrics = self.get_metrics_description(indiv1)
        prompt_content = self.prompt_task+"\n"+self._reflection_memory_block()+self._aware_guidance_block()+\
"I have one algorithm with its code as follows. \n"\
+ metrics + "\n"\
"Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content

    def get_prompt_custom_1(self, pops):
        prompt_content = (self.prompt_task + "\n" + self._reflection_memory_block() + self._aware_guidance_block() + "I have some algorithms sorted by their performance (the later the better), please try to generate a better algorithm in similar format."
                          + "\n\n".join([self.get_metrics_description(pop) + "\n" + pop['algorithm'] for pop in pops]) + "\n\n Do NOT give additional explanations.")
        return prompt_content

    def get_prompt_custom_3(self, thought, pops):
        prompt_content = (self.prompt_task + "\n" + self._reflection_memory_block() + self._aware_guidance_block() + thought + "\n"
                + "Please try to improve a better algorithm in similar format based on the following algorithms which is sorted by their performance (the later the better). The new algorithm should not be the same as the one below. \n\n"
                          + "\n".join([f'{self.get_metrics_description(pop)} :{pop["algorithm"]}' for pop in pops]) + "\n\n Do NOT give additional explanations.")
        return prompt_content

    def get_prompt_custom_2(self, thought):
        prompt_content = self.prompt_task + "\n" + self._reflection_memory_block() + self._aware_guidance_block() + "Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. \n"\
"The main idea of algorithm is " + thought + \
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Do not give additional explanations."
        return prompt_content



    def get_prompt_thought_chain(self, pops):
        prompt_content = self.prompt_task + "\n" + self._aware_guidance_block()
        prompt_content += "I have a list of algorithms sorted by performance (the later the better):\n"
        for i, pop in enumerate(pops):
             metrics = self.get_metrics_description(pop)
             prompt_content += f"Algorithm {i+1}:\n{metrics}\nCode:\n{pop['code']}\n\n"
        
        prompt_content += "Please perform a multi-level analysis on these algorithms to identify the key bottleneck and propose a better strategy.\n"
        prompt_content += "1. Identify the common patterns and effective strategies in the high-performing algorithms.\n"
        prompt_content += "2. Analyze the potential limitations or bottlenecks of the current best algorithm.\n"
        prompt_content += "3. Propose a new, innovative strategy that addresses these limitations.\n"
        prompt_content += "Finally, implement this new strategy as a Python function.\n"
        prompt_content += "Output format:\n"
        prompt_content += "Analysis: {Your analysis here}\n"
        prompt_content += "Thought: {Your thought process for the new algorithm}\n"
        prompt_content += "Reflection: {Your reflection here}\n"
        prompt_content += self._aware_control_request()
        prompt_content += "Algorithm Description: {One sentence description inside braces}\n"
        prompt_content += "Code: \n```python\n...\n```\n"
        
        prompt_content += f"The function must be named {self.prompt_func_name}. This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. {self.prompt_inout_inf} {self.prompt_other_inf}\nDo not give additional explanations."
        
        return prompt_content

    def thought_chain(self, pops):
        prompt_content = self.get_prompt_thought_chain(pops)
        # Temporarily switch to a stronger model for deep thinking
        original_model = self.interface_llm.model_LLM
        self.interface_llm.model_LLM = "gpt-5" # or "gpt-4o"
        code_all, algorithm, meta = self._get_alg(prompt_content, return_meta=True)
        self.interface_llm.model_LLM = original_model
        return [code_all, algorithm, meta]

    def get_prompt_daily_reflection(self, pops):
        prompt_content = self.prompt_task + "\n" + self._aware_guidance_block()
        prompt_content += "I have a list of recent algorithms generated in this generation, sorted by performance (the later the better):\n"
        for i, pop in enumerate(pops):
             metrics = self.get_metrics_description(pop)
             prompt_content += f"Algorithm {i+1}:\n{metrics}\nCode:\n{pop['code']}\n\n"
        
        prompt_content += "Please provide a quick summary of the evolutionary progress in this generation.\n"
        prompt_content += "1. What small improvements were made?\n"
        prompt_content += "2. What failed?\n"
        prompt_content += "Based on this short-term reflection, propose a small tweak or optimization to the best algorithm.\n"
        prompt_content += "Output format:\n"
        prompt_content += "Reflection: {Your short reflection here}\n"
        prompt_content += self._aware_control_request()
        prompt_content += "Algorithm Description: {One sentence description inside braces}\n"
        prompt_content += "Code: \n```python\n...\n```\n"
        
        prompt_content += f"The function must be named {self.prompt_func_name}. This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. {self.prompt_inout_inf} {self.prompt_other_inf}\nDo not give additional explanations."
        
        return prompt_content

    def daily_reflection(self, pops):
        prompt_content = self.get_prompt_daily_reflection(pops)
        # Temporarily switch to a stronger model for daily reflection
        original_model = self.interface_llm.model_LLM
        self.interface_llm.model_LLM = "gpt-5-mini" # or "gpt-4o"
        code_all, algorithm, meta = self._get_alg(prompt_content, return_meta=True)
        self.interface_llm.model_LLM = original_model
        return [code_all, algorithm, meta]

    # def _get_alg(self,prompt_content):

    #     response = self.interface_llm.get_response(prompt_content)

    #     algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
    #     if len(algorithm) == 0:
    #         if 'python' in response:
    #             algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
    #         elif 'import' in response:
    #             algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
    #         else:
    #             algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

    #     code = re.findall(r"import.*return", response, re.DOTALL)
    #     if len(code) == 0:
    #         code = re.findall(r"def.*return", response, re.DOTALL)

    #     n_retry = 1
    #     while (len(algorithm) == 0 or len(code) == 0):
    #         if self.debug_mode:
    #             print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

    #         response = self.interface_llm.get_response(prompt_content)

    #         algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
    #         if len(algorithm) == 0:
    #             if 'python' in response:
    #                 algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
    #             elif 'import' in response:
    #                 algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
    #             else:
    #                 algorithm = re.findall(r'^.*?(?=def)', response,re.DOTALL)

    #         code = re.findall(r"import.*return", response, re.DOTALL)
    #         if len(code) == 0:
    #             code = re.findall(r"def.*return", response, re.DOTALL)

    #         if n_retry > 3:
    #             break
    #         n_retry +=1

    #     algorithm = algorithm[0]
    #     code = code[0]

    #     code_all = code+" "+", ".join(s for s in self.prompt_func_outputs)

    #     # print(">>> check code: \n", code_all)
    #     return [code_all, algorithm]

    def _extract_labeled_field(self, text, label):
        if not text:
            return None
        label_re = re.escape(label)
        stop = r"(?:\n\s*(?:Analysis|Thought|Reflection|AwareProfile|Algorithm Description|Code)\s*:|\Z)"
        m = re.search(rf"{label_re}\s*:\s*([\s\S]*?){stop}", text, flags=re.IGNORECASE)
        if not m:
            return None
        val = m.group(1).strip()
        return val or None

    def _get_alg(self, prompt_content, return_meta=False):

        def extract_algorithm(text):
            """提取 { ... } 中的摘要描述，允许包含多行。"""
            alg = re.findall(r"\{([\s\S]*?)\}", text)
            if alg:
                return alg[-1].strip()
            # fallback：取第一个 def 前的文本最为描述
            parts = re.split(r"```python|def\s+\w+\s*\(", text)
            head = parts[0].strip()
            head = re.sub(r"[\s`#*]+$", "", head)
            return head if head else "No algorithm description found."

        def extract_python_code(text):
            """从文本中 robust 捕获 python 代码。"""
            # 1) 优先解析三引号 code block
            blocks = re.findall(r"```python([\s\S]*?)```", text)
            if blocks:
                return blocks[-1].strip()

            # 2) 捕获最后一个 def 开头的函数（直到文件末尾）
            func_matches = list(re.finditer(r"(def\s+\w+\s*\([\s\S]*?$)", text))
            if func_matches:
                return func_matches[-1].group(1).strip()

            # 3) 捕获 def ... return 结构
            dr = re.findall(r"(def[\s\S]*?return[^\n]*)", text)
            if dr:
                return dr[-1].strip()

            # 4) fallback：尝试从 import 开始
            imp = re.findall(r"(import[\s\S]*?return[^\n]*)", text)
            if imp:
                return imp[-1].strip()

            return None

        # -----------------------------
        #       执行 LLM 调用 + 重试
        # -----------------------------
        MAX_RETRY = 5
        wait = 1.0
        last_response = None

        for attempt in range(MAX_RETRY):

            response = self.interface_llm.get_response(prompt_content)
            last_response = response

            # 清洗 markdown 杂质
            clean = response.replace("```", "").replace("**", "").strip()

            # 提取 algorithm
            algorithm = extract_algorithm(clean)

            # 提取 python code
            code = extract_python_code(response)

            # 成功
            if code:
                if return_meta:
                    meta = {
                        "raw_response": response,
                        "analysis": self._extract_labeled_field(response, "Analysis"),
                        "thought": self._extract_labeled_field(response, "Thought"),
                        "reflection": self._extract_labeled_field(response, "Reflection"),
                        "aware_profile": self._normalize_aware_profile(self._extract_labeled_field(response, "AwareProfile")),
                    }
                    return code, algorithm, meta
                return [code, algorithm]

            # 失败 → 等待 + 重试
            time.sleep(wait)
            wait *= 1.6

        # -----------------------------
        #     多次失败后的 fallback
        # -----------------------------
        fallback_code = "# No valid code extracted."
        fallback_alg = "Parsing failed after retries."
        if return_meta:
            meta = {
                "raw_response": last_response,
                "analysis": self._extract_labeled_field(last_response, "Analysis") if last_response else None,
                "thought": self._extract_labeled_field(last_response, "Thought") if last_response else None,
                "reflection": self._extract_labeled_field(last_response, "Reflection") if last_response else None,
                "aware_profile": self._normalize_aware_profile(self._extract_labeled_field(last_response, "AwareProfile")) if last_response else None,
            }
            return fallback_code, fallback_alg, meta
        return [fallback_code, fallback_alg]


    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            # input()

        return [code_all, algorithm]

    def e1(self,parents):

        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e2(self,parents):

        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m1(self,parents):

        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m2(self,parents):

        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def custom(self, parents):
        parents.sort(key=lambda x: x["objective"])
        prompt_content_1 = self.get_prompt_custom_1(parents)
        thought = self.interface_llm.get_response(prompt_content_1, temp=1.2)

        use_embed = self.embedding_vectors.shape[0] > 0
        thought_embedding = None
        if use_embed:
            thought_embedding = self.interface_llm.get_embedding(thought)
            use_embed = isinstance(thought_embedding, list) or isinstance(thought_embedding, np.ndarray)

        if use_embed:
            similarities = np.dot(self.embedding_vectors, thought_embedding)
            candidate_indices = np.argsort(similarities)[-4:]
            parents = [self.embedding_values[k] for k in candidate_indices]
            parents += [max(self.embedding_values, key=lambda x: x["objective"]) ]
        else:
            k = min(5, len(parents))
            parents = parents[-k:]

        parents.sort(key=lambda x: x["objective"])
        prompt_content_3 = self.get_prompt_custom_3(thought, parents)
        thought = self.interface_llm.get_response(prompt_content_3, temp=1.2)

        prompt_content = self.get_prompt_custom_2(thought)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            # input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
