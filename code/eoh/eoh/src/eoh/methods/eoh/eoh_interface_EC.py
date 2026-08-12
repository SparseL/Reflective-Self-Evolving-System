import math
import multiprocessing
import pickle
from hashlib import md5
import requests
import numpy as np
import time
import os
import signal
from .eoh_evolution import Evolution
import warnings
import json
from joblib import Parallel, delayed, parallel_backend
try:
  from pebble import ProcessPool, ThreadPool
except Exception:
  ProcessPool = None
  ThreadPool = None
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
try:
  from memory_profiler import profile
except Exception:
  def profile(func):
    return func
from hashlib import md5


die_code = """

def select_next_node(graph, unv):
    n = 1
    for _ in range(100000):
        for i in range(100000):
            n += i     

"""

def simple_fun(x):
  return x*x

class InterfaceEC():
  instance = None
  def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):
    InterfaceEC.instance = self
    # LLM settings
    self.pop_size = pop_size
    self.interface_eval = interface_prob
    prompts = interface_prob.prompts
    self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs)
    self.m = m
    self.debug = debug_mode

    if not self.debug:
      warnings.filterwarnings("ignore")

    self.select = select
    self.n_p = n_p

    self.timeout = timeout
    self.use_numba = use_numba

  def code2file(self, code):
    with open("./ael_alg.py", "w") as file:
      # Write the code to the file
      file.write(code)
    return

  def add2pop(self, population, offspring):
    for ind in population:
      if ind['objective'] == offspring['objective']:
        if self.debug:
          print("duplicated result, retrying ... ")
        return False
    population.append(offspring)
    return True

  def check_duplicate(self, population, code):
    for ind in population:
      if code == ind['code']:
        return True
    return False

  # def population_management(self,pop):
  #     # Delete the worst individual
  #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
  #     return pop_new

  # def parent_selection(self,pop,m):
  #     ranks = [i for i in range(len(pop))]
  #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
  #     parents = random.choices(pop, weights=probs, k=m)
  #     return parents

  def population_generation(self):

    n_create = 2

    population = []

    for i in range(n_create):
      _, pop = self.get_algorithm([], 'i1')
      for p in pop:
        population.append(p)

    return population

  def population_generation_seed(self, seeds, n_p=None):

    population = []

    if n_p is None:
      n_p = getattr(self, "n_p", 1) or 1

    resolved = []
    for s in seeds:
      payload = None
      code = s.get('code')
      if (code is None or code == "") and s.get('source_json'):
        with open(s['source_json'], 'r', encoding='utf-8') as f:
          payload = json.load(f)
        code = payload.get('code')
      if code is None or code == "":
        raise ValueError("Seed entry must provide 'code' or 'source_json' with a 'code' field inside.")
      alg_name = s.get('algorithm')
      if alg_name is None or alg_name == "":
        alg_name = payload.get('algorithm') if payload else "seed"
      resolved.append({'algorithm': alg_name, 'code': code})

    print(f"Seed initialization: evaluating {len(seeds)} seed algorithms with n_jobs={n_p} ...", flush=True)
    fitness = Parallel(n_jobs=n_p, verbose=10)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in resolved)

    for i in range(len(seeds)):
      try:
        seed_alg = {
          'algorithm': resolved[i]['algorithm'],
          'code':      resolved[i]['code'],
          'objective': None,
          'other_inf': None,
          'problem': None
        }
        seed_alg['unique_id'] = md5(seed_alg['code'].encode()).hexdigest()

        val = fitness[i]
        if isinstance(val, Exception):
          seed_alg['problem'] = str(val)
          population.append(seed_alg)
          continue
        times = None
        if isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], dict):
          times = val[1]
          val = val[0]

        seed_alg['objective'] = np.round(float(val), 5)
        if times:
          seed_alg['time_select'] = times.get('time_select')
          seed_alg['time_anc'] = times.get('time_anc')
          metrics = {k: v for k, v in times.items() if k not in ('time_select', 'time_anc')}
          if metrics:
            seed_alg['other_inf'] = metrics
        population.append(seed_alg)

      except Exception as e:
        raise e


    print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")

    return population

  def _get_alg(self, pop, operator):
    offspring = {
      'algorithm': None,
      'code':      None,
      'objective': None,
      'other_inf': None,
      'problem':   None,
      'time':      None,
      'time_select': None,
      'time_anc': None
    }
    if operator == "i1":
      parents = []
      [offspring['code'], offspring['algorithm']] = self.evol.i1()
    elif operator == "e1":
      parents = self.select.parent_selection(pop, self.m)
      [offspring['code'], offspring['algorithm']] = self.evol.e1(parents)
    elif operator == "e2":
      parents = self.select.parent_selection(pop, self.m)
      [offspring['code'], offspring['algorithm']] = self.evol.e2(parents)
    elif operator == "m1":
      parents = self.select.parent_selection(pop, 1)
      [offspring['code'], offspring['algorithm']] = self.evol.m1(parents[0])
    elif operator == "m2":
      parents = self.select.parent_selection(pop, 1)
      [offspring['code'], offspring['algorithm']] = self.evol.m2(parents[0])
    elif operator == "cu":
      parents = self.select.parent_selection(pop, 5)
      [offspring['code'], offspring['algorithm']] = self.evol.custom(parents)
    elif operator == "t1":
      # Select top 5 individuals for thought chain analysis
      # Assuming pop is not necessarily sorted, we sort it here
      sorted_pop = sorted(pop, key=lambda x: x['objective'] if x['objective'] is not None else -float('inf'))
      parents = sorted_pop[-5:] if len(sorted_pop) >= 5 else sorted_pop
      code, alg, meta = self.evol.thought_chain(parents)
      offspring['code'] = code
      offspring['algorithm'] = alg
      if isinstance(meta, dict):
        offspring['llm_raw_response'] = meta.get('raw_response')
        offspring['llm_analysis'] = meta.get('analysis')
        offspring['llm_thought'] = meta.get('thought')
        offspring['llm_reflection'] = meta.get('reflection')
        offspring['llm_aware_profile'] = meta.get('aware_profile')
        self.evol.append_reflection_memory(meta.get('reflection'), source="t1")
    elif operator == "t2":
      # Daily reflection: use top 3 from current batch
      sorted_pop = sorted(pop, key=lambda x: x['objective'] if x['objective'] is not None else -float('inf'))
      parents = sorted_pop[-3:] if len(sorted_pop) >= 3 else sorted_pop
      code, alg, meta = self.evol.daily_reflection(parents)
      offspring['code'] = code
      offspring['algorithm'] = alg
      if isinstance(meta, dict):
        offspring['llm_raw_response'] = meta.get('raw_response')
        offspring['llm_reflection'] = meta.get('reflection')
        offspring['llm_analysis'] = meta.get('analysis')
        offspring['llm_thought'] = meta.get('thought')
        offspring['llm_aware_profile'] = meta.get('aware_profile')
        self.evol.append_reflection_memory(meta.get('reflection'), source="t2")
    else:
      print(f"Evolution operator [{operator}] has not been implemented ! \n")
    offspring["other_inf"] = {"operator": operator, "parent": [po["unique_id"] for po in parents]}
    offspring['unique_id'] = md5(offspring['code'].encode()).hexdigest()
    return parents, offspring



  def get_offspring(self, pop, operator):

    try:
      p, offspring = self._get_alg(pop, operator)

      if self.use_numba:

        # Regular expression pattern to match function definitions
        pattern = r"def\s+(\w+)\s*\(.*\):"

        # Search for function definitions in the code
        match = re.search(pattern, offspring['code'])

        function_name = match.group(1)

        code = add_numba_decorator(program=offspring['code'], function_name=function_name)
      else:
        code = offspring['code']

      n_retry = 1
      while self.check_duplicate(pop, offspring['code']):

        n_retry += 1
        if self.debug:
          print("duplicated code, wait 1 second and retrying ... ")

        p, offspring = self._get_alg(pop, operator)

        if self.use_numba:
          # Regular expression pattern to match function definitions
          pattern = r"def\s+(\w+)\s*\(.*\):"

          # Search for function definitions in the code
          match = re.search(pattern, offspring['code'])

          function_name = match.group(1)

          code = add_numba_decorator(program=offspring['code'], function_name=function_name)
        else:
          code = offspring['code']

        if n_retry > 1:
          break

      # self.code2file(offspring['code'])
      # fitness = self.interface_eval.evaluate(code)

      with multiprocessing.Pool(1) as pool:
        fitness = None
        try:
          t = time.time()
          future = pool.apply_async(self.interface_eval.evaluate, args=[code])
          fitness = future.get(self.timeout)

        except multiprocessing.TimeoutError:
          # print(f"Evaluation timed out after {self.timeout} seconds")
          pool.terminate()
          pool.join()
          offspring['problem'] = f"Timeout after {self.timeout}s"
          offspring['time'] = time.time() - t
        if isinstance(fitness, Exception):
          print("execute code, ", time.time() - t, fitness)
          try:
            offspring['problem'] = str(fitness)
          except Exception:
            offspring['problem'] = 'Exception during evaluation'
          offspring['time'] = time.time() - t
        elif fitness is not None:
          val = fitness
          times = None
          if isinstance(fitness, tuple) and len(fitness) == 2 and isinstance(fitness[1], dict):
            val = fitness[0]
            times = fitness[1]
          print(f"execute code, {time.time() - t}, [[{val}]]")
          offspring['objective'] = np.round(val, 5)
          offspring['problem'] = None
          offspring['time'] = time.time() - t
          if times:
            offspring['time_select'] = times.get('time_select')
            offspring['time_anc'] = times.get('time_anc')
            metrics = {k: v for k, v in times.items() if k not in ('time_select', 'time_anc')}
            if metrics:
              if isinstance(offspring.get('other_inf'), dict):
                offspring['other_inf'].update(metrics)
              else:
                offspring['other_inf'] = metrics
              print(f"metrics: {metrics}")
            print(f"time_select: {offspring['time_select']} | time_anc: {offspring['time_anc']}")
        else:
          print("execute code, ", time.time() - t, "Timeout")
          if offspring['problem'] is None:
            offspring['problem'] = 'No result'
          offspring['time'] = time.time() - t

      # with ProcessPool() as pool:
      #   t = time.time()
      #   future = pool.schedule(simple_fun, args=[die_code])
      #   try:
      #     fitness = future.result(timeout=2)
      #   except concurrent.futures.TimeoutError:
      #     print(f"Evaluation timed out after {2} seconds")
      #     print("cancel", future.cancel())
      #     print("stop", pool.stop())
      #     pool.join(timeout=1)
      #
      #   print("execute code, ", time.time() - t, fitness)
      #   offspring['objective'] = np.round(fitness, 5)
      #   future.cancel()


    except Exception as e:
      if not isinstance(e, TypeError):
        print("  Error in offspring generation", e)
      offspring = {
        'algorithm': None,
        'code':      None,
        'objective': None,
        'other_inf': None,
        'problem':   str(e)
      }
      p = None

    # Round the objective values
    return p, offspring

  def get_algorithm(self, pop, operator):
    results = []

    try:
      if self.debug:
        results = [self.get_offspring(pop, operator) for _ in range(self.pop_size)]
      else:
        num_batch = math.ceil(self.pop_size/self.n_p)
        results = []
        for batch in range(num_batch):
          n_jobs = min(self.pop_size - batch * self.n_p, self.n_p)
          batch_results = Parallel(n_jobs=n_jobs)(
              delayed(self.get_offspring)(pop, operator) for _ in range(n_jobs))
          results.extend(batch_results)

    except Exception as e:
      if self.debug:
        print(f"Error: {e}")
      print("Parallel time out .")


    out_p = []
    out_off = []

    for p, off in results:
      out_p.append(p)
      out_off.append(off)
      self.evol.add_embedding(off)
      if self.debug:
        print(f">>> check offsprings: \n {off}")
    return out_p, out_off
