import logging
import os
import argparse

logging.basicConfig(format='[[%(levelname)s]] %(asctime)s - %(pathname)s[%(lineno)d]\n\t%(message)s')
logging.getLogger('eoh').setLevel(logging.INFO)
from eoh.src.eoh import eoh
from eoh.src.eoh.utils.getParas import Paras

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run EoH Experiment')
  parser.add_argument('--no_precompute', action='store_true', help='Disable precomputed features')
  parser.add_argument('--api_key', type=str, default=os.environ.get('EOH_API_KEY', ''), help='LLM API key (or set EOH_API_KEY)')
  parser.add_argument('--api_endpoint', type=str, default=os.environ.get('EOH_API_ENDPOINT', ''), help='LLM API endpoint (or set EOH_API_ENDPOINT)')
  parser.add_argument('--model', type=str, default='gpt-4.1-mini', help='LLM model name')
  parser.add_argument('--dataset', type=str, default='Enron_sampled', help='Dataset name')
  parser.add_argument('--new_run', action='store_true', help='Force start a new run (ignore continue settings)')
  parser.add_argument('--n_pop', type=int, default=50, help='Total number of generations')
  parser.add_argument('--pop_size', type=int, default=10, help='Population size (algorithms per generation)')
  parser.add_argument('--n_proc', type=int, default=1, help='Parallel evaluation processes')
  parser.add_argument('--timeout', type=int, default=400, help='Evaluation timeout (seconds)')
  parser.add_argument('--num_instance', type=int, default=5, help='Number of graph instances used to evaluate one algorithm')
  parser.add_argument('--failure_stop_patience', type=int, default=30, help='Early stop if best objective unchanged for N generations (0 disables)')
  parser.add_argument('--continue_id', type=int, default=None, help='Generation ID to continue from')
  parser.add_argument('--continue_path', type=str, default=None, help='Path to population file to continue from')
  parser.add_argument('--output_path', type=str, default=None, help='Output folder for results (exp_output_path)')
  parser.add_argument('--no_reflection', action='store_true', help='Disable LLM reflection (t2) and thought chain (t1)')
  parser.add_argument('--reflection_period', type=int, default=5, help='Periodic reflection interval (generations); 0 disables')
  parser.add_argument('--no_reflection_log', action='store_true', help='Disable writing per-generation/periodic reflection logs to results/reflections/')
  parser.add_argument('--phase_start', type=str, default='auto', choices=['auto', 'exploration', 'transition', 'exploitation'], help='Start dynamic phase from a specified stage')
  parser.add_argument('--adaptive_scheme', type=str, default='legacy', choices=['legacy', 'equal_fixed', 'state_soft', 'state_credit', 'continuous'], help='Adaptive phase/operator control scheme')
  parser.add_argument('--init_prompt_profile', type=str, default='standard', choices=['standard', 'enhanced_i1'], help='Initial i1 prompt profile for cold-start population generation')
  parser.add_argument('--aware_prompt_profile', type=str, default='none', choices=['none', 'random_noise_edge', 'ci_boundary', 'ci_reinsertion', 'auto'], help='Optional CN-aware guidance injected into evolution prompts. Use random_noise_edge for random noise-edge perturbed graphs.')
  args = parser.parse_args()
  if not args.api_endpoint:
      parser.error("An LLM API endpoint is required via --api_endpoint or EOH_API_ENDPOINT")
  if not args.api_key:
      parser.error("An LLM API key is required via --api_key or EOH_API_KEY")

  # Parameter initilization #
  paras = Paras()

  # Set parameters #

  dataset_name = args.dataset
  use_precompute = not args.no_precompute
  exp_output_path_base = 'enron_post' if dataset_name == 'Enron_sample' else f'{dataset_name}_post'
  exp_output_path_default = f"{exp_output_path_base}_{'with_precompute' if use_precompute else 'no_precompute'}"
  exp_output_path = args.output_path or exp_output_path_default
  if dataset_name.lower() in ("hiii14", "hi-ii-14_sampled", "hi-ii-14"):
    exp_output_path = args.output_path or "HIII14continue"
  if not os.path.exists(exp_output_path):
    os.makedirs(exp_output_path)
  
  # Default continue settings
  exp_use_continue = True
  # Default fallback values if not provided via CLI
  if dataset_name == "Enron_sample":
    exp_continue_id = 25
  elif dataset_name.lower() in ("hiii14", "hi-ii-14_sampled", "hi-ii-14"):
    exp_continue_id = 14
  else:
    exp_continue_id = 0
  previous_run_folder = exp_output_path_default
  exp_continue_path = os.path.join(os.getcwd(), previous_run_folder, "results", "pops", dataset_name, f"population_generation_{exp_continue_id}.json")

  # Override if new run requested
  if args.new_run:
      exp_use_continue = False
      exp_continue_id = 0
      exp_continue_path = ""
      print(f"Starting NEW run for dataset: {dataset_name}")
  else:
      # If CLI arguments are provided, use them
      if args.continue_id is not None:
          exp_continue_id = args.continue_id
      if args.continue_path is not None:
          exp_continue_path = args.continue_path
      else:
          exp_continue_path = os.path.join(os.getcwd(), previous_run_folder, "results", "pops", dataset_name, f"population_generation_{exp_continue_id}.json")
          
      print(f"Continuing run for dataset: {dataset_name} from gen {exp_continue_id}")
      print(f"Loading population from: {exp_continue_path}")

  paras.set_paras(method="eoh",
                  problem="cn",

                  llm_api_endpoint=args.api_endpoint,
                  llm_api_key=args.api_key,
                  llm_model=args.model,

                  ec_pop_size=args.pop_size,
                  ec_n_pop=args.n_pop,
                  exp_n_proc=args.n_proc,
                  exp_debug_mode=False,

                  # ec_pop_size=2,  # number of samples in each population
                  # ec_n_pop=3,  # number of populations
                  # exp_n_proc=1,  # multi-core parallel
                  # exp_debug_mode=True,

                  exp_output_path=exp_output_path,
                  exp_use_continue=exp_use_continue,
                  exp_continue_id=exp_continue_id,
                  exp_continue_path=exp_continue_path,
                  ec_operators=['e1','e2','m1','m2','cu'],
                  ec_operator_weights=[0.8,0.7,0.5,0.5,0.2],
                  ec_selection='roulette_wheel',
                  ##分阶段设置weight和selection方法？
                  ec_dynamic_phase=True, # 开启分阶段动态调整：初期偏重探索(Prob Rank)，中期过渡(Roulette)，后期偏重开发(Tournament)
                  ec_m=5,
                  ec_use_reflection=(not args.no_reflection) and (dataset_name.lower() not in ("hiii14", "hi-ii-14_sampled", "hi-ii-14")),
                  ec_phase_start=args.phase_start,
                  ec_adaptive_scheme=args.adaptive_scheme,
                  eva_numba_decorator=False,
                  dataset_name= dataset_name,
                  use_precompute=use_precompute,
                  cn_init_prompt_profile=args.init_prompt_profile,
                  cn_aware_prompt_profile=args.aware_prompt_profile,
                  eva_timeout=args.timeout,
                  num_instance=args.num_instance,
                  ec_failure_stop_patience=args.failure_stop_patience,
                  ec_reflection_period=args.reflection_period,
                  ec_reflection_log=not args.no_reflection_log,
                  )
  # initilization
  evolution = eoh.EVOL(paras)
  # run
  evolution.run()
