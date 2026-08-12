
class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'eoh'
        self.problem = 'tsp_construct'
        self.selection = None
        self.management = None

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5  # number of algorithms in each population, default = 10
        self.ec_n_pop = 5 # number of populations, default = 10
        self.ec_operators = None # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
        self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]
        self.ec_dynamic_phase = False # If True, dynamically adjusts weights and selection based on generation progress.
        self.ec_use_reflection = True
        self.ec_reflection_period = 5
        self.ec_reflection_log = True
        self.ec_failure_stop_patience = 7
        self.ec_phase_start = "auto"
        self.ec_adaptive_scheme = "legacy"

        #####################
        ### LLM settings  ###
        #####################
        self.llm_use_local = False  # if use local model
        self.llm_local_url = None  # Optional local inference endpoint.
        self.llm_api_endpoint = None # endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # model type for remote LLM, e.g., deepseek-chat

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # if debug
        self.exp_output_path = "./"  # default folder for ael outputs
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seeds.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.exp_n_proc = 1
        
        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 30
        self.eva_numba_decorator = False
        self.dataset_name = 'Crime'
        self.use_precompute = True
        self.num_instance = 5
        self.cn_init_prompt_profile = 'standard'
        self.cn_aware_prompt_profile = 'none'


    def set_parallel(self):
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")
    
    def set_ec(self):    
        
        if self.management == None:
            if self.method in ['ael','eoh']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'
        
        if self.selection == None:
            self.selection = 'prob_rank'
            
        
        if self.ec_operators == None:
            if self.method == 'eoh':
                # self.ec_operators  = ['e1','e2','m1','m2']
                self.ec_operators = ['cu', 'cu']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1] * len(self.ec_operators)
            elif self.method == 'ael':
                self.ec_operators  = ['crossover','mutation']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1]
            elif self.method == 'ls':
                self.ec_operators  = ['m1']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1]
            elif self.method == 'sa':
                self.ec_operators  = ['m1']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1]
                    
        if self.method in ['ls','sa'] and self.ec_pop_size >1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1. ")
            
    def set_evaluation(self):
        # Initialize evaluation settings
        if self.problem == 'bp_online':
            self.eva_timeout = 20
            # self.eva_numba_decorator  = True
        elif self.problem == 'tsp_construct':
            self.eva_timeout = 20
        elif self.problem == 'cn_gls':
            self.eva_timeout = 600
        elif self.problem == 'cn':
            self.eva_timeout = 1800
                
    def set_paras(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key == 'ec_selection':
                self.selection = value
            elif key == 'ec_dynamic_phase':
                self.ec_dynamic_phase = value
            else:
                print(f"Warning: {key} is not a valid parameter.")
        
        self.set_parallel()
        self.set_ec()
        self.set_evaluation()
        
        if 'eva_timeout' in kwargs:
            self.eva_timeout = kwargs['eva_timeout']




            
            
            
