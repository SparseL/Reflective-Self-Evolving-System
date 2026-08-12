import numpy as np
import json
import random
import time
import os

from .adaptive_phase import AdaptivePhaseController
from .eoh_interface_EC import InterfaceEC

# main class for eoh
class EOH:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc
        
        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        self.dataset_name  = paras.dataset_name

        self.dynamic_phase = paras.ec_dynamic_phase
        self.use_reflection = getattr(paras, "ec_use_reflection", True)
        self.reflection_period = getattr(paras, "ec_reflection_period", 5)
        self.reflection_log = getattr(paras, "ec_reflection_log", True)
        self.failure_stop_patience = getattr(paras, "ec_failure_stop_patience", 0)
        self.phase_start = getattr(paras, "ec_phase_start", "auto")
        self.adaptive_scheme = getattr(paras, "ec_adaptive_scheme", "legacy")

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)
    

    # run eoh 
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()
        
        # Store initial weights for phase-based adaptation
        initial_weights = list(self.operator_weights)

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model, self.use_local_llm, self.llm_local_url,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba
                                   )

        # initialization
        population = []
        all_population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data)
            filename = self.output_path + f"/results/pops/{self.dataset_name}/population_generation_0.json"

            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                if isinstance(data, dict):
                    data = [data]
                for individual in data:
                    population.append(individual)
                all_pops_path = os.path.join(
                    self.output_path,
                    "results",
                    "pops",
                    self.dataset_name,
                    "all_pops.json",
                )
                if os.path.exists(all_pops_path):
                    try:
                        with open(all_pops_path) as file:
                            old_all_population = json.load(file)
                        if isinstance(old_all_population, list):
                            all_population = old_all_population
                            print(
                                f"loaded previous all_pops history: {len(all_population)} individuals"
                            )
                    except Exception as exc:
                        print(f"warning: failed to load previous all_pops history: {exc}")
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()

                filename = self.output_path + f"/results/pops/{self.dataset_name}/population_generation_-1.json"

                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))

                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)

                population = self.manage.population_management(population, self.pop_size)
                self.add2pop(all_population, population)

                # print(len(population))
                # if len(population)<self.pop_size:
                #     for op in [self.operators[0],self.operators[2]]:
                #         _,new_ind = interface_ec.get_algorithm(population, op)
                #         self.add2pop(population, new_ind)
                #         population = self.manage.population_management(population, self.pop_size)
                #         if len(population) >= self.pop_size:
                #             break
                #         print(len(population))


                print(f"Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                # Save population to a file
                filename = self.output_path + f"/results/pops/{self.dataset_name}/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        # main loop
        n_op = len(self.operators)
        same_best_count = 0
        last_best_objective = None

        def _write_reflection_file(gen_id, events):
            if not self.reflection_log:
                return
            folder = os.path.join(self.output_path, "results", "reflections", self.dataset_name)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"generation_{gen_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2, ensure_ascii=False)

        def _snapshot_best(pop, k):
            sorted_pop = sorted(pop, key=lambda x: x["objective"] if x.get("objective") is not None else -float("inf"))
            best = sorted_pop[-k:] if len(sorted_pop) >= k else sorted_pop
            return [
                {
                    "unique_id": p.get("unique_id"),
                    "objective": p.get("objective"),
                    "operator": (p.get("other_inf") or {}).get("operator"),
                }
                for p in best
            ]

        def _normalize_objective(obj):
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                flat = obj.reshape(-1).tolist()
                return tuple(flat)
            try:
                return obj.item()
            except Exception:
                return obj

        def _objective_from_population(pop):
            values = [
                _normalize_objective(ind.get("objective"))
                for ind in pop
                if ind.get("objective") is not None
            ]
            return min(values) if values else None

        auto_aware_best = _objective_from_population(population)
        auto_aware_stale = 0

        def _build_operator_stats():
            return {
                op: {
                    "attempted": 0,
                    "triggered": 0,
                    "offspring_count": 0,
                    "valid_offspring": 0,
                    "accepted_offspring": 0,
                    "best_offspring": None,
                    "time_total": 0.0,
                }
                for op in self.operators
            }

        def _operator_rewards(operator_stats):
            rewards = {}
            for op, stats in operator_stats.items():
                offspring_count = stats["offspring_count"]
                if offspring_count <= 0:
                    rewards[op] = 0.0
                    continue

                valid_ratio = stats["valid_offspring"] / offspring_count
                accepted_ratio = stats["accepted_offspring"] / offspring_count
                fail_ratio = 1.0 - valid_ratio
                avg_time = stats["time_total"] / max(offspring_count, 1)
                time_penalty = avg_time / (avg_time + 1.0)
                best_child = stats["best_offspring"]
                best_before = stats.get("best_before")
                improvement = 0.0
                if best_child is not None and best_before is not None:
                    improvement = max(0.0, best_before - best_child) / (abs(best_before) + 1e-8)

                rewards[op] = max(
                    -1.0,
                    min(
                        1.0,
                        1.6 * improvement
                        + 0.5 * accepted_ratio
                        + 0.2 * valid_ratio
                        - 0.3 * fail_ratio
                        - 0.15 * time_penalty,
                    ),
                )
            return rewards

        def _apply_auto_aware_from_reflection(offspring, source, generation_idx):
            evol = getattr(interface_ec, "evol", None)
            if not evol or not getattr(evol, "aware_auto_enabled", False):
                return
            suggested = offspring.get("llm_aware_profile")
            if suggested:
                current = getattr(evol, "active_aware_prompt_profile", "none")
                if suggested == "none" and current != "none":
                    return
                if suggested == "ci_boundary" and current == "ci_reinsertion":
                    return
                changed = evol.set_active_aware_profile(
                    suggested,
                    reason=f"{source} reflection at generation {generation_idx} suggested {suggested}",
                )
                if changed:
                    print(f"[AwareAuto] switched to {evol.active_aware_prompt_profile} from {source} reflection.")

        def _apply_auto_aware_from_progress(generation_idx):
            nonlocal auto_aware_best, auto_aware_stale
            evol = getattr(interface_ec, "evol", None)
            if not evol or not getattr(evol, "aware_auto_enabled", False):
                return
            current_best = _objective_from_population(population)
            if current_best is None:
                return
            if auto_aware_best is None or current_best < auto_aware_best - 1e-12:
                auto_aware_best = current_best
                auto_aware_stale = 0
                return
            auto_aware_stale += 1
            progress = generation_idx / max(self.n_pop, 1)
            profile = None
            if evol.active_aware_prompt_profile == "none" and generation_idx >= 8 and auto_aware_stale >= 2:
                profile = "ci_boundary"
            elif evol.active_aware_prompt_profile == "ci_boundary" and progress >= 0.4 and auto_aware_stale >= 3:
                profile = "ci_reinsertion"
            if profile:
                changed = evol.set_active_aware_profile(
                    profile,
                    reason=f"best objective plateaued for {auto_aware_stale} generations by generation {generation_idx}",
                )
                if changed:
                    print(f"[AwareAuto] fallback switched to {profile} after objective plateau.")

        phase_controller = AdaptivePhaseController(
            self.operators,
            initial_weights,
            self.select,
            scheme=self.adaptive_scheme,
            phase_start=self.phase_start,
        )

        for pop in range(n_start, self.n_pop):
            
            # Track valid evaluations for this generation to detect API failure
            gen_valid_evals = 0
            gen_total_evals = 0
            gen_reflection_events = []
            operator_stats = _build_operator_stats()
            
            # --- Phase-based Adaptation ---
            if self.dynamic_phase:
                phase_config = phase_controller.begin_generation(pop, self.n_pop, population)
                interface_ec.select = phase_config["select"]
                self.operator_weights = phase_config["weights"]
                print(
                    f"--- Gen {pop+1}/{self.n_pop} | Scheme: {self.adaptive_scheme} | "
                    f"Phase: {phase_config['phase_name']} | Weights: {self.operator_weights} | "
                    f"State: {phase_config['state']} ---"
                )
            # -------------------------------

            # --- Thought Chain Injection ---
            if self.use_reflection:
                print(f"Gen {pop}: Daily Reflection...")
                t2_context = _snapshot_best(population, 3)
                _, offspring = interface_ec.get_offspring(population, "t2")
                gen_total_evals += 1
                gen_reflection_events.append(
                    {
                        "generation_in": pop,
                        "generation_out": pop + 1,
                        "operator": "t2",
                        "context_best": t2_context,
                        "offspring_unique_id": offspring.get("unique_id"),
                        "parents": (offspring.get("other_inf") or {}).get("parent"),
                        "objective": offspring.get("objective"),
                        "time_select": offspring.get("time_select"),
                        "time_anc": offspring.get("time_anc"),
                        "llm_raw_response": offspring.get("llm_raw_response"),
                        "llm_analysis": offspring.get("llm_analysis"),
                        "llm_thought": offspring.get("llm_thought"),
                        "llm_reflection": offspring.get("llm_reflection"),
                        "llm_aware_profile": offspring.get("llm_aware_profile"),
                    }
                )
                if offspring.get("objective") is not None:
                    gen_valid_evals += 1
                self.add2pop(population, [offspring])
                self.add2pop(all_population, [offspring])
                _apply_auto_aware_from_reflection(offspring, "t2", pop)

                if self.reflection_period and pop > 0 and pop % self.reflection_period == 0:
                    print(f"Gen {pop}: Periodic Reflection (every {self.reflection_period} gens)...")
                    t1_context = _snapshot_best(population, 5)
                    _, offspring = interface_ec.get_offspring(population, "t1")
                    gen_total_evals += 1
                    gen_reflection_events.append(
                        {
                            "generation_in": pop,
                            "generation_out": pop + 1,
                            "operator": "t1",
                            "context_best": t1_context,
                            "offspring_unique_id": offspring.get("unique_id"),
                            "parents": (offspring.get("other_inf") or {}).get("parent"),
                            "objective": offspring.get("objective"),
                            "time_select": offspring.get("time_select"),
                            "time_anc": offspring.get("time_anc"),
                            "llm_raw_response": offspring.get("llm_raw_response"),
                            "llm_analysis": offspring.get("llm_analysis"),
                            "llm_thought": offspring.get("llm_thought"),
                            "llm_reflection": offspring.get("llm_reflection"),
                            "llm_aware_profile": offspring.get("llm_aware_profile"),
                        }
                    )
                    if offspring.get("objective") is not None:
                        gen_valid_evals += 1
                    self.add2pop(population, [offspring])
                    self.add2pop(all_population, [offspring])
                    _apply_auto_aware_from_reflection(offspring, "t1", pop)

            # -------------------------------

            # 1. Parents Selection
            # parent = interface_ec.select.parent_selection(population, self.m)
            
            # 2. Evolutionary Operators (with Adaptive Weights)
            #print(f" [{na + 1} / {self.pop_size}] ", end="|")         
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                op_w = self.operator_weights[i]
                operator_stats[op]["attempted"] += 1
                previous_ids = set()
                if (np.random.rand() < op_w):
                    operator_stats[op]["triggered"] += 1
                    operator_stats[op]["best_before"] = _objective_from_population(population)
                    previous_ids = {ind.get("unique_id") for ind in population if ind.get("unique_id") is not None}
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                    gen_total_evals += len(offsprings)
                    self.add2pop(population, offsprings)  # Check duplication, and add the new offspring
                    self.add2pop(all_population, offsprings)
                    for off in offsprings:
                        operator_stats[op]["offspring_count"] += 1
                        if off['objective'] is not None:
                            gen_valid_evals += 1
                            operator_stats[op]["valid_offspring"] += 1
                            off_obj = _normalize_objective(off["objective"])
                            best_child = operator_stats[op]["best_offspring"]
                            if best_child is None or off_obj < best_child:
                                operator_stats[op]["best_offspring"] = off_obj
                        time_parts = [off.get("time_select"), off.get("time_anc")]
                        operator_stats[op]["time_total"] += sum(
                            float(val) for val in time_parts if val is not None
                        )
                        print(" Obj: ", off['objective'], end="|")
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                # populatin management
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                current_ids = {ind.get("unique_id") for ind in population if ind.get("unique_id") is not None}
                accepted_ids = current_ids - previous_ids
                if accepted_ids:
                    operator_stats[op]["accepted_offspring"] += len(accepted_ids)
                print()


            # Save population to a file
            filename = self.output_path + f"/results/pops/{self.dataset_name}/population_generation_" + str(pop + 1) + ".json"
            
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            _write_reflection_file(pop + 1, gen_reflection_events)

            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            
            # Save the best one to a file
            filename = self.output_path + f"/results/pops_best/{self.dataset_name}/population_generation_" + str(pop + 1) + ".json"
            
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()
            _apply_auto_aware_from_progress(pop + 1)

            # Check if all evaluations failed (e.g. API quota exceeded)
            if gen_total_evals > 0 and gen_valid_evals == 0:
                print(f"\n[System] Critical Warning: All {gen_total_evals} evaluations in Generation {pop} returned None.")
                print("[System] Stopping execution as it indicates potential API exhaustion or system failure.")
                break

            if self.dynamic_phase:
                phase_controller.end_generation(
                    {
                        "validity_ratio": (gen_valid_evals / gen_total_evals) if gen_total_evals > 0 else 1.0,
                        "operator_rewards": _operator_rewards(operator_stats),
                    }
                )

            if self.failure_stop_patience and self.failure_stop_patience >= 1:
                current_best = _normalize_objective(population[0].get("objective"))
                if current_best is None:
                    same_best_count = 0
                    last_best_objective = None
                elif last_best_objective is None or current_best != last_best_objective:
                    last_best_objective = current_best
                    same_best_count = 1
                else:
                    same_best_count += 1
                if same_best_count >= self.failure_stop_patience:
                    print(f"Early stop: best objective unchanged for {same_best_count} generations (>= {self.failure_stop_patience}).")
                    break

        filename = self.output_path + f"/results/pops/{self.dataset_name}/all_pops.json"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, 'w') as f:
            json.dump(all_population, f, indent=5)
