import json
import os
import sys

# Add the EoH directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
bundle_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
eoh_dir = os.path.join(bundle_root, "code", "eoh")
if eoh_dir not in sys.path:
    sys.path.append(eoh_dir)

from eoh.src.eoh.problems.optimization.cn.run import CriticalNode

def run_specific_algorithm():
    json_path = os.path.join(bundle_root, "evolve_experiment", "evolution", "real", "Youtube_sampled_post_with_precompute", "results", "pops_best", "Youtube_sampled", "population_generation_41.json")
    
    print(f"Reading algorithm from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            algorithm_code = data.get('code', '')
            
        if not algorithm_code:
            print("Error: No code found in JSON file.")
            return

        print("Algorithm code extracted.")
        
        # Initialize CriticalNode with Youtube_sample4000
        dataset_name = "Youtube_sample4000"
        print(f"Initializing CriticalNode with dataset: {dataset_name}...")
        # Note: dataset_name should match the filename prefix in the dataset folder or how CriticalNode expects it.
        # Based on file listing: Youtube_sample4000.txt exists.
        # CriticalNode typically adds .txt if missing or handles paths. 
        # Checking run_youtube8000_specific_algo.py, it used "Youtube_sample8000".
        cn = CriticalNode(dataset_name=dataset_name, use_precompute=True)
        
        print("Evaluating the algorithm...")
        result = cn.evaluate(algorithm_code)
        
        if isinstance(result, Exception):
            print(f"Evaluation failed with error: {result}")
        else:
            print(f"Evaluation Result (Objective Value): {result}")
            
        # Create results directory
        results_dir = os.path.join(bundle_root, "evolve_experiment", "transfer")
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, "youtube4000_specific_algo_result.json")
        
        output_data = {
            "dataset": dataset_name,
            "source_json": json_path,
            "algorithm": algorithm_code,
            "result": str(result) if isinstance(result, Exception) else result,
            "status": "failed" if isinstance(result, Exception) else "success"
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        print(f"Result saved to {output_file}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_specific_algorithm()
