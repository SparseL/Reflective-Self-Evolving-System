# from machinelearning import *
# from mathematics import *
# from optimization import *
# from physics import *
class Probs():
    def __init__(self,paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "tsp_construct":
            from .optimization.tsp_greedy import run
            self.prob = run.TSPCONST()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "bp_online":
            from .optimization.bp_online import run
            self.prob = run.BPONLINE()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "cn":
            from .optimization.cn import run
            self.prob = run.CriticalNode(
                paras.num_instance,
                paras.dataset_name,
                paras.use_precompute,
                getattr(paras, "cn_init_prompt_profile", "standard"),
                getattr(paras, "cn_aware_prompt_profile", "none"),
            )  # Crime  Digg  Enron  Epinions  Facebook  Flickr  Cnutella31  HI-II-14  Youtube
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "cn_gls":
            from .optimization.cn_gls import run
            self.prob = run.CriticalNode(2)
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "hn_im":
            from .optimization.hn_im import run
            self.prob = run.HyperGraph()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "hn_cn":
            from .optimization.hn_cn import run
            self.prob = run.HyperGraph()
            print("- Prob " + paras.problem + " loaded ")
        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):

        return self.prob
