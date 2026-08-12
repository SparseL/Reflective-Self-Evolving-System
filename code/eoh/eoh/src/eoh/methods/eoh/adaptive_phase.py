import math
import numpy as np

from ..selection import prob_rank, roulette_wheel, tournament


class AdaptivePhaseController:
    """Unified controller for legacy and adaptive phase/operator schedules."""

    PHASES = ("exploration", "transition", "exploitation")
    PHASE_LABELS = {
        "exploration": "Exploration",
        "transition": "Transition",
        "exploitation": "Exploitation",
    }
    PHASE_SELECTIONS = {
        "exploration": prob_rank,
        "transition": roulette_wheel,
        "exploitation": tournament,
    }
    LEGACY_ANCHORS = {
        "exploration": 0.0,
        "transition": 0.3,
        "exploitation": 0.7,
    }

    def __init__(self, operators, initial_weights, base_select, scheme="legacy", phase_start="auto"):
        self.operators = list(operators)
        self.initial_weights = [float(w) for w in initial_weights]
        self.base_select = base_select
        self.scheme = (scheme or "legacy").lower()
        self.phase_start = phase_start

        self.window = 5
        self.ema_rho = 0.75
        self.credit_beta = 0.25
        self.credit_temperature = 0.6
        self.score_temperature = 0.7
        self.phase_switch_margin = 0.08
        self.min_phase_dwell = 2
        self.current_phase = None
        self.current_phase_age = 0

        self.best_history = []
        self.state_ema = {
            "improvement": 0.0,
            "diversity": 1.0,
            "stagnation": 0.0,
            "validity": 1.0,
        }
        self.operator_credit = {op: 0.0 for op in self.operators}
        self.last_summary = {}

    def begin_generation(self, generation_idx, total_generations, population):
        state = self._observe_population(population)
        if self.scheme == "legacy":
            config = self._legacy_config(generation_idx, total_generations)
        elif self.scheme == "equal_fixed":
            config = self._equal_fixed_config()
        elif self.scheme == "state_soft":
            config = self._state_soft_config()
        elif self.scheme == "state_credit":
            config = self._state_credit_config()
        elif self.scheme == "continuous":
            config = self._continuous_config()
        else:
            raise ValueError(f"Unknown adaptive scheme: {self.scheme}")

        config["state"] = state
        self.last_summary = config
        return config

    def end_generation(self, generation_stats):
        validity = generation_stats.get("validity_ratio")
        if validity is not None:
            self.state_ema["validity"] = self._ema(self.state_ema["validity"], validity)
        if self.scheme == "state_credit":
            self._update_operator_credit(generation_stats)

    def _legacy_config(self, generation_idx, total_generations):
        progress = generation_idx / max(total_generations, 1)
        if self.phase_start != "auto":
            progress = max(progress, self.LEGACY_ANCHORS.get(self.phase_start, 0.0))

        if progress < 0.3:
            phase = "exploration"
            current_weights = [
                w * 1.5 if self._operator_role(op) == "mutation" else w * 0.8
                for w, op in zip(self.initial_weights, self.operators)
            ]
        elif progress < 0.7:
            phase = "transition"
            current_weights = list(self.initial_weights)
        else:
            phase = "exploitation"
            current_weights = [
                w * 1.5 if self._operator_role(op) == "crossover" else w * 0.8
                for w, op in zip(self.initial_weights, self.operators)
            ]

        return {
            "phase": phase,
            "phase_name": f"{self.PHASE_LABELS[phase]} ({self._selection_label(self.PHASE_SELECTIONS[phase])})",
            "select": self.PHASE_SELECTIONS[phase],
            "weights": self._clip_weights(current_weights),
            "details": {"progress": round(progress, 4)},
        }

    def _equal_fixed_config(self):
        equal_weight = float(np.mean(self.initial_weights)) if self.initial_weights else 1.0
        current_weights = [equal_weight] * len(self.operators)
        return {
            "phase": "fixed",
            "phase_name": f"Equal Fixed ({self._selection_label(self.base_select)})",
            "select": self.base_select,
            "weights": self._clip_weights(current_weights),
            "details": {"equal_weight": round(equal_weight, 4)},
        }

    def _state_soft_config(self):
        scores, mixture = self._phase_scores()
        phase = self._select_phase(scores)
        current_weights = self._blend_phase_weights(mixture)
        return {
            "phase": phase,
            "phase_name": f"{self.PHASE_LABELS[phase]} Soft",
            "select": self.PHASE_SELECTIONS[phase],
            "weights": current_weights,
            "details": {"scores": scores, "mixture": mixture},
        }

    def _state_credit_config(self):
        scores, mixture = self._phase_scores()
        phase = self._select_phase(scores)
        blended = self._blend_phase_weights(mixture)
        credit_bias = self._credit_bias()
        current_weights = []
        for weight, op in zip(blended, self.operators):
            adjusted = weight * (1.0 + 0.35 * credit_bias.get(op, 0.0))
            current_weights.append(adjusted)

        return {
            "phase": phase,
            "phase_name": f"{self.PHASE_LABELS[phase]} Credit",
            "select": self.PHASE_SELECTIONS[phase],
            "weights": self._clip_weights(current_weights),
            "details": {
                "scores": scores,
                "mixture": mixture,
                "credit": dict(self.operator_credit),
            },
        }

    def _continuous_config(self):
        diversity = self.state_ema["diversity"]
        stagnation = self.state_ema["stagnation"]
        improvement = self.state_ema["improvement"]

        explore_pressure = self._clip01(0.55 * diversity + 0.60 * stagnation - 0.25 * improvement)
        exploit_pressure = self._clip01(0.70 * (1.0 - diversity) + 0.55 * improvement - 0.45 * stagnation)
        transition_pressure = self._clip01(1.0 - abs(explore_pressure - exploit_pressure))

        if exploit_pressure - explore_pressure > 0.12:
            phase = "exploitation"
        elif explore_pressure - exploit_pressure > 0.12:
            phase = "exploration"
        else:
            phase = "transition"

        current_weights = []
        for base_weight, op in zip(self.initial_weights, self.operators):
            role = self._operator_role(op)
            if role == "mutation":
                factor = 0.75 + 0.90 * explore_pressure
            elif role == "crossover":
                factor = 0.75 + 0.95 * exploit_pressure
            else:
                factor = 0.70 + 0.60 * transition_pressure + 0.20 * diversity
            current_weights.append(base_weight * factor)

        return {
            "phase": phase,
            "phase_name": "Continuous Control",
            "select": self.PHASE_SELECTIONS[phase],
            "weights": self._clip_weights(current_weights),
            "details": {
                "explore": round(explore_pressure, 4),
                "transition": round(transition_pressure, 4),
                "exploit": round(exploit_pressure, 4),
            },
        }

    def _observe_population(self, population):
        objectives = [
            float(ind["objective"])
            for ind in population
            if ind.get("objective") is not None
        ]
        if not objectives:
            best = None
            diversity = 0.0
            improvement = 0.0
            stagnation = 1.0
        else:
            best = min(objectives)
            q10, median, q90 = np.percentile(objectives, [10, 50, 90])
            diversity = self._clip01((q90 - q10) / (abs(median) + 1e-8))
            previous_best = self.best_history[-self.window] if len(self.best_history) >= self.window else (
                self.best_history[0] if self.best_history else best
            )
            improvement = self._clip01(
                max(0.0, previous_best - best) / (abs(previous_best) + 1e-8) / 0.15
            )
            stagnation = self._stagnation_ratio(best)

        self.best_history.append(best)
        if len(self.best_history) > self.window * 4:
            self.best_history = self.best_history[-self.window * 4 :]

        self.state_ema["diversity"] = self._ema(self.state_ema["diversity"], diversity)
        self.state_ema["improvement"] = self._ema(self.state_ema["improvement"], improvement)
        self.state_ema["stagnation"] = self._ema(self.state_ema["stagnation"], stagnation)

        return {
            "best": best,
            "diversity": round(self.state_ema["diversity"], 4),
            "improvement": round(self.state_ema["improvement"], 4),
            "stagnation": round(self.state_ema["stagnation"], 4),
            "validity": round(self.state_ema["validity"], 4),
        }

    def _phase_scores(self):
        diversity = self.state_ema["diversity"]
        stagnation = self.state_ema["stagnation"]
        improvement = self.state_ema["improvement"]
        validity = self.state_ema["validity"]

        explore = 1.2 * diversity + 1.0 * stagnation - 0.8 * improvement + 0.15 * (1.0 - validity)
        transition = 1.0 - 2.0 * abs(diversity - 0.5) - 0.5 * stagnation + 0.10 * validity
        exploit = 1.3 * (1.0 - diversity) + 1.0 * improvement - 1.2 * stagnation + 0.20 * validity

        scores = {
            "exploration": round(explore, 4),
            "transition": round(transition, 4),
            "exploitation": round(exploit, 4),
        }
        mixture = self._softmax_dict(scores, self.score_temperature)
        return scores, mixture

    def _select_phase(self, scores):
        target_phase = max(scores, key=scores.get)
        if self.current_phase is None:
            self.current_phase = self.phase_start if self.phase_start in self.PHASES else target_phase
            self.current_phase_age = 1
            return self.current_phase

        current_score = scores.get(self.current_phase, -math.inf)
        target_score = scores.get(target_phase, -math.inf)
        if target_phase == self.current_phase:
            self.current_phase_age += 1
            return self.current_phase

        if (
            self.current_phase_age >= self.min_phase_dwell
            and target_score > current_score + self.phase_switch_margin
        ):
            self.current_phase = target_phase
            self.current_phase_age = 1
        else:
            self.current_phase_age += 1

        return self.current_phase

    def _blend_phase_weights(self, mixture):
        phase_profiles = {
            "exploration": self._profile_weights(1.45, 0.80, 1.10),
            "transition": self._profile_weights(1.00, 1.00, 1.00),
            "exploitation": self._profile_weights(0.70, 1.40, 0.95),
        }
        current_weights = []
        for idx, _ in enumerate(self.operators):
            weight = 0.0
            for phase in self.PHASES:
                weight += mixture[phase] * phase_profiles[phase][idx]
            current_weights.append(weight)
        return self._clip_weights(current_weights)

    def _profile_weights(self, mutation_scale, crossover_scale, custom_scale):
        profile = []
        for base_weight, op in zip(self.initial_weights, self.operators):
            role = self._operator_role(op)
            if role == "mutation":
                profile.append(base_weight * mutation_scale)
            elif role == "crossover":
                profile.append(base_weight * crossover_scale)
            else:
                profile.append(base_weight * custom_scale)
        return profile

    def _credit_bias(self):
        return self._softmax_centered(self.operator_credit, self.credit_temperature)

    def _update_operator_credit(self, generation_stats):
        rewards = generation_stats.get("operator_rewards", {})
        for op in self.operators:
            reward = rewards.get(op, 0.0)
            self.operator_credit[op] = (1.0 - self.credit_beta) * self.operator_credit[op] + self.credit_beta * reward

    def _stagnation_ratio(self, current_best):
        if current_best is None or not self.best_history:
            return 0.0
        streak = 0
        tolerance = max(1e-6, abs(current_best) * 1e-4)
        for previous in reversed(self.best_history):
            if previous is None:
                break
            if abs(previous - current_best) <= tolerance:
                streak += 1
            else:
                break
        return self._clip01(streak / max(self.window, 1))

    def _selection_label(self, selection_module):
        if selection_module is prob_rank:
            return "Prob Rank"
        if selection_module is roulette_wheel:
            return "Roulette"
        if selection_module is tournament:
            return "Tournament"
        return getattr(selection_module, "__name__", "Selection")

    def _operator_role(self, operator):
        if operator.startswith("m"):
            return "mutation"
        if operator.startswith("e"):
            return "crossover"
        return "custom"

    def _softmax_dict(self, data, temperature):
        keys = list(data.keys())
        values = np.array([float(data[key]) for key in keys], dtype=float)
        weights = self._softmax(values, temperature)
        return {key: round(float(weight), 4) for key, weight in zip(keys, weights)}

    def _softmax_centered(self, data, temperature):
        keys = list(data.keys())
        values = np.array([float(data[key]) for key in keys], dtype=float)
        weights = self._softmax(values, temperature)
        centered = weights - (1.0 / max(len(keys), 1))
        return {key: float(value) for key, value in zip(keys, centered)}

    def _softmax(self, values, temperature):
        scale = max(float(temperature), 1e-6)
        shifted = values - np.max(values)
        exp_values = np.exp(shifted / scale)
        total = np.sum(exp_values)
        if total <= 0:
            return np.full_like(values, 1.0 / max(len(values), 1))
        return exp_values / total

    def _clip_weights(self, weights):
        return [max(0.1, min(1.0, float(weight))) for weight in weights]

    def _ema(self, old, new):
        return self.ema_rho * old + (1.0 - self.ema_rho) * new

    def _clip01(self, value):
        return max(0.0, min(1.0, float(value)))
