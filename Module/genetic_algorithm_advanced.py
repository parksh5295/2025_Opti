"""Advanced Genetic Algorithm with state-of-the-art techniques.

This module extends the enhanced GA with:
- Self-adaptive crossover and mutation rates
- Self-adaptive mutation distribution (σ)
- Fitness Sharing / Niching for diversity
- Surrogate model for fitness approximation
- Adaptive population size
- Island Model (distributed GA with migration)
- Multi-Objective GA (NSGA-II style)
- Advanced replacement strategies (μ+λ, Steady-state)
- Transfer Learning based initialization
- Advanced local search (Simulated Annealing, Tabu Search)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from collections import deque


@dataclass
class AdvancedGAConfig:
    """Advanced configuration for genetic algorithm with self-adaptation and niching.
    
    Attributes
    ----------
    population_size:
        Initial population size (can be adaptive).
    generations:
        Number of evolutionary iterations.
    min_crossover_prob:
        Minimum crossover probability for self-adaptation.
    max_crossover_prob:
        Maximum crossover probability for self-adaptation.
    min_mutation_prob:
        Minimum mutation probability for self-adaptation.
    max_mutation_prob:
        Maximum mutation probability for self-adaptation.
    mutation_sigma_init:
        Initial mutation distribution scale (σ).
    mutation_sigma_min:
        Minimum mutation sigma.
    mutation_sigma_max:
        Maximum mutation sigma.
    elitism:
        If True, preserve best individuals.
    elite_size:
        Number of elite individuals.
    tournament_size:
        Tournament selection size.
    min_features:
        Minimum number of active features.
    max_features:
        Maximum number of active features.
    random_state:
        Random seed.
    crossover_type:
        Type of crossover: 'single', 'two_point', 'uniform'.
    use_local_search:
        Enable local search.
    local_search_prob:
        Probability of applying local search.
    local_search_type:
        Type of local search: 'hill_climbing', 'simulated_annealing'.
    sa_initial_temp:
        Initial temperature for simulated annealing.
    sa_cooling_rate:
        Cooling rate for simulated annealing.
    use_fitness_sharing:
        Enable fitness sharing for niching.
    fitness_sharing_sigma:
        Sharing radius (distance threshold).
    fitness_sharing_alpha:
        Sharing function exponent.
    use_surrogate:
        Enable surrogate model for fitness approximation.
    surrogate_type:
        Type of surrogate: 'random_forest', 'gp' (future).
    surrogate_update_interval:
        Update surrogate every N generations.
    surrogate_sample_size:
        Number of real evaluations before using surrogate.
    adaptive_population:
        Enable adaptive population size.
    population_alpha:
        Population size reduction factor (0.0 to 1.0).
    population_beta:
        Population size increase factor (> 1.0).
    diversity_threshold:
        Minimum diversity to maintain.
    early_stopping_patience:
        Early stopping patience.
    heuristic_init_ratio:
        Ratio of population initialized heuristically.
    use_island_model:
        Enable island model (distributed GA).
    num_islands:
        Number of islands (sub-populations).
    migration_interval:
        Migration interval (generations).
    migration_rate:
        Fraction of population to migrate.
    """

    population_size: int = 80
    generations: int = 100
    min_crossover_prob: float = 0.6
    max_crossover_prob: float = 0.95
    min_mutation_prob: float = 0.01
    max_mutation_prob: float = 0.15
    mutation_sigma_init: float = 0.1
    mutation_sigma_min: float = 0.01
    mutation_sigma_max: float = 0.5
    elitism: bool = True
    elite_size: int = 3
    tournament_size: int = 4
    min_features: int = 3
    max_features: Optional[int] = None
    random_state: Optional[int] = None
    crossover_type: str = "uniform"
    use_local_search: bool = True
    local_search_prob: float = 0.3
    local_search_type: str = "simulated_annealing"  # 'hill_climbing' or 'simulated_annealing'
    sa_initial_temp: float = 1.0
    sa_cooling_rate: float = 0.95
    use_fitness_sharing: bool = True
    fitness_sharing_sigma: float = 0.3
    fitness_sharing_alpha: float = 1.0
    use_surrogate: bool = False  # Disabled by default (requires more samples)
    surrogate_type: str = "random_forest"
    surrogate_update_interval: int = 5
    surrogate_sample_size: int = 50  # Minimum real evaluations before using surrogate
    adaptive_population: bool = True
    population_alpha: float = 0.9  # Reduce by 10% when diversity is low
    population_beta: float = 1.1  # Increase by 10% when diversity is high
    diversity_threshold: float = 0.15
    early_stopping_patience: int = 20
    heuristic_init_ratio: float = 0.3
    use_island_model: bool = False  # Disabled by default (complex)
    num_islands: int = 4
    migration_interval: int = 10
    migration_rate: float = 0.1
    use_multi_objective: bool = False  # Enable NSGA-II style multi-objective optimization
    multi_objective_weights: Optional[Dict[str, float]] = None  # Weights for multiple objectives
    replacement_strategy: str = "generational"  # 'generational', 'mu_plus_lambda', 'steady_state'
    mu_plus_lambda_mu: Optional[int] = None  # μ for μ+λ selection
    steady_state_replace_worst: bool = True  # Replace worst in steady-state
    use_transfer_learning: bool = False  # Initialize from previous best solutions
    transfer_solutions: Optional[List[np.ndarray]] = None  # Previous best solutions to transfer
    use_tabu_search: bool = False  # Tabu Search for local search
    tabu_tenure: int = 5  # Tabu list size


class AdvancedGeneticFeatureSelector:
    """Advanced genetic algorithm with self-adaptation, niching, and surrogate models."""

    def __init__(
        self,
        estimator: ClassifierMixin,
        config: Optional[AdvancedGAConfig] = None,
        scoring: str = "roc_auc",
        cv: Optional[StratifiedKFold] = None,
        verbose: bool = False,
        fitness_function: Optional[Callable[[np.ndarray, np.ndarray, ArrayLike], float]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> None:
        self.estimator = estimator
        self.config = config or AdvancedGAConfig()
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.fitness_function = fitness_function
        self.feature_importance = feature_importance

        self.feature_names_: Optional[Sequence[str]] = None
        self.best_individual_: Optional[np.ndarray] = None
        self.best_score_: float = -np.inf
        self.history_: List[float] = []
        self.diversity_history_: List[float] = []
        self.population_size_history_: List[int] = []

        self._rng = np.random.default_rng(self.config.random_state)
        self._fitness_cache: Dict[Tuple, float] = {}
        self._no_improvement_count = 0
        self._surrogate_model: Optional[RandomForestRegressor] = None
        self._surrogate_training_data: List[Tuple[np.ndarray, float]] = []
        self._current_population_size: int = self.config.population_size
        self._islands: Optional[List[Tuple[np.ndarray, List[Dict[str, float]], np.ndarray]]] = None
        self._tabu_list: deque = deque(maxlen=self.config.tabu_tenure if self.config.use_tabu_search else 0)
        self._pareto_front: Optional[List[Tuple[np.ndarray, Dict[str, float]]]] = None  # For multi-objective

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AdvancedGeneticFeatureSelector":
        X_arr, feature_names = self._coerce_input(X)
        n_features = X_arr.shape[1]

        self.feature_names_ = feature_names
        max_features = self.config.max_features or n_features
        min_features = min(self.config.min_features, n_features)

        # Initialize population with self-adaptive parameters
        if self.config.use_island_model:
            # Initialize multiple islands
            self._islands = []
            island_size = self._current_population_size // self.config.num_islands
            for island_idx in range(self.config.num_islands):
                island_pop, island_params = self._initialize_island_population(
                    island_size, n_features, min_features, max_features, X_arr, y, island_idx
                )
                island_fitness = self._evaluate_population(island_pop, X_arr, y, island_params)
                self._islands.append((island_pop, island_params, island_fitness))
            # Use first island as main population for tracking
            population, adaptive_params, fitness = self._islands[0]
        else:
            population, adaptive_params = self._initialize_population_with_params(
                n_features, min_features, max_features, X_arr, y
            )
            fitness = self._evaluate_population(population, X_arr, y, adaptive_params)

        for generation in range(self.config.generations):
            if self.config.use_island_model:
                # Evolve each island separately
                for island_idx in range(len(self._islands)):
                    island_pop, island_params, island_fitness = self._islands[island_idx]
                    
                    # Apply fitness sharing if enabled
                    if self.config.use_fitness_sharing:
                        island_fitness = self._apply_fitness_sharing(island_pop, island_fitness)
                    
                    # Evolve island
                    island_pop, island_params, island_fitness = self._evolve_generation_adaptive(
                        island_pop, island_params, island_fitness, X_arr, y, min_features, max_features
                    )
                    
                    # Local search
                    if self.config.use_local_search and self._rng.random() < self.config.local_search_prob:
                        best_idx = int(np.argmax(island_fitness))
                        improved = self._local_search_advanced(
                            island_pop[best_idx],
                            island_params[best_idx],
                            island_fitness[best_idx],
                            X_arr,
                            y,
                            min_features,
                            max_features,
                            generation,
                        )
                        if improved is not None:
                            island_pop[best_idx] = improved[0]
                            island_params[best_idx] = improved[1]
                            island_fitness[best_idx] = improved[2]
                    
                    self._islands[island_idx] = (island_pop, island_params, island_fitness)
                
                # Migration between islands
                if generation > 0 and generation % self.config.migration_interval == 0:
                    self._migrate_between_islands()
                
                # Update main population from best island
                best_island_idx = max(
                    range(len(self._islands)),
                    key=lambda i: np.max(self._islands[i][2])
                )
                population, adaptive_params, fitness = self._islands[best_island_idx]
            else:
                # Single population evolution
                # Adaptive population size
                if self.config.adaptive_population:
                    diversity = self._compute_diversity(population)
                    if diversity < self.config.diversity_threshold:
                        new_size = max(
                            int(self._current_population_size * self.config.population_alpha),
                            self.config.elite_size + 2,
                        )
                    else:
                        new_size = min(
                            int(self._current_population_size * self.config.population_beta),
                            self.config.population_size * 2,
                        )
                    if new_size != self._current_population_size:
                        population, adaptive_params = self._resize_population(
                            population, adaptive_params, fitness, new_size, min_features, max_features
                        )
                        self._current_population_size = new_size
                        fitness = self._evaluate_population(population, X_arr, y, adaptive_params)

                # Apply fitness sharing if enabled
                if self.config.use_fitness_sharing:
                    fitness = self._apply_fitness_sharing(population, fitness)

                # Evolve generation with self-adaptive parameters
                population, adaptive_params, fitness = self._evolve_generation_adaptive(
                    population, adaptive_params, fitness, X_arr, y, min_features, max_features
                )

            # Local search with SA, hill climbing, or Tabu Search
            if not self.config.use_island_model:
                if self.config.use_local_search and self._rng.random() < self.config.local_search_prob:
                    best_idx = int(np.argmax(fitness))
                    improved = self._local_search_advanced(
                        population[best_idx],
                        adaptive_params[best_idx],
                        fitness[best_idx],
                        X_arr,
                        y,
                        min_features,
                        max_features,
                        generation,
                    )
                    if improved is not None:
                        population[best_idx] = improved[0]
                        adaptive_params[best_idx] = improved[1]
                        fitness[best_idx] = improved[2]

            # Update surrogate model
            if self.config.use_surrogate and generation % self.config.surrogate_update_interval == 0:
                self._update_surrogate_model(population, fitness)

            # Track history
            best_gen_score = float(np.max(fitness))
            self.history_.append(best_gen_score)
            diversity = self._compute_diversity(population)
            self.diversity_history_.append(diversity)
            self.population_size_history_.append(self._current_population_size)

            # Update best
            if best_gen_score > self.best_score_:
                idx = int(np.argmax(fitness))
                self.best_score_ = float(fitness[idx])
                self.best_individual_ = population[idx].copy()
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1

            if self.verbose and self.best_individual_ is not None:
                avg_crossover = np.mean([p["crossover_prob"] for p in adaptive_params])
                avg_mutation = np.mean([p["mutation_prob"] for p in adaptive_params])
                avg_sigma = np.mean([p["mutation_sigma"] for p in adaptive_params])
                print(
                    f"[Advanced GA] Gen {generation + 1:03d} | "
                    f"Best: {self.best_score_:.4f} | "
                    f"Active: {int(self.best_individual_.sum())} | "
                    f"Diversity: {diversity:.3f} | "
                    f"Pop: {self._current_population_size} | "
                    f"P_c: {avg_crossover:.3f} | "
                    f"P_m: {avg_mutation:.3f} | "
                    f"σ: {avg_sigma:.3f}"
                )

            # Early stopping
            if (
                self.config.early_stopping_patience > 0
                and self._no_improvement_count >= self.config.early_stopping_patience
            ):
                if self.verbose:
                    print(f"[Advanced GA] Early stopping at generation {generation + 1}")
                break

        return self

    def _initialize_population_with_params(
        self,
        n_features: int,
        min_features: int,
        max_features: int,
        X: np.ndarray,
        y: ArrayLike,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Initialize population with self-adaptive parameters encoded in each individual."""
        population = np.zeros((self._current_population_size, n_features), dtype=int)
        adaptive_params = []

        # Transfer learning: initialize from previous best solutions
        transfer_start_idx = 0
        if self.config.use_transfer_learning and self.config.transfer_solutions:
            n_transfer = min(len(self.config.transfer_solutions), self._current_population_size // 4)
            for idx in range(n_transfer):
                if idx < len(self.config.transfer_solutions):
                    transfer_sol = self.config.transfer_solutions[idx]
                    if len(transfer_sol) == n_features:
                        population[idx] = transfer_sol.copy()
                        transfer_start_idx = idx + 1

        n_heuristic = int((self._current_population_size - transfer_start_idx) * self.config.heuristic_init_ratio)

        # Heuristic initialization
        if self.feature_importance and n_heuristic > 0:
            sorted_features = sorted(
                self.feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            top_features = [f[0] for f in sorted_features[:max_features]]

            for idx in range(transfer_start_idx, transfer_start_idx + n_heuristic):
                size = self._rng.integers(min_features, min(max_features, len(top_features)) + 1)
                if size <= len(top_features):
                    selected = self._rng.choice(len(top_features), size=size, replace=False)
                    feature_indices = [self.feature_names_.index(top_features[i]) for i in selected]
                    population[idx, feature_indices] = 1

        # Random initialization for remaining
        for idx in range(transfer_start_idx + n_heuristic, self._current_population_size):
            size = self._rng.integers(min_features, max_features + 1)
            active_idx = self._rng.choice(n_features, size=size, replace=False)
            population[idx, active_idx] = 1

        # Initialize adaptive parameters for each individual
        for idx in range(self._current_population_size):
            adaptive_params.append({
                "crossover_prob": self._rng.uniform(
                    self.config.min_crossover_prob, self.config.max_crossover_prob
                ),
                "mutation_prob": self._rng.uniform(
                    self.config.min_mutation_prob, self.config.max_mutation_prob
                ),
                "mutation_sigma": self._rng.uniform(
                    self.config.mutation_sigma_min, self.config.mutation_sigma_max
                ),
            })

        return population, adaptive_params

    def _initialize_island_population(
        self,
        island_size: int,
        n_features: int,
        min_features: int,
        max_features: int,
        X: np.ndarray,
        y: ArrayLike,
        island_idx: int,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Initialize population for a specific island (with different random seed per island)."""
        island_rng = np.random.default_rng(
            (self.config.random_state or 42) + island_idx * 1000
        )
        population = np.zeros((island_size, n_features), dtype=int)
        adaptive_params = []

        for idx in range(island_size):
            size = island_rng.integers(min_features, max_features + 1)
            active_idx = island_rng.choice(n_features, size=size, replace=False)
            population[idx, active_idx] = 1

            adaptive_params.append({
                "crossover_prob": island_rng.uniform(
                    self.config.min_crossover_prob, self.config.max_crossover_prob
                ),
                "mutation_prob": island_rng.uniform(
                    self.config.min_mutation_prob, self.config.max_mutation_prob
                ),
                "mutation_sigma": island_rng.uniform(
                    self.config.mutation_sigma_min, self.config.mutation_sigma_max
                ),
            })

        return population, adaptive_params

    def _evolve_generation_adaptive(
        self,
        population: np.ndarray,
        adaptive_params: List[Dict[str, float]],
        fitness: np.ndarray,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
    ) -> Tuple[np.ndarray, List[Dict[str, float]], np.ndarray]:
        """Evolve generation with self-adaptive crossover and mutation rates."""
        pop_size = population.shape[0]
        
        if self.config.replacement_strategy == "steady_state":
            return self._evolve_steady_state(
                population, adaptive_params, fitness, X, y, min_features, max_features
            )
        elif self.config.replacement_strategy == "mu_plus_lambda":
            return self._evolve_mu_plus_lambda(
                population, adaptive_params, fitness, X, y, min_features, max_features
            )
        else:  # generational
            return self._evolve_generational(
                population, adaptive_params, fitness, X, y, min_features, max_features
            )

    def _evolve_generational(
        self,
        population: np.ndarray,
        adaptive_params: List[Dict[str, float]],
        fitness: np.ndarray,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
    ) -> Tuple[np.ndarray, List[Dict[str, float]], np.ndarray]:
        """Standard generational replacement."""
        next_population = []
        next_params = []

        # Enhanced elitism with diversity
        if self.config.elitism:
            elite_indices = self._select_elite(population, fitness, self.config.elite_size)
            for idx in elite_indices:
                next_population.append(population[idx].copy())
                next_params.append(adaptive_params[idx].copy())

        # Generate offspring
        while len(next_population) < population.shape[0]:
            parent1_idx = self._tournament_selection_idx(population, fitness)
            parent2_idx = self._tournament_selection_idx(population, fitness)

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            params1 = adaptive_params[parent1_idx]
            params2 = adaptive_params[parent2_idx]

            # Use parent's crossover probability
            crossover_prob = (params1["crossover_prob"] + params2["crossover_prob"]) / 2.0

            if self._rng.random() < crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
                child_params1 = self._mutate_adaptive_params(params1, params2)
                child_params2 = self._mutate_adaptive_params(params2, params1)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
                child_params1 = params1.copy()
                child_params2 = params2.copy()

            child1 = self._mutate_adaptive(child1, min_features, max_features, child_params1)
            child2 = self._mutate_adaptive(child2, min_features, max_features, child_params2)

            next_population.extend([child1, child2])
            next_params.extend([child_params1, child_params2])

        population = np.array(next_population[: population.shape[0]])
        params = next_params[: population.shape[0]]
        fitness = self._evaluate_population(population, X, y, params)

        return population, params, fitness

    def _evolve_mu_plus_lambda(
        self,
        population: np.ndarray,
        adaptive_params: List[Dict[str, float]],
        fitness: np.ndarray,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
    ) -> Tuple[np.ndarray, List[Dict[str, float]], np.ndarray]:
        """μ+λ selection: keep parents and select best from (parents + offspring)."""
        mu = self.config.mu_plus_lambda_mu or population.shape[0]
        lambda_size = population.shape[0] - mu

        # Generate λ offspring
        offspring = []
        offspring_params = []
        for _ in range(lambda_size):
            parent1_idx = self._tournament_selection_idx(population, fitness)
            parent2_idx = self._tournament_selection_idx(population, fitness)

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            params1 = adaptive_params[parent1_idx]
            params2 = adaptive_params[parent2_idx]

            crossover_prob = (params1["crossover_prob"] + params2["crossover_prob"]) / 2.0

            if self._rng.random() < crossover_prob:
                child, _ = self._crossover(parent1, parent2)
                child_params = self._mutate_adaptive_params(params1, params2)
            else:
                child = parent1.copy()
                child_params = params1.copy()

            child = self._mutate_adaptive(child, min_features, max_features, child_params)
            offspring.append(child)
            offspring_params.append(child_params)

        # Evaluate offspring
        offspring_fitness = self._evaluate_population(
            np.array(offspring), X, y, offspring_params
        )

        # Combine parents and offspring
        combined_pop = np.vstack([population, np.array(offspring)])
        combined_params = adaptive_params + offspring_params
        combined_fitness = np.concatenate([fitness, offspring_fitness])

        # Select best μ individuals
        sorted_indices = np.argsort(combined_fitness)[::-1]
        selected_indices = sorted_indices[:mu]

        new_population = combined_pop[selected_indices]
        new_params = [combined_params[i] for i in selected_indices]
        new_fitness = combined_fitness[selected_indices]

        return new_population, new_params, new_fitness

    def _evolve_steady_state(
        self,
        population: np.ndarray,
        adaptive_params: List[Dict[str, float]],
        fitness: np.ndarray,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
    ) -> Tuple[np.ndarray, List[Dict[str, float]], np.ndarray]:
        """Steady-state GA: replace worst individuals incrementally."""
        # Generate a few offspring
        n_offspring = max(1, population.shape[0] // 10)  # 10% of population
        offspring = []
        offspring_params = []

        for _ in range(n_offspring):
            parent1_idx = self._tournament_selection_idx(population, fitness)
            parent2_idx = self._tournament_selection_idx(population, fitness)

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            params1 = adaptive_params[parent1_idx]
            params2 = adaptive_params[parent2_idx]

            crossover_prob = (params1["crossover_prob"] + params2["crossover_prob"]) / 2.0

            if self._rng.random() < crossover_prob:
                child, _ = self._crossover(parent1, parent2)
                child_params = self._mutate_adaptive_params(params1, params2)
            else:
                child = parent1.copy()
                child_params = params1.copy()

            child = self._mutate_adaptive(child, min_features, max_features, child_params)
            offspring.append(child)
            offspring_params.append(child_params)

        # Evaluate offspring
        offspring_fitness = self._evaluate_population(
            np.array(offspring), X, y, offspring_params
        )

        # Replace worst individuals
        new_population = population.copy()
        new_params = adaptive_params.copy()
        new_fitness = fitness.copy()

        for i in range(len(offspring)):
            if self.config.steady_state_replace_worst:
                worst_idx = int(np.argmin(new_fitness))
            else:
                # Random replacement
                worst_idx = self._rng.integers(0, len(new_population))

            if offspring_fitness[i] > new_fitness[worst_idx]:
                new_population[worst_idx] = offspring[i]
                new_params[worst_idx] = offspring_params[i]
                new_fitness[worst_idx] = offspring_fitness[i]

        return new_population, new_params, new_fitness

    def _mutate_adaptive_params(
        self, params1: Dict[str, float], params2: Dict[str, float]
    ) -> Dict[str, float]:
        """Create child adaptive parameters by blending and mutating parent parameters."""
        child = {
            "crossover_prob": (params1["crossover_prob"] + params2["crossover_prob"]) / 2.0,
            "mutation_prob": (params1["mutation_prob"] + params2["mutation_prob"]) / 2.0,
            "mutation_sigma": (params1["mutation_sigma"] + params2["mutation_sigma"]) / 2.0,
        }

        # Mutate parameters (self-adaptation)
        tau = 1.0 / np.sqrt(2.0 * len(child))
        child["crossover_prob"] += self._rng.normal(0, tau * 0.1)
        child["crossover_prob"] = np.clip(
            child["crossover_prob"],
            self.config.min_crossover_prob,
            self.config.max_crossover_prob,
        )

        child["mutation_prob"] += self._rng.normal(0, tau * 0.05)
        child["mutation_prob"] = np.clip(
            child["mutation_prob"],
            self.config.min_mutation_prob,
            self.config.max_mutation_prob,
        )

        # Mutate sigma (mutation distribution parameter)
        child["mutation_sigma"] *= np.exp(self._rng.normal(0, tau))
        child["mutation_sigma"] = np.clip(
            child["mutation_sigma"],
            self.config.mutation_sigma_min,
            self.config.mutation_sigma_max,
        )

        return child

    def _mutate_adaptive(
        self,
        individual: np.ndarray,
        min_features: int,
        max_features: int,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Mutate individual using self-adaptive mutation rate and sigma."""
        mutant = individual.copy()
        mutation_prob = params["mutation_prob"]
        mutation_sigma = params["mutation_sigma"]

        # Use sigma to control mutation intensity (for continuous-like behavior)
        # For binary features, we still flip bits but sigma affects the probability
        for idx in range(mutant.size):
            # Adjust mutation probability based on sigma
            adjusted_prob = mutation_prob * (1.0 + mutation_sigma)
            if self._rng.random() < adjusted_prob:
                mutant[idx] = 1 - mutant[idx]

        # Ensure constraints
        active = mutant.sum()
        if active < min_features:
            zeros = np.where(mutant == 0)[0]
            if zeros.size > 0:
                flip_count = min_features - active
                chosen = self._rng.choice(zeros, size=min(flip_count, zeros.size), replace=False)
                mutant[chosen] = 1
        elif active > max_features:
            ones = np.where(mutant == 1)[0]
            flip_count = active - max_features
            chosen = self._rng.choice(ones, size=min(flip_count, ones.size), replace=False)
            mutant[chosen] = 0

        if mutant.sum() == 0:
            idx = self._rng.integers(0, mutant.size)
            mutant[idx] = 1

        return mutant

    def _apply_fitness_sharing(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> np.ndarray:
        """Apply fitness sharing to promote diversity (niching)."""
        shared_fitness = fitness.copy()
        n = population.shape[0]
        sigma_share = self.config.fitness_sharing_sigma
        alpha = self.config.fitness_sharing_alpha

        for i in range(n):
            sharing_sum = 0.0
            for j in range(n):
                # Hamming distance (normalized)
                distance = np.sum(population[i] != population[j]) / population.shape[1]
                if distance < sigma_share:
                    sharing = 1.0 - (distance / sigma_share) ** alpha
                    sharing_sum += sharing

            if sharing_sum > 0:
                shared_fitness[i] = fitness[i] / sharing_sum

        return shared_fitness

    def _local_search_advanced(
        self,
        individual: np.ndarray,
        params: Dict[str, float],
        current_fitness: float,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
        generation: int,
    ) -> Optional[Tuple[np.ndarray, Dict[str, float], float]]:
        """Advanced local search with Simulated Annealing, Hill Climbing, or Tabu Search."""
        if self.config.use_tabu_search:
            return self._tabu_search(
                individual, params, current_fitness, X, y, min_features, max_features, generation
            )
        elif self.config.local_search_type == "simulated_annealing":
            return self._simulated_annealing_search(
                individual, params, current_fitness, X, y, min_features, max_features, generation
            )
        else:
            return self._hill_climbing_search(
                individual, params, current_fitness, X, y, min_features, max_features
            )

    def _simulated_annealing_search(
        self,
        individual: np.ndarray,
        params: Dict[str, float],
        current_fitness: float,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
        generation: int,
    ) -> Optional[Tuple[np.ndarray, Dict[str, float], float]]:
        """Simulated Annealing local search."""
        best_individual = individual.copy()
        best_params = params.copy()
        best_fitness = current_fitness

        # Temperature decreases with generation
        temperature = self.config.sa_initial_temp * (
            self.config.sa_cooling_rate ** generation
        )

        # Try a few neighbors
        for _ in range(5):  # Try 5 neighbors
            candidate = best_individual.copy()
            # Randomly flip a feature
            flip_idx = self._rng.integers(0, candidate.size)
            candidate[flip_idx] = 1 - candidate[flip_idx]

            # Ensure constraints
            active = candidate.sum()
            if active < min_features or active > max_features:
                continue

            # Evaluate candidate
            candidate_fitness = self._evaluate_individual(candidate, X, y)

            # Accept if better, or with probability if worse (SA)
            if candidate_fitness > best_fitness:
                best_individual = candidate
                best_fitness = candidate_fitness
            elif temperature > 0:
                delta = candidate_fitness - best_fitness
                accept_prob = np.exp(delta / (temperature + 1e-10))
                if self._rng.random() < accept_prob:
                    best_individual = candidate
                    best_fitness = candidate_fitness

        if best_fitness > current_fitness:
            return best_individual, best_params, best_fitness
        return None

    def _hill_climbing_search(
        self,
        individual: np.ndarray,
        params: Dict[str, float],
        current_fitness: float,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
    ) -> Optional[Tuple[np.ndarray, Dict[str, float], float]]:
        """Hill climbing local search (greedy)."""
        best_individual = individual.copy()
        best_params = params.copy()
        best_fitness = current_fitness

        for idx in range(individual.size):
            candidate = best_individual.copy()
            candidate[idx] = 1 - candidate[idx]

            active = candidate.sum()
            if active < min_features or active > max_features:
                continue

            candidate_fitness = self._evaluate_individual(candidate, X, y)
            if candidate_fitness > best_fitness:
                best_individual = candidate
                best_fitness = candidate_fitness
                break  # Greedy: take first improvement

        if best_fitness > current_fitness:
            return best_individual, best_params, best_fitness
        return None

    def _tabu_search(
        self,
        individual: np.ndarray,
        params: Dict[str, float],
        current_fitness: float,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
        generation: int,
    ) -> Optional[Tuple[np.ndarray, Dict[str, float], float]]:
        """Tabu Search local search."""
        best_individual = individual.copy()
        best_params = params.copy()
        best_fitness = current_fitness
        current_individual = individual.copy()

        # Try multiple neighbors
        for _ in range(10):  # Try 10 neighbors
            best_neighbor = None
            best_neighbor_fitness = -np.inf
            best_move = None

            # Evaluate all neighbors
            for idx in range(individual.size):
                candidate = current_individual.copy()
                candidate[idx] = 1 - candidate[idx]

                active = candidate.sum()
                if active < min_features or active > max_features:
                    continue

                # Check if move is tabu
                move_key = tuple(candidate.tolist())
                if move_key in self._tabu_list:
                    continue

                candidate_fitness = self._evaluate_individual(candidate, X, y)
                if candidate_fitness > best_neighbor_fitness:
                    best_neighbor = candidate
                    best_neighbor_fitness = candidate_fitness
                    best_move = move_key

            if best_neighbor is None:
                break

            # Accept best neighbor (even if worse than current - aspiration criterion)
            if best_neighbor_fitness > best_fitness:
                best_individual = best_neighbor.copy()
                best_fitness = best_neighbor_fitness

            # Move to neighbor
            current_individual = best_neighbor.copy()
            # Add to tabu list
            if best_move is not None:
                self._tabu_list.append(best_move)

        if best_fitness > current_fitness:
            return best_individual, best_params, best_fitness
        return None

    def _migrate_between_islands(self) -> None:
        """Migrate individuals between islands."""
        if self._islands is None or len(self._islands) < 2:
            return

        n_islands = len(self._islands)
        migration_count = max(1, int(self.config.migration_rate * self._islands[0][0].shape[0]))

        for source_idx in range(n_islands):
            source_pop, source_params, source_fitness = self._islands[source_idx]
            target_idx = (source_idx + 1) % n_islands  # Migrate to next island

            # Select best individuals to migrate
            sorted_indices = np.argsort(source_fitness)[::-1]
            migrants_indices = sorted_indices[:migration_count]

            migrants_pop = source_pop[migrants_indices]
            migrants_params = [source_params[i] for i in migrants_indices]
            migrants_fitness = source_fitness[migrants_indices]

            # Replace worst individuals in target island
            target_pop, target_params, target_fitness = self._islands[target_idx]
            sorted_target = np.argsort(target_fitness)
            replace_indices = sorted_target[:migration_count]

            for i, replace_idx in enumerate(replace_indices):
                if i < len(migrants_pop):
                    target_pop[replace_idx] = migrants_pop[i]
                    target_params[replace_idx] = migrants_params[i]
                    target_fitness[replace_idx] = migrants_fitness[i]

            self._islands[target_idx] = (target_pop, target_params, target_fitness)

    def _compute_multi_objective_fitness(
        self,
        chromosome: np.ndarray,
        X: np.ndarray,
        y: ArrayLike,
    ) -> Dict[str, float]:
        """Compute multiple objectives for multi-objective GA."""
        if not self.config.use_multi_objective:
            return {"fitness": self._evaluate_individual(chromosome, X, y)}

        objectives = {}
        
        # Objective 1: Model performance (fitness)
        objectives["performance"] = self._evaluate_individual(chromosome, X, y)
        
        # Objective 2: Feature count (minimize)
        objectives["feature_count"] = -float(chromosome.sum())  # Negative for minimization
        
        # Objective 3: Redundancy (if penalty matrix available)
        # This would need to be passed in, but for now we'll use a simple metric
        if hasattr(self, '_penalty_matrix') and self._penalty_matrix is not None:
            # Simplified: use feature count as proxy for redundancy
            objectives["redundancy"] = float(chromosome.sum()) / X.shape[1]
        else:
            objectives["redundancy"] = 0.0

        return objectives

    def _non_dominated_sort(
        self,
        population: np.ndarray,
        objectives: List[Dict[str, float]],
    ) -> List[List[int]]:
        """NSGA-II style non-dominated sorting."""
        n = len(population)
        fronts = []
        dominated_by = [[] for _ in range(n)]
        domination_count = [0] * n

        # Build domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                obj_i = objectives[i]
                obj_j = objectives[j]

                # Check if i dominates j (all objectives better or equal, at least one strictly better)
                i_dominates = all(obj_i[k] >= obj_j[k] for k in obj_i.keys())
                i_strictly_better = any(obj_i[k] > obj_j[k] for k in obj_i.keys())

                j_dominates = all(obj_j[k] >= obj_i[k] for k in obj_i.keys())
                j_strictly_better = any(obj_j[k] > obj_i[k] for k in obj_j.keys())

                if i_dominates and i_strictly_better:
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif j_dominates and j_strictly_better:
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        # Build fronts
        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []

            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            current_front = next_front

        return fronts

    def _crowding_distance(
        self,
        front: List[int],
        objectives: List[Dict[str, float]],
    ) -> np.ndarray:
        """Compute crowding distance for NSGA-II."""
        if len(front) <= 2:
            return np.full(len(front), np.inf)

        distances = np.zeros(len(front))
        obj_keys = list(objectives[0].keys())

        for key in obj_keys:
            values = np.array([objectives[i][key] for i in front])
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]

            # Boundary points get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Normalize range
            value_range = sorted_values[-1] - sorted_values[0]
            if value_range > 0:
                for idx in range(1, len(front) - 1):
                    distances[sorted_indices[idx]] += (
                        sorted_values[idx + 1] - sorted_values[idx - 1]
                    ) / value_range

        return distances

    def _update_surrogate_model(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> None:
        """Update surrogate model for fitness approximation."""
        if len(self._surrogate_training_data) < self.config.surrogate_sample_size:
            # Not enough data yet, collect more
            for i in range(population.shape[0]):
                self._surrogate_training_data.append((population[i].copy(), float(fitness[i])))
            return

        # Update training data (keep recent samples)
        max_samples = 200
        for i in range(population.shape[0]):
            self._surrogate_training_data.append((population[i].copy(), float(fitness[i])))
        if len(self._surrogate_training_data) > max_samples:
            # Keep most recent samples
            self._surrogate_training_data = self._surrogate_training_data[-max_samples:]

        # Train surrogate model
        if self.config.surrogate_type == "random_forest":
            X_train = np.array([data[0] for data in self._surrogate_training_data])
            y_train = np.array([data[1] for data in self._surrogate_training_data])

            self._surrogate_model = RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=self.config.random_state
            )
            self._surrogate_model.fit(X_train, y_train)

    def _evaluate_population(
        self,
        population: np.ndarray,
        X: np.ndarray,
        y: ArrayLike,
        adaptive_params: List[Dict[str, float]],
    ) -> np.ndarray:
        """Evaluate population using surrogate model if available, otherwise real fitness."""
        if self.config.use_multi_objective:
            # Multi-objective evaluation
            objectives_list = []
            for idx, chromosome in enumerate(population):
                if chromosome.sum() == 0:
                    objectives_list.append({"performance": -np.inf, "feature_count": 0.0, "redundancy": 0.0})
                else:
                    objectives_list.append(self._compute_multi_objective_fitness(chromosome, X, y))
            
            # Convert to scalar fitness using weighted sum (for selection)
            if self.config.multi_objective_weights:
                weights = self.config.multi_objective_weights
            else:
                weights = {"performance": 0.7, "feature_count": 0.2, "redundancy": 0.1}
            
            scores = np.array([
                sum(obj.get(k, 0.0) * weights.get(k, 0.0) for k in weights.keys())
                for obj in objectives_list
            ])
            
            # Store Pareto front
            fronts = self._non_dominated_sort(population, objectives_list)
            if fronts:
                self._pareto_front = [
                    (population[i].copy(), objectives_list[i])
                    for i in fronts[0]  # First front (non-dominated)
                ]
            
            return scores

        # Single-objective evaluation
        scores = np.zeros(population.shape[0], dtype=float)

        # Use surrogate if available and enough samples collected
        use_surrogate = (
            self.config.use_surrogate
            and self._surrogate_model is not None
            and len(self._surrogate_training_data) >= self.config.surrogate_sample_size
        )

        for idx, chromosome in enumerate(population):
            if chromosome.sum() == 0:
                scores[idx] = -np.inf
                continue

            # Check cache
            chrom_tuple = tuple(chromosome.tolist())
            if chrom_tuple in self._fitness_cache:
                scores[idx] = self._fitness_cache[chrom_tuple]
                continue

            # Use surrogate or real evaluation
            if use_surrogate and idx % 3 != 0:  # Use surrogate for 2/3 of population
                try:
                    pred_fitness = self._surrogate_model.predict(chromosome.reshape(1, -1))[0]
                    scores[idx] = float(pred_fitness)
                    continue
                except Exception:
                    # Fallback to real evaluation if surrogate fails
                    pass

            # Real fitness evaluation
            score = self._evaluate_individual(chromosome, X, y)
            scores[idx] = score

            # Cache
            if len(self._fitness_cache) < 1000:
                self._fitness_cache[chrom_tuple] = score

        return scores

    def _evaluate_individual(
        self,
        chromosome: np.ndarray,
        X: np.ndarray,
        y: ArrayLike,
        cv: Optional[StratifiedKFold] = None,
    ) -> float:
        """Evaluate a single individual."""
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)

        if self.fitness_function is not None:
            try:
                return float(self.fitness_function(chromosome, X, y))
            except (ValueError, RuntimeError):
                return -np.inf

        selected_features = X[:, chromosome.astype(bool)]
        estimator = clone(self.estimator)

        try:
            cv_scores = cross_val_score(
                estimator,
                selected_features,
                y,
                scoring=self.scoring,
                cv=cv,
                n_jobs=None,
            )
            return float(np.mean(cv_scores))
        except (ValueError, RuntimeError):
            return -np.inf

    def _tournament_selection_idx(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> int:
        """Tournament selection returning index."""
        participants = self._rng.choice(
            population.shape[0],
            size=self.config.tournament_size,
            replace=False,
        )
        best_idx = participants[np.argmax(fitness[participants])]
        return int(best_idx)

    def _select_elite(
        self, population: np.ndarray, fitness: np.ndarray, elite_size: int
    ) -> List[int]:
        """Select elite individuals considering diversity."""
        if elite_size == 1:
            return [int(np.argmax(fitness))]

        sorted_indices = np.argsort(fitness)[::-1]
        elite = [int(sorted_indices[0])]

        for idx in sorted_indices[1:]:
            if len(elite) >= elite_size:
                break
            candidate = int(idx)
            min_distance = min(
                np.sum(population[candidate] != population[e]) for e in elite
            )
            if min_distance > population.shape[1] * 0.2:
                elite.append(candidate)

        for idx in sorted_indices:
            if len(elite) >= elite_size:
                break
            if int(idx) not in elite:
                elite.append(int(idx))

        return elite[:elite_size]

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover with multiple strategies."""
        if parent1.size <= 1:
            return parent1.copy(), parent2.copy()

        if self.config.crossover_type == "uniform":
            mask = self._rng.random(parent1.size) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        elif self.config.crossover_type == "two_point":
            points = sorted(self._rng.choice(parent1.size, size=2, replace=False))
            child1 = np.concatenate([parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]])
            child2 = np.concatenate([parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]])
            return child1, child2
        else:  # single-point
            point = self._rng.integers(1, parent1.size)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2

    def _compute_diversity(self, population: np.ndarray) -> float:
        """Compute population diversity."""
        if population.shape[0] < 2:
            return 0.0

        distances = []
        for i in range(population.shape[0]):
            for j in range(i + 1, population.shape[0]):
                dist = np.sum(population[i] != population[j])
                distances.append(dist / population.shape[1])

        return float(np.mean(distances)) if distances else 0.0

    def _resize_population(
        self,
        population: np.ndarray,
        adaptive_params: List[Dict[str, float]],
        fitness: np.ndarray,
        new_size: int,
        min_features: int,
        max_features: int,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Resize population while preserving best individuals."""
        if new_size == population.shape[0]:
            return population, adaptive_params

        if new_size > population.shape[0]:
            # Increase: add random individuals
            n_add = new_size - population.shape[0]
            new_individuals = np.zeros((n_add, population.shape[1]), dtype=int)
            new_params = []

            for i in range(n_add):
                size = self._rng.integers(min_features, max_features + 1)
                active_idx = self._rng.choice(population.shape[1], size=size, replace=False)
                new_individuals[i, active_idx] = 1
                new_params.append({
                    "crossover_prob": self._rng.uniform(
                        self.config.min_crossover_prob, self.config.max_crossover_prob
                    ),
                    "mutation_prob": self._rng.uniform(
                        self.config.min_mutation_prob, self.config.max_mutation_prob
                    ),
                    "mutation_sigma": self._rng.uniform(
                        self.config.mutation_sigma_min, self.config.mutation_sigma_max
                    ),
                })

            population = np.vstack([population, new_individuals])
            adaptive_params.extend(new_params)
        else:
            # Decrease: keep best individuals
            sorted_indices = np.argsort(fitness)[::-1]
            keep_indices = sorted_indices[:new_size]
            population = population[keep_indices]
            adaptive_params = [adaptive_params[i] for i in keep_indices]

        return population, adaptive_params

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform data to selected features."""
        if self.best_individual_ is None:
            raise RuntimeError("The GA selector must be fitted before calling transform().")
        X_arr, _ = self._coerce_input(X, expect_names=False)
        mask = self.get_support()
        return X_arr[:, mask]

    def get_support(self) -> np.ndarray:
        """Get boolean mask of selected features."""
        if self.best_individual_ is None:
            raise RuntimeError("Call fit() before querying the selected features.")
        return self.best_individual_.astype(bool)

    def get_feature_names(self) -> List[str]:
        """Get names of selected features."""
        if self.feature_names_ is None:
            raise RuntimeError("Call fit() before requesting feature names.")
        mask = self.get_support()
        return [name for name, active in zip(self.feature_names_, mask) if active]

    def _coerce_input(
        self,
        X: ArrayLike,
        expect_names: bool = True,
    ) -> Tuple[np.ndarray, Optional[Sequence[str]]]:
        """Coerce input to numpy array and extract feature names."""
        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy(dtype=float)
            feature_names = list(X.columns) if expect_names else None
        else:
            X_arr = np.asarray(X, dtype=float)
            feature_names = None
        return X_arr, feature_names


__all__ = ["AdvancedGAConfig", "AdvancedGeneticFeatureSelector"]

