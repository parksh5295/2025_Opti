"""Enhanced Genetic Algorithm with advanced techniques for better performance.

This module provides an improved genetic algorithm implementation with:
- Heuristic initialization based on feature importance
- Multiple crossover strategies (single-point, two-point, uniform)
- Adaptive mutation rate
- Enhanced elitism with diversity preservation
- Local search (hill climbing) for refinement
- Fitness caching to avoid redundant evaluations
- Early stopping based on convergence
- Diversity monitoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class EnhancedGAConfig:
    """Enhanced configuration parameters for the genetic algorithm.

    Attributes
    ----------
    population_size:
        Number of individuals maintained in the population.
    generations:
        Number of evolutionary iterations to perform.
    crossover_prob:
        Probability of crossover between two selected parents.
    mutation_prob:
        Initial mutation probability (can be adaptive).
    adaptive_mutation:
        If True, mutation rate decreases over generations.
    elitism:
        If ``True`` the best individual is carried forward to the next
        generation to preserve progress.
    elite_size:
        Number of elite individuals to preserve (if > 1, uses diversity-based selection).
    tournament_size:
        Number of individuals competing in each tournament selection round.
    min_features:
        Minimum number of active features permitted in an individual.
    max_features:
        Optional upper bound on the number of active features in an
        individual.  When ``None`` defaults to the total number of features.
    random_state:
        Seed for the RNG to ensure reproducible runs.
    crossover_type:
        Type of crossover: 'single', 'two_point', 'uniform'.
    use_local_search:
        If True, apply local search (hill climbing) to best individual.
    local_search_prob:
        Probability of applying local search each generation.
    fitness_cache_size:
        Maximum size of fitness cache (0 to disable).
    early_stopping_patience:
        Number of generations without improvement before stopping (0 to disable).
    diversity_threshold:
        Minimum population diversity (Hamming distance) to maintain.
    heuristic_init_ratio:
        Ratio of population initialized using feature importance (0.0 to 1.0).
    """

    population_size: int = 60
    generations: int = 80
    crossover_prob: float = 0.9
    mutation_prob: float = 0.05
    adaptive_mutation: bool = True
    elitism: bool = True
    elite_size: int = 2
    tournament_size: int = 4
    min_features: int = 3
    max_features: Optional[int] = None
    random_state: Optional[int] = None
    crossover_type: str = "uniform"  # 'single', 'two_point', 'uniform'
    use_local_search: bool = True
    local_search_prob: float = 0.3
    fitness_cache_size: int = 1000
    early_stopping_patience: int = 15
    diversity_threshold: float = 0.1
    heuristic_init_ratio: float = 0.3


class EnhancedGeneticFeatureSelector:
    """Enhanced genetic algorithm based feature subset search.

    This implementation includes advanced techniques for better convergence
    and performance compared to the basic GA.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        config: Optional[EnhancedGAConfig] = None,
        scoring: str = "roc_auc",
        cv: Optional[StratifiedKFold] = None,
        verbose: bool = False,
        fitness_function: Optional[Callable[[np.ndarray, np.ndarray, ArrayLike], float]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> None:
        self.estimator = estimator
        self.config = config or EnhancedGAConfig()
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

        self._rng = np.random.default_rng(self.config.random_state)
        self._fitness_cache: Dict[Tuple, float] = {}
        self._no_improvement_count = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "EnhancedGeneticFeatureSelector":
        X_arr, feature_names = self._coerce_input(X)
        n_features = X_arr.shape[1]

        self.feature_names_ = feature_names
        max_features = self.config.max_features or n_features
        min_features = min(self.config.min_features, n_features)

        # Initialize population with heuristic + random
        population = self._initialize_population(
            n_features, min_features, max_features, X_arr, y
        )
        fitness = self._evaluate_population(population, X_arr, y)

        for generation in range(self.config.generations):
            # Adaptive mutation rate
            current_mutation_prob = self._get_adaptive_mutation_rate(generation)

            # Evolve generation
            population, fitness = self._evolve_generation(
                population, fitness, X_arr, y, min_features, max_features, current_mutation_prob
            )

            # Local search on best individual (with probability)
            if self.config.use_local_search and self._rng.random() < self.config.local_search_prob:
                best_idx = int(np.argmax(fitness))
                improved = self._local_search(
                    population[best_idx], fitness[best_idx], X_arr, y, min_features, max_features
                )
                if improved is not None:
                    population[best_idx] = improved[0]
                    fitness[best_idx] = improved[1]

            # Track history
            best_gen_score = float(np.max(fitness))
            self.history_.append(best_gen_score)
            diversity = self._compute_diversity(population)
            self.diversity_history_.append(diversity)

            # Update best
            if best_gen_score > self.best_score_:
                idx = int(np.argmax(fitness))
                self.best_score_ = float(fitness[idx])
                self.best_individual_ = population[idx].copy()
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1

            if self.verbose and self.best_individual_ is not None:
                print(
                    f"[Enhanced GA] Gen {generation + 1:03d} | "
                    f"Best: {self.best_score_:.4f} | "
                    f"Active: {int(self.best_individual_.sum())} | "
                    f"Diversity: {diversity:.3f} | "
                    f"NoImprove: {self._no_improvement_count}"
                )

            # Early stopping
            if (
                self.config.early_stopping_patience > 0
                and self._no_improvement_count >= self.config.early_stopping_patience
            ):
                if self.verbose:
                    print(f"[Enhanced GA] Early stopping at generation {generation + 1}")
                break

            # Diversity maintenance
            if diversity < self.config.diversity_threshold:
                population = self._maintain_diversity(population, fitness, min_features, max_features)

        return self

    def _initialize_population(
        self,
        n_features: int,
        min_features: int,
        max_features: int,
        X: np.ndarray,
        y: ArrayLike,
    ) -> np.ndarray:
        """Initialize population with mix of heuristic and random individuals."""
        population = np.zeros((self.config.population_size, n_features), dtype=int)
        n_heuristic = int(self.config.population_size * self.config.heuristic_init_ratio)

        # Heuristic initialization based on feature importance
        if self.feature_importance and n_heuristic > 0:
            sorted_features = sorted(
                self.feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            top_features = [f[0] for f in sorted_features[:max_features]]

            for idx in range(n_heuristic):
                # Select subset of top features
                size = self._rng.integers(min_features, min(max_features, len(top_features)) + 1)
                if size <= len(top_features):
                    selected = self._rng.choice(len(top_features), size=size, replace=False)
                    feature_indices = [self.feature_names_.index(top_features[i]) for i in selected]
                    population[idx, feature_indices] = 1

        # Random initialization for remaining
        for idx in range(n_heuristic, self.config.population_size):
            size = self._rng.integers(min_features, max_features + 1)
            active_idx = self._rng.choice(n_features, size=size, replace=False)
            population[idx, active_idx] = 1

        return population

    def _evolve_generation(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
        mutation_prob: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve one generation with enhanced operators."""
        next_population = []

        # Enhanced elitism: preserve top-k with diversity
        if self.config.elitism:
            elite_indices = self._select_elite(population, fitness, self.config.elite_size)
            for idx in elite_indices:
                next_population.append(population[idx].copy())

        # Generate offspring
        while len(next_population) < self.config.population_size:
            parent1 = self._tournament_selection(population, fitness)
            parent2 = self._tournament_selection(population, fitness)

            if self._rng.random() < self.config.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = self._mutate(child1, min_features, max_features, mutation_prob)
            child2 = self._mutate(child2, min_features, max_features, mutation_prob)

            next_population.extend([child1, child2])

        population = np.array(next_population[: self.config.population_size])
        fitness = self._evaluate_population(population, X, y)
        return population, fitness

    def _select_elite(
        self, population: np.ndarray, fitness: np.ndarray, elite_size: int
    ) -> List[int]:
        """Select elite individuals considering diversity."""
        if elite_size == 1:
            return [int(np.argmax(fitness))]

        # Sort by fitness
        sorted_indices = np.argsort(fitness)[::-1]
        elite = [int(sorted_indices[0])]  # Always include best

        # Add diverse individuals
        for idx in sorted_indices[1:]:
            if len(elite) >= elite_size:
                break
            candidate = int(idx)
            # Check diversity: minimum Hamming distance from existing elite
            min_distance = min(
                np.sum(population[candidate] != population[e]) for e in elite
            )
            if min_distance > population.shape[1] * 0.2:  # At least 20% different
                elite.append(candidate)

        # Fill remaining with top fitness if not enough diverse
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
            # Uniform crossover: each bit chosen from either parent with 50% probability
            mask = self._rng.random(parent1.size) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2

        elif self.config.crossover_type == "two_point":
            # Two-point crossover
            points = sorted(self._rng.choice(parent1.size, size=2, replace=False))
            child1 = np.concatenate([parent1[:points[0]], parent2[points[0]:points[1]], parent1[points[1]:]])
            child2 = np.concatenate([parent2[:points[0]], parent1[points[0]:points[1]], parent2[points[1]:]])
            return child1, child2

        else:  # single-point (default)
            point = self._rng.integers(1, parent1.size)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2

    def _mutate(
        self, individual: np.ndarray, min_features: int, max_features: int, mutation_prob: float
    ) -> np.ndarray:
        """Mutate individual with adaptive rate."""
        mutant = individual.copy()
        for idx in range(mutant.size):
            if self._rng.random() < mutation_prob:
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

    def _local_search(
        self,
        individual: np.ndarray,
        current_fitness: float,
        X: np.ndarray,
        y: ArrayLike,
        min_features: int,
        max_features: int,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Apply hill climbing local search to improve individual."""
        best_individual = individual.copy()
        best_fitness = current_fitness
        improved = False

        # Try adding/removing each feature
        for idx in range(individual.size):
            candidate = best_individual.copy()
            candidate[idx] = 1 - candidate[idx]

            # Check constraints
            active = candidate.sum()
            if active < min_features or active > max_features:
                continue

            # Evaluate
            fitness = self._evaluate_individual(candidate, X, y)
            if fitness > best_fitness:
                best_individual = candidate
                best_fitness = fitness
                improved = True
                break  # Greedy: take first improvement

        if improved:
            return best_individual, best_fitness
        return None

    def _evaluate_population(
        self, population: np.ndarray, X: np.ndarray, y: ArrayLike
    ) -> np.ndarray:
        """Evaluate population with fitness caching."""
        scores = np.zeros(population.shape[0], dtype=float)
        cv = self.cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)

        for idx, chromosome in enumerate(population):
            if chromosome.sum() == 0:
                scores[idx] = -np.inf
                continue

            # Check cache
            chrom_tuple = tuple(chromosome.tolist())
            if self.config.fitness_cache_size > 0 and chrom_tuple in self._fitness_cache:
                scores[idx] = self._fitness_cache[chrom_tuple]
                continue

            # Evaluate
            score = self._evaluate_individual(chromosome, X, y, cv)

            # Cache (with size limit)
            if self.config.fitness_cache_size > 0:
                if len(self._fitness_cache) >= self.config.fitness_cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._fitness_cache))
                    del self._fitness_cache[oldest_key]
                self._fitness_cache[chrom_tuple] = score

            scores[idx] = score

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

    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection."""
        participants = self._rng.choice(
            population.shape[0],
            size=self.config.tournament_size,
            replace=False,
        )
        best_idx = participants[np.argmax(fitness[participants])]
        return population[best_idx]

    def _get_adaptive_mutation_rate(self, generation: int) -> float:
        """Compute adaptive mutation rate."""
        if not self.config.adaptive_mutation:
            return self.config.mutation_prob

        # Linear decay: start high, end low
        progress = generation / self.config.generations
        return self.config.mutation_prob * (1.0 - 0.7 * progress)

    def _compute_diversity(self, population: np.ndarray) -> float:
        """Compute population diversity (average pairwise Hamming distance)."""
        if population.shape[0] < 2:
            return 0.0

        distances = []
        for i in range(population.shape[0]):
            for j in range(i + 1, population.shape[0]):
                dist = np.sum(population[i] != population[j])
                distances.append(dist / population.shape[1])  # Normalize

        return float(np.mean(distances)) if distances else 0.0

    def _maintain_diversity(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        min_features: int,
        max_features: int,
    ) -> np.ndarray:
        """Maintain diversity by replacing similar individuals."""
        n_replace = max(1, int(population.shape[0] * 0.2))  # Replace 20%
        new_population = population.copy()

        # Find similar pairs and replace one
        replaced = set()
        for i in range(population.shape[0]):
            if len(replaced) >= n_replace:
                break
            if i in replaced:
                continue

            for j in range(i + 1, population.shape[0]):
                if j in replaced:
                    continue

                similarity = 1.0 - np.sum(population[i] != population[j]) / population.shape[1]
                if similarity > 0.9:  # Very similar
                    # Replace lower fitness one
                    if fitness[i] < fitness[j]:
                        new_population[i] = self._generate_random_individual(
                            population.shape[1], min_features, max_features
                        )
                        replaced.add(i)
                    else:
                        new_population[j] = self._generate_random_individual(
                            population.shape[1], min_features, max_features
                        )
                        replaced.add(j)
                    break

        return new_population

    def _generate_random_individual(
        self, n_features: int, min_features: int, max_features: int
    ) -> np.ndarray:
        """Generate a random individual."""
        individual = np.zeros(n_features, dtype=int)
        size = self._rng.integers(min_features, max_features + 1)
        active_idx = self._rng.choice(n_features, size=size, replace=False)
        individual[active_idx] = 1
        return individual

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


__all__ = ["EnhancedGAConfig", "EnhancedGeneticFeatureSelector"]

