"""Genetic algorithm utilities tailored for feature selection workflows.

The code in this module replaces the original string-matching example with
components that operate on numerical feature matrices.  It provides a
configurable genetic algorithm that searches for the subset of features that
maximises a selected cross-validated metric for a scikit-learn estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class GAConfig:
    """Configuration parameters for the genetic algorithm.

    Attributes
    ----------
    population_size:
        Number of individuals maintained in the population.
    generations:
        Number of evolutionary iterations to perform.
    crossover_prob:
        Probability of crossover between two selected parents.
    mutation_prob:
        Probability of flipping each bit in an offspring chromosome.
    elitism:
        If ``True`` the best individual is carried forward to the next
        generation to preserve progress.
    tournament_size:
        Number of individuals competing in each tournament selection round.
    min_features:
        Minimum number of active features permitted in an individual.
    max_features:
        Optional upper bound on the number of active features in an
        individual.  When ``None`` defaults to the total number of features.
    random_state:
        Seed for the RNG to ensure reproducible runs.
    """

    population_size: int = 40
    generations: int = 40
    crossover_prob: float = 0.85
    mutation_prob: float = 0.05
    elitism: bool = True
    tournament_size: int = 3
    min_features: int = 3
    max_features: Optional[int] = None
    random_state: Optional[int] = None


class GeneticFeatureSelector:
    """Genetic algorithm based feature subset search.

    Parameters
    ----------
    estimator:
        Scikit-learn compatible classifier. The estimator is cloned for each
        fitness evaluation to avoid state leakage.
    config:
        :class:`GAConfig` instance holding the GA hyper-parameters.
    scoring:
        Name of a scikit-learn scorer or a callable used by
        :func:`sklearn.model_selection.cross_val_score`.
    cv:
        Cross-validation splitter or integer number of folds used during
        fitness evaluation.  When ``None`` a 5-fold stratified split is used.
    verbose:
        If ``True`` evolutionary progress metrics are printed each generation.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        config: Optional[GAConfig] = None,
        scoring: str = "roc_auc",
        cv: Optional[StratifiedKFold] = None,
        verbose: bool = False,
        fitness_function: Optional[Callable[[np.ndarray, np.ndarray, ArrayLike], float]] = None,
    ) -> None:
        self.estimator = estimator
        self.config = config or GAConfig()
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.fitness_function = fitness_function

        self.feature_names_: Optional[Sequence[str]] = None
        self.best_individual_: Optional[np.ndarray] = None
        self.best_score_: float = -np.inf
        self.history_: List[float] = []

        self._rng = np.random.default_rng(self.config.random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, y: ArrayLike) -> "GeneticFeatureSelector":
        X_arr, feature_names = self._coerce_input(X)
        n_features = X_arr.shape[1]

        self.feature_names_ = feature_names
        max_features = self.config.max_features or n_features
        min_features = min(self.config.min_features, n_features)

        population = self._initialize_population(n_features, min_features, max_features)
        fitness = self._evaluate_population(population, X_arr, y)

        for generation in range(self.config.generations):
            population, fitness = self._evolve_generation(
                population, fitness, X_arr, y, min_features, max_features
            )

            self.history_.append(float(np.max(fitness)))
            if fitness.max() > self.best_score_:
                idx = int(np.argmax(fitness))
                self.best_score_ = float(fitness[idx])
                self.best_individual_ = population[idx].copy()

            if self.verbose and self.best_individual_ is not None:
                print(
                    f"[GA] Generation {generation + 1:02d} | "
                    f"Best Score: {self.best_score_:.4f} | "
                    f"Active Features: {int(self.best_individual_.sum())}"
                )

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        if self.best_individual_ is None:
            raise RuntimeError("The GA selector must be fitted before calling transform().")
        X_arr, _ = self._coerce_input(X, expect_names=False)
        mask = self.get_support()
        return X_arr[:, mask]

    def get_support(self) -> np.ndarray:
        if self.best_individual_ is None:
            raise RuntimeError("Call fit() before querying the selected features.")
        return self.best_individual_.astype(bool)

    def get_feature_names(self) -> List[str]:
        if self.feature_names_ is None:
            raise RuntimeError("Call fit() before requesting feature names.")
        mask = self.get_support()
        return [name for name, active in zip(self.feature_names_, mask) if active]

    # ------------------------------------------------------------------
    # GA internals
    # ------------------------------------------------------------------
    def _initialize_population(
        self, n_features: int, min_features: int, max_features: int
    ) -> np.ndarray:
        population = np.zeros((self.config.population_size, n_features), dtype=int)
        for idx in range(self.config.population_size):
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        next_population = []

        if self.config.elitism:
            elite_idx = int(np.argmax(fitness))
            next_population.append(population[elite_idx].copy())

        while len(next_population) < self.config.population_size:
            parent1 = self._tournament_selection(population, fitness)
            parent2 = self._tournament_selection(population, fitness)

            if self._rng.random() < self.config.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = self._mutate(child1, min_features, max_features)
            child2 = self._mutate(child2, min_features, max_features)

            next_population.extend([child1, child2])

        population = np.array(next_population[: self.config.population_size])
        fitness = self._evaluate_population(population, X, y)
        return population, fitness

    def _evaluate_population(
        self, population: np.ndarray, X: np.ndarray, y: ArrayLike
    ) -> np.ndarray:
        scores = np.zeros(population.shape[0], dtype=float)
        cv = self.cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)

        for idx, chromosome in enumerate(population):
            if chromosome.sum() == 0:
                scores[idx] = -np.inf
                continue

            if self.fitness_function is not None:
                try:
                    scores[idx] = float(self.fitness_function(chromosome, X, y))
                except ValueError:
                    scores[idx] = -np.inf
                continue

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
                scores[idx] = float(np.mean(cv_scores))
            except ValueError:
                # If the estimator fails (e.g., due to singular matrices), penalise individual.
                scores[idx] = -np.inf

        return scores

    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        participants = self._rng.choice(
            population.shape[0],
            size=self.config.tournament_size,
            replace=False,
        )
        best_idx = participants[np.argmax(fitness[participants])]
        return population[best_idx]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if parent1.size <= 1:
            return parent1.copy(), parent2.copy()

        point = self._rng.integers(1, parent1.size)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, individual: np.ndarray, min_features: int, max_features: int) -> np.ndarray:
        mutant = individual.copy()
        for idx in range(mutant.size):
            if self._rng.random() < self.config.mutation_prob:
                mutant[idx] = 1 - mutant[idx]

        active = mutant.sum()
        if active < min_features:
            zeros = np.where(mutant == 0)[0]
            if zeros.size > 0:
                flip_count = min_features - active
                chosen = self._rng.choice(zeros, size=flip_count, replace=False)
                mutant[chosen] = 1
        elif active > max_features:
            ones = np.where(mutant == 1)[0]
            flip_count = active - max_features
            chosen = self._rng.choice(ones, size=flip_count, replace=False)
            mutant[chosen] = 0

        if mutant.sum() == 0:
            idx = self._rng.integers(0, mutant.size)
            mutant[idx] = 1

        return mutant

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _coerce_input(
        self,
        X: ArrayLike,
        expect_names: bool = True,
    ) -> Tuple[np.ndarray, Optional[Sequence[str]]]:
        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy(dtype=float)
            feature_names = list(X.columns) if expect_names else None
        else:
            X_arr = np.asarray(X, dtype=float)
            feature_names = None
        return X_arr, feature_names


__all__ = ["GAConfig", "GeneticFeatureSelector"]

