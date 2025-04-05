# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import sys
import time
import unittest
import uuid
import dataclasses
import traceback
import unittest.mock
from abc import ABC, abstractmethod
from collections.abc import Iterator, MutableSequence, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    Tuple,
    cast,
    runtime_checkable,
)

try:
    from PIL import Image, ImageDraw
    import numpy as np
    import numpy.typing as npt
    import matplotlib.figure
    import matplotlib.pyplot as plt
    import scipy.stats as stats
except ImportError as import_error:
    print(
        f"Error: Missing dependencies -> {import_error}. "
        "Please run: pip install Pillow numpy matplotlib scipy"
    )
    sys.exit(1)


Position: TypeAlias = float
Frequency: TypeAlias = int
Color: TypeAlias = Tuple[int, int, int]
NDArrayF64: TypeAlias = npt.NDArray[np.float64]
NDArrayInt: TypeAlias = npt.NDArray[np.int_]
MarkovMethod: TypeAlias = Literal[
    "brute_force", "eigenvalue", "linear_system", "monte_carlo"
]
GaltonSimMode: TypeAlias = Literal["physics_based", "simple_random_walk"]

PRECISION_TOLERANCE: Final[float] = 1e-9
DEFAULT_OUTPUT_DIR: Final[Path] = Path("./simulation_outputs").resolve()
KILO: Final[int] = 1000
DEFAULT_MAX_WORKERS: Final[int] = min(8, os.cpu_count() or 1)
DEFAULT_SEED_FUNC: Callable[[], int] = lambda: int(
    time.time() * 1_000_000
) % (2**32)


class SimulationError(Exception):
    pass


class VisualizationError(Exception):
    pass


class ConfigError(ValueError):
    pass


class InvalidTransitionMatrixError(SimulationError):
    pass


def _create_unique_filename(prefix: str, suffix: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{unique_id}.{suffix}"


def _normalize_probability_vector(
    vector: NDArrayF64, num_states: int
) -> NDArrayF64:
    if num_states <= 0:
        return np.array([], dtype=np.float64)
    if np.any(np.isnan(vector)):
        return vector

    vec = np.maximum(vector, 0.0)
    vec_sum = np.sum(vec)

    uniform_dist = (
        np.ones(num_states) / num_states
        if num_states > 0
        else np.array([])
    )
    if abs(vec_sum) < PRECISION_TOLERANCE:
        return uniform_dist

    if abs(vec_sum - 1.0) > 1e-6:
        if vec_sum > PRECISION_TOLERANCE:
            vec = vec / vec_sum
        else:
            return uniform_dist

    return np.clip(vec, 0.0, 1.0)


def _ensure_output_dir(file_path: Path) -> Path | None:
    try:
        resolved_path = file_path.resolve()
        output_dir = resolved_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return resolved_path
    except OSError as e:
        print(
            f"Warning: Directory access error for {output_dir}: {e}",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"Warning: Unexpected error checking/creating directory {output_dir}: {e}",
            file=sys.stderr,
        )
        return None


def _validate_positive_ints(*named_values: Tuple[str, Any]) -> None:
    for name, value in named_values:
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(
                f"Configuration error: '{name}' must be a positive integer, got {value}."
            )


def _validate_non_negative_ints(*named_values: Tuple[str, Any]) -> None:
    for name, value in named_values:
        if not isinstance(value, int) or value < 0:
            raise ConfigError(
                f"Configuration error: '{name}' must be a non-negative integer, got {value}."
            )


def _validate_non_negative_nums(*named_values: Tuple[str, Any]) -> None:
    for name, value in named_values:
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(
                f"Configuration error: '{name}' must be a non-negative number, got {value}."
            )


def _validate_floats_0_1(*named_values: Tuple[str, Any]) -> None:
    for name, value in named_values:
        if not isinstance(value, float) or not (0.0 <= value <= 1.0):
            raise ConfigError(
                f"Configuration error: '{name}' must be a float between 0.0 and 1.0 (inclusive), got {value}."
            )


def _validate_floats_exclusive_0_1(*named_values: Tuple[str, Any]) -> None:
    for name, value in named_values:
        if not isinstance(value, float) or not (0.0 < value <= 1.0):
            raise ConfigError(
                f"Configuration error: '{name}' must be a float between 0.0 (exclusive) and 1.0 (inclusive), got {value}."
            )


@dataclass(frozen=True)
class GaltonConfig:
    MODE: Final[GaltonSimMode] = "physics_based"
    NUM_ROWS: Final[int] = 15
    NUM_BALLS: Final[int] = 50 * KILO
    BOARD_WIDTH: Final[int] = 800
    BOARD_HEIGHT: Final[int] = 600
    PEG_RADIUS: Final[int] = 3
    DAMPING_FACTOR: Final[float] = 0.75
    ELASTICITY: Final[float] = 0.65
    INITIAL_VARIANCE: Final[float] = 2.5
    MIN_BOUNCE_PROBABILITY: Final[float] = 0.1
    MAX_BOUNCE_PROBABILITY: Final[float] = 0.9
    BOUNCE_DISTANCE_FACTOR: Final[float] = 0.15
    BOUNCE_PROB_CENTER: Final[float] = 0.5
    BACKGROUND_COLOR: Final[Color] = (40, 40, 80)
    LEFT_COLOR: Final[Color] = (100, 100, 220)
    RIGHT_COLOR: Final[Color] = (100, 220, 100)
    SMOOTHING_WINDOW: Final[int] = 3
    HISTOGRAM_BAR_MIN_WIDTH: Final[int] = 1
    BOUNCE_PROB_CACHE_SIZE: Final[int] = 256
    DEFAULT_IMAGE_FILENAME: Final[str] = field(
        default_factory=lambda: _create_unique_filename(
            "galton_board", "png"
        )
    )

    def __post_init__(self) -> None:
        _validate_positive_ints(
            ("NUM_ROWS", self.NUM_ROWS),
            ("NUM_BALLS", self.NUM_BALLS),
            ("BOARD_WIDTH", self.BOARD_WIDTH),
            ("BOARD_HEIGHT", self.BOARD_HEIGHT),
            ("PEG_RADIUS", self.PEG_RADIUS),
            ("HISTOGRAM_BAR_MIN_WIDTH", self.HISTOGRAM_BAR_MIN_WIDTH),
        )
        _validate_non_negative_nums(
            ("INITIAL_VARIANCE", self.INITIAL_VARIANCE),
            ("SMOOTHING_WINDOW", self.SMOOTHING_WINDOW),
        )
        _validate_floats_0_1(
            ("DAMPING_FACTOR", self.DAMPING_FACTOR),
            ("ELASTICITY", self.ELASTICITY),
            ("MIN_BOUNCE_PROBABILITY", self.MIN_BOUNCE_PROBABILITY),
            ("MAX_BOUNCE_PROBABILITY", self.MAX_BOUNCE_PROBABILITY),
            ("BOUNCE_PROB_CENTER", self.BOUNCE_PROB_CENTER),
        )
        if self.MIN_BOUNCE_PROBABILITY > self.MAX_BOUNCE_PROBABILITY:
            raise ConfigError(
                "MIN_BOUNCE_PROBABILITY cannot be greater than MAX_BOUNCE_PROBABILITY."
            )
        valid_modes: set[GaltonSimMode] = {
            "physics_based",
            "simple_random_walk",
        }
        if self.MODE not in valid_modes:
            raise ConfigError(
                f"Invalid Galton board mode '{self.MODE}'. Must be one of {valid_modes}."
            )


@dataclass(frozen=True)
class MarkovConfig:
    RANDOM_WALK_NUM: int = 500
    RANDOM_WALK_STEPS: int = 150
    STEADY_STATE_ITERATIONS: int = 1_000
    STEADY_STATE_BURN_IN: int = 100
    ANALYSIS_METHODS: tuple[MarkovMethod, ...] = field(
        default=(
            "brute_force",
            "eigenvalue",
            "linear_system",
            "monte_carlo",
        )
    )
    SEED: int | None = field(default_factory=DEFAULT_SEED_FUNC)

    def __post_init__(self) -> None:
        _validate_non_negative_ints(
            ("RANDOM_WALK_NUM", self.RANDOM_WALK_NUM),
            ("RANDOM_WALK_STEPS", self.RANDOM_WALK_STEPS),
            ("STEADY_STATE_BURN_IN", self.STEADY_STATE_BURN_IN),
        )
        _validate_positive_ints(
            ("STEADY_STATE_ITERATIONS", self.STEADY_STATE_ITERATIONS)
        )

        if self.STEADY_STATE_BURN_IN >= self.STEADY_STATE_ITERATIONS:
            raise ConfigError(
                "STEADY_STATE_BURN_IN must be less than STEADY_STATE_ITERATIONS."
            )

        valid_methods: set[MarkovMethod] = {
            "brute_force",
            "eigenvalue",
            "linear_system",
            "monte_carlo",
        }
        invalid_methods = set(self.ANALYSIS_METHODS) - valid_methods
        if invalid_methods:
            raise ConfigError(
                f"Invalid analysis methods specified: {sorted(list(invalid_methods))}. Valid are: {valid_methods}"
            )
        if not self.ANALYSIS_METHODS:
            raise ConfigError(
                "At least one analysis method must be specified in ANALYSIS_METHODS."
            )


@dataclass(frozen=True)
class VisConfig:
    FIGSIZE: tuple[int, int] = (12, 7)
    DPI: int = 150
    RANDOM_WALK_ALPHA: float = 0.2
    RANDOM_WALK_LINEWIDTH: float = 0.7
    DIST_GRID_ALPHA: float = 0.3
    DIST_HIST_ALPHA: float = 0.6
    DIST_HIST_BINS: int | str = "auto"
    DIST_X_RANGE: tuple[float, float] = (-30.0, 30.0)
    DIST_X_POINTS: int = 200
    DIST_SNAPSHOT_FIGSIZE: tuple[int, int] = (16, 4)
    DEFAULT_MARKOV_PLOT_FILENAME: Final[str] = field(
        default_factory=lambda: _create_unique_filename(
            "markov_walks", "png"
        )
    )
    DEFAULT_DIST_PLOT_FILENAME: Final[str] = field(
        default_factory=lambda: _create_unique_filename(
            "distribution_snapshots", "png"
        )
    )

    def __post_init__(self) -> None:
        _validate_positive_ints(
            ("FIGSIZE width", self.FIGSIZE[0]),
            ("FIGSIZE height", self.FIGSIZE[1]),
            ("DIST_SNAPSHOT_FIGSIZE width", self.DIST_SNAPSHOT_FIGSIZE[0]),
            ("DIST_SNAPSHOT_FIGSIZE height", self.DIST_SNAPSHOT_FIGSIZE[1]),
            ("DPI", self.DPI),
            ("DIST_X_POINTS", self.DIST_X_POINTS),
        )
        _validate_floats_exclusive_0_1(
            ("RANDOM_WALK_ALPHA", self.RANDOM_WALK_ALPHA),
            ("DIST_GRID_ALPHA", self.DIST_GRID_ALPHA),
            ("DIST_HIST_ALPHA", self.DIST_HIST_ALPHA),
        )

        is_int_bins = isinstance(self.DIST_HIST_BINS, int)
        allowed_str_bins = {
            "auto",
            "fd",
            "doane",
            "scott",
            "stone",
            "rice",
            "sturges",
            "sqrt",
        }
        is_allowed_str_bins = (
            isinstance(self.DIST_HIST_BINS, str)
            and self.DIST_HIST_BINS in allowed_str_bins
        )

        if not (is_int_bins or is_allowed_str_bins):
            raise ConfigError(
                f"DIST_HIST_BINS must be a positive integer or a valid strategy string "
                f"{sorted(list(allowed_str_bins))}, got '{self.DIST_HIST_BINS}'."
            )
        if is_int_bins and cast(int, self.DIST_HIST_BINS) <= 0:
            raise ConfigError(
                "If DIST_HIST_BINS is an integer, it must be positive."
            )


@runtime_checkable
class SteadyStateComputable(Protocol):
    def compute(
        self, matrix: NDArrayF64, init_state: NDArrayF64 | None = None
    ) -> NDArrayF64:
        ...


class SteadyStateComputer(ABC):
    @staticmethod
    def _validate_square_matrix(matrix: NDArrayF64) -> int:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise InvalidTransitionMatrixError(
                f"Matrix must be square, but got shape {matrix.shape}."
            )
        return matrix.shape[0]

    @abstractmethod
    def compute(
        self, matrix: NDArrayF64, init_state: NDArrayF64 | None = None
    ) -> NDArrayF64:
        ...


def _validate_and_normalize_init_state(
    init_state: NDArrayF64, num_states: int
) -> NDArrayF64:
    if init_state.ndim != 1 or init_state.shape[0] != num_states:
        raise SimulationError(
            f"Initial state shape {init_state.shape} is incompatible with matrix dimension ({num_states})."
        )
    return _normalize_probability_vector(init_state.copy(), num_states)


class BruteForceComputer(SteadyStateComputer):
    def __init__(self, iterations: int) -> None:
        _validate_positive_ints(("BruteForce iterations", iterations))
        self._iterations: Final = iterations

    def compute(
        self, matrix: NDArrayF64, init_state: NDArrayF64 | None = None
    ) -> NDArrayF64:
        num_states = self._validate_square_matrix(matrix)
        if num_states == 0:
            return np.array([], dtype=np.float64)

        uniform_dist = np.ones(num_states) / num_states
        current_state = (
            uniform_dist
            if init_state is None
            else _validate_and_normalize_init_state(init_state, num_states)
        )

        try:
            for _ in range(self._iterations):
                current_state = current_state @ matrix
            return _normalize_probability_vector(current_state, num_states)
        except Exception as e:
            raise SimulationError(
                "Brute force (power iteration) computation failed."
            ) from e


class EigenvalueComputer(SteadyStateComputer):
    def compute(
        self, matrix: NDArrayF64, _: NDArrayF64 | None = None
    ) -> NDArrayF64:
        num_states = self._validate_square_matrix(matrix)
        if num_states == 0:
            return np.array([], dtype=np.float64)

        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
            one_indices = np.where(
                np.isclose(eigenvalues, 1.0, atol=PRECISION_TOLERANCE)
            )[0]

            steady_state_transposed: NDArrayF64
            if one_indices.size == 0:
                closest_index = np.argmin(np.abs(eigenvalues - 1.0))
                closest_eigenvalue = eigenvalues[closest_index]
                if not np.isclose(closest_eigenvalue, 1.0, atol=1e-5):
                    print(
                        f"Warning (Eigenvalue Method): No eigenvalue exactly 1 found. Using closest value: {closest_eigenvalue:.5f}. "
                        "Chain might be periodic or reducible.",
                        file=sys.stderr,
                    )
                steady_state_transposed = eigenvectors[:, closest_index]
            else:
                if one_indices.size > 1:
                    print(
                        f"Warning (Eigenvalue Method): Multiple eigenvalues close to 1 found ({eigenvalues[one_indices]}). "
                        "Using the first one. The Markov chain might be reducible.",
                        file=sys.stderr,
                    )
                steady_state_transposed = eigenvectors[:, one_indices[0]]

            steady_state_vector = np.real(steady_state_transposed).flatten()
            return _normalize_probability_vector(
                steady_state_vector, num_states
            )

        except np.linalg.LinAlgError as e:
            raise SimulationError(
                "Eigenvalue decomposition failed. Matrix might be singular or ill-conditioned."
            ) from e
        except Exception as e:
            raise SimulationError(
                "Unexpected error during eigenvalue computation."
            ) from e


class LinearSystemComputer(SteadyStateComputer):
    def compute(
        self, matrix: NDArrayF64, _: NDArrayF64 | None = None
    ) -> NDArrayF64:
        num_states = self._validate_square_matrix(matrix)
        if num_states == 0:
            return np.array([], dtype=np.float64)

        try:
            a_matrix = matrix.T - np.identity(num_states)
            a_matrix[-1, :] = 1.0
            b_vector = np.zeros(num_states)
            b_vector[-1] = 1.0

            solution, residuals, _, _ = np.linalg.lstsq(
                a_matrix, b_vector, rcond=None
            )

            if residuals.size > 0 and residuals[0] > 1e-6:
                print(
                    f"Warning (Linear System Method): Solution has non-trivial residuals ({residuals[0]:.2e}). "
                    "The matrix might be ill-conditioned or the chain reducible.",
                    file=sys.stderr,
                )

            return _normalize_probability_vector(
                solution.flatten(), num_states
            )

        except np.linalg.LinAlgError as e:
            raise SimulationError(
                "Linear system solution failed. Matrix might be singular."
            ) from e
        except Exception as e:
            raise SimulationError(
                "Unexpected error during linear system computation."
            ) from e


class MonteCarloComputer(SteadyStateComputer):
    def __init__(
        self, iterations: int, burn_in: int, rng: np.random.Generator
    ) -> None:
        _validate_positive_ints(("MonteCarlo iterations", iterations))
        _validate_non_negative_ints(("MonteCarlo burn_in", burn_in))
        if burn_in >= iterations:
            raise ValueError(
                "Monte Carlo burn-in period must be less than total iterations."
            )
        self._iterations: Final = iterations
        self._burn_in: Final = burn_in
        self._rng: Final = rng

    def compute(
        self, matrix: NDArrayF64, _: NDArrayF64 | None = None
    ) -> NDArrayF64:
        num_states = self._validate_square_matrix(matrix)
        uniform_dist = (
            np.ones(num_states) / num_states
            if num_states > 0
            else np.array([])
        )
        if num_states == 0:
            return uniform_dist

        num_samples = self._iterations - self._burn_in
        if num_samples <= 0:
            print(
                "Warning (Monte Carlo Method): Effective samples <= 0. Returning uniform distribution.",
                file=sys.stderr,
            )
            return uniform_dist

        try:
            current_state_index = self._rng.integers(0, num_states)
            state_visit_counts = np.zeros(num_states, dtype=np.int64)

            for i in range(self._iterations):
                transition_probs = _normalize_probability_vector(
                    matrix[current_state_index], num_states
                )

                if not np.any(transition_probs > 0):
                    print(
                        f"Warning (Monte Carlo Method): State {current_state_index} has no outgoing transitions in matrix row. "
                        "Performing random jump. Check matrix reducibility/absorbing states.",
                        file=sys.stderr,
                    )
                    next_state_index = self._rng.integers(0, num_states)
                else:
                    next_state_index = self._rng.choice(
                        num_states, p=transition_probs
                    )

                if i >= self._burn_in:
                    state_visit_counts[current_state_index] += 1

                current_state_index = next_state_index

            total_counted_visits = np.sum(state_visit_counts)
            if total_counted_visits == 0:
                print(
                    "Warning (Monte Carlo Method): Simulation recorded zero visits after burn-in. Returning uniform.",
                    file=sys.stderr,
                )
                return uniform_dist

            estimated_distribution = (
                state_visit_counts.astype(float) / total_counted_visits
            )
            return _normalize_probability_vector(
                estimated_distribution, num_states
            )

        except Exception as e:
            raise SimulationError(
                f"Monte Carlo simulation failed unexpectedly: {e}"
            ) from e


class TransitionMatrix:
    _IVME = InvalidTransitionMatrixError

    def __init__(self, data: Any) -> None:
        try:
            matrix = np.array(data, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise self._IVME(
                f"Invalid data type for transition matrix: {e}"
            ) from e

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise self._IVME(
                f"Transition matrix must be square, but got shape {matrix.shape}."
            )

        num_states = matrix.shape[0]
        if num_states > 0:
            if np.any(matrix < -PRECISION_TOLERANCE):
                raise self._IVME(
                    "Transition matrix contains negative probabilities."
                )
            matrix = np.maximum(matrix, 0.0)

            row_sums = matrix.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-8):
                bad_indices = np.where(
                    ~np.isclose(row_sums, 1.0, atol=1e-8)
                )[0]
                bad_sums = row_sums[bad_indices]
                raise self._IVME(
                    f"Rows of the transition matrix must sum to 1 (tolerance 1e-8). "
                    f"Invalid rows found at indices: {bad_indices}. Their sums: {bad_sums}"
                )

        self._matrix: Final[NDArrayF64] = matrix

    @property
    def matrix(self) -> NDArrayF64:
        return self._matrix

    @property
    def num_states(self) -> int:
        return self._matrix.shape[0]

    def __len__(self) -> int:
        return self.num_states

    def __getitem__(self, state_index: int) -> NDArrayF64:
        if not isinstance(state_index, int):
            raise TypeError("State index must be an integer.")
        if not 0 <= state_index < self.num_states:
            raise IndexError(
                f"State index {state_index} out of range for {self.num_states} states."
            )
        return self._matrix[state_index]

    def __repr__(self) -> str:
        return f"TransitionMatrix(num_states={self.num_states})"


class MarkovChainSimulator:
    _SE = SimulationError

    def __init__(self, config: MarkovConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.SEED)
        self._computers = self._initialize_steady_state_computers(config)

    def _initialize_steady_state_computers(
        self, config: MarkovConfig
    ) -> dict[MarkovMethod, SteadyStateComputable]:
        computers: dict[MarkovMethod, SteadyStateComputable] = {}
        method_map: dict[MarkovMethod, type[SteadyStateComputer]] = {
            "brute_force": BruteForceComputer,
            "eigenvalue": EigenvalueComputer,
            "linear_system": LinearSystemComputer,
            "monte_carlo": MonteCarloComputer,
        }

        for method_name in config.ANALYSIS_METHODS:
            computer_class = method_map[method_name]
            try:
                if method_name == "brute_force":
                    computers[method_name] = computer_class(
                        config.STEADY_STATE_ITERATIONS
                    )
                elif method_name == "monte_carlo":
                    computers[method_name] = computer_class(
                        config.STEADY_STATE_ITERATIONS,
                        config.STEADY_STATE_BURN_IN,
                        self._rng,
                    )
                else:
                    computers[method_name] = computer_class()
            except ValueError as e:
                raise ConfigError(
                    f"Error initializing steady-state computer '{method_name}': {e}"
                ) from e
        return computers

    def simulate_random_walks(self) -> NDArrayF64:
        num_walks = self.config.RANDOM_WALK_NUM
        num_steps = self.config.RANDOM_WALK_STEPS

        if num_walks <= 0 or num_steps < 0:
            print(
                "Info: Skipping random walk simulation (0 walks or negative steps)."
            )
            return np.zeros((max(0, num_walks), max(0, num_steps + 1)))

        try:
            steps = self._rng.choice(
                [-1, 1], size=(num_walks, num_steps), p=[0.5, 0.5]
            )
            positions = np.zeros(
                (num_walks, num_steps + 1), dtype=float
            )
            np.cumsum(steps, axis=1, out=positions[:, 1:])
            return positions
        except (ValueError, MemoryError) as e:
            raise self._SE(
                f"Failed to generate random walks (check parameters/memory): {e}"
            ) from e
        except Exception as e:
            raise self._SE(
                f"An unexpected error occurred during random walk simulation: {e}"
            ) from e

    def analyze_steady_state(
        self,
        transition_matrix: TransitionMatrix,
        initial_distribution: NDArrayF64 | None = None,
    ) -> dict[MarkovMethod, NDArrayF64]:
        matrix = transition_matrix.matrix
        num_states = transition_matrix.num_states
        methods_to_run = list(self._computers.keys())

        if num_states == 0 or not methods_to_run:
            return {
                method: np.array([], dtype=np.float64)
                for method in methods_to_run
            }

        normalized_initial_state: NDArrayF64 | None = None
        if initial_distribution is not None:
            try:
                normalized_initial_state = _validate_and_normalize_init_state(
                    initial_distribution, num_states
                )
            except SimulationError as e:
                print(
                    f"Warning: Invalid initial state provided for steady state analysis. "
                    f"Brute force method will use uniform distribution. Error: {e}",
                    file=sys.stderr,
                )

        results: dict[MarkovMethod, NDArrayF64] = {}
        max_workers = min(len(methods_to_run), DEFAULT_MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_method = {
                executor.submit(
                    self._computers[method].compute,
                    matrix,
                    normalized_initial_state
                    if method == "brute_force"
                    and normalized_initial_state is not None
                    else None,
                ): method
                for method in methods_to_run
            }

            for future in as_completed(future_to_method):
                method = future_to_method[future]
                try:
                    result_vector = future.result()
                    if not isinstance(
                        result_vector, np.ndarray
                    ) or result_vector.shape != (num_states,):
                        raise SimulationError(
                            f"Method '{method}' returned result with unexpected shape {result_vector.shape} (expected ({num_states},))."
                        )
                    if np.any(~np.isfinite(result_vector)):
                        raise SimulationError(
                            f"Method '{method}' returned result containing non-finite values (NaN or Inf)."
                        )

                    results[method] = _normalize_probability_vector(
                        result_vector, num_states
                    )

                except (
                    SimulationError,
                    InvalidTransitionMatrixError,
                ) as e:
                    print(
                        f"Warning: Steady-state computation failed for method '{method}': {e}",
                        file=sys.stderr,
                    )
                    results[method] = np.full(num_states, np.nan)
                except Exception as e:
                    print(
                        f"Warning: Unexpected error occurred in steady-state method '{method}': {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                    results[method] = np.full(num_states, np.nan)

        return results


@dataclass
class GaltonBoard:
    config: GaltonConfig = field(default_factory=GaltonConfig)
    slot_counts: MutableSequence[Frequency] = field(
        init=False, default_factory=list
    )
    _rng: np.random.Generator = field(init=False, repr=False)
    _board_center: Position = field(init=False)
    _horizontal_bounds: Tuple[Position, Position] = field(init=False)
    _image: Image.Image | None = field(
        init=False, default=None, repr=False
    )
    _draw: ImageDraw.Draw | None = field(
        init=False, default=None, repr=False
    )

    @staticmethod
    @lru_cache(maxsize=GaltonConfig.BOUNCE_PROB_CACHE_SIZE)
    def _bounce_prob_calculator(
        normalized_distance: float,
        center_prob: float,
        distance_factor: float,
        min_prob: float,
        max_prob: float,
    ) -> float:
        probability = center_prob + distance_factor * normalized_distance
        return np.clip(probability, min_prob, max_prob)

    def __post_init__(self) -> None:
        GaltonBoard._bounce_prob_calculator.cache_clear()
        self.set_rng_seed(DEFAULT_SEED_FUNC())

        cfg = self.config
        self.slot_counts = [0] * cfg.BOARD_WIDTH
        self._board_center = cfg.BOARD_WIDTH / 2.0

        peg_radius_f = float(cfg.PEG_RADIUS)
        board_width_f = float(cfg.BOARD_WIDTH)
        is_physics_mode = cfg.MODE == "physics_based"

        min_bound_physics = peg_radius_f
        max_bound_physics = max(
            min_bound_physics, board_width_f - peg_radius_f
        )

        min_bound_simple = 0.0
        max_bound_simple = max(0.0, board_width_f - 1.0)

        min_bound = min_bound_physics if is_physics_mode else min_bound_simple
        max_bound = max_bound_physics if is_physics_mode else max_bound_simple

        if min_bound > max_bound + PRECISION_TOLERANCE:
            raise ConfigError(
                f"Invalid horizontal bounds calculated (min={min_bound}, max={max_bound}). "
                "Check board width and peg radius configuration."
            )
        self._horizontal_bounds = (min_bound, max_bound)

        self._invalidate_image_cache()

    def set_rng_seed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def simulate(self) -> None:
        num_balls = self.config.NUM_BALLS
        board_width = self.config.BOARD_WIDTH

        self.slot_counts = [0] * board_width

        if num_balls <= 0 or board_width <= 0:
            print("Info: Simulation skipped (0 balls or 0 board width).")
            self._invalidate_image_cache()
            return

        path_generator = (
            self._generate_physics_paths()
            if self.config.MODE == "physics_based"
            else self._generate_simple_paths()
        )

        try:
            final_positions = np.fromiter(
                path_generator, dtype=float, count=num_balls
            )
            slot_indices = np.clip(
                np.round(final_positions).astype(np.intp),
                0,
                max(0, board_width - 1),
            )
            counts = np.bincount(slot_indices, minlength=board_width)
            self._apply_smoothing(counts)

        except Exception as e:
            raise SimulationError(
                f"Error during Galton simulation or binning results: {e}"
            ) from e

        self._invalidate_image_cache()

    def _generate_physics_paths(self) -> Iterator[Position]:
        cfg = self.config
        peg_radius_f = float(cfg.PEG_RADIUS)

        if peg_radius_f <= PRECISION_TOLERANCE:
            print(
                "Warning: Physics mode selected but peg radius is near zero. "
                "Falling back to simple random walk simulation.",
                file=sys.stderr,
            )
            yield from self._generate_simple_paths()
            return

        peg_diameter = 2.0 * peg_radius_f
        inv_peg_radius = 1.0 / peg_radius_f
        bounce_calculator = self._bounce_prob_calculator
        normal_noise = self._rng.normal
        random_uniform = self._rng.random
        constrain = self._constrain_position
        num_rows = cfg.NUM_ROWS
        num_balls = cfg.NUM_BALLS
        board_center = self._board_center
        initial_variance = cfg.INITIAL_VARIANCE
        damping = cfg.DAMPING_FACTOR
        elasticity = cfg.ELASTICITY
        prob_center = cfg.BOUNCE_PROB_CENTER
        dist_factor = cfg.BOUNCE_DISTANCE_FACTOR
        min_prob = cfg.MIN_BOUNCE_PROBABILITY
        max_prob = cfg.MAX_BOUNCE_PROBABILITY

        for _ in range(num_balls):
            current_pos = constrain(
                board_center + normal_noise(0, initial_variance)
            )
            horizontal_momentum = 0.0
            row_parity = 0

            for _ in range(num_rows):
                row_offset = row_parity * peg_radius_f
                peg_center_x = (
                    np.round((current_pos - row_offset) / peg_diameter)
                    * peg_diameter
                    + row_offset
                )
                normalized_dist = np.clip(
                    (current_pos - peg_center_x) * inv_peg_radius, -1.0, 1.0
                )
                prob_bounce_right = bounce_calculator(
                    normalized_dist,
                    prob_center,
                    dist_factor,
                    min_prob,
                    max_prob,
                )
                bounce_direction = (
                    1 if random_uniform() < prob_bounce_right else -1
                )
                impact_factor = (1.0 - abs(normalized_dist)) * elasticity
                momentum_change = (
                    bounce_direction * impact_factor * peg_diameter
                )
                horizontal_momentum = (
                    horizontal_momentum * damping + momentum_change
                )
                current_pos = constrain(current_pos + horizontal_momentum)
                row_parity ^= 1

            yield current_pos

    def _constrain_position(self, position: Position) -> Position:
        return np.clip(
            position, self._horizontal_bounds[0], self._horizontal_bounds[1]
        )

    def _generate_simple_paths(self) -> Iterator[Position]:
        cfg = self.config
        num_rows = cfg.NUM_ROWS
        board_width = cfg.BOARD_WIDTH
        num_balls = cfg.NUM_BALLS

        start_position_index = board_width // 2
        max_index = max(0, board_width - 1)

        if num_rows == 0:
            yield from np.full(num_balls, float(start_position_index))
            return

        total_shifts = self._rng.choice(
            [-1, 1], size=(num_balls, num_rows), p=[0.5, 0.5]
        ).sum(axis=1)
        final_indices = np.clip(
            start_position_index + total_shifts, 0, max_index
        )
        yield from final_indices.astype(float)

    def _apply_smoothing(
        self, counts: NDArrayInt | Sequence[Frequency]
    ) -> None:
        window_size = self.config.SMOOTHING_WINDOW
        num_slots = len(counts)

        if window_size <= 1 or num_slots == 0:
            self.slot_counts = list(counts)
            return

        window_size = min(window_size, num_slots)
        kernel = np.ones(window_size) / window_size
        data = np.asarray(counts, dtype=np.float64)
        smoothed_counts = np.convolve(data, kernel, mode="same")
        self.slot_counts = [
            int(round(x)) for x in np.maximum(0, smoothed_counts)
        ]

    def _invalidate_image_cache(self) -> None:
        self._image = None
        self._draw = None

    def _prepare_image_context(self) -> None:
        if self._image is not None and self._draw is not None:
            return

        cfg = self.config
        try:
            self._image = Image.new(
                "RGB",
                (cfg.BOARD_WIDTH, cfg.BOARD_HEIGHT),
                cfg.BACKGROUND_COLOR,
            )
            self._draw = ImageDraw.Draw(self._image)
        except Exception as e:
            self._invalidate_image_cache()
            raise VisualizationError(
                f"Failed to initialize image context (size: {cfg.BOARD_WIDTH}x{cfg.BOARD_HEIGHT}): {e}"
            ) from e

    def generate_image(self) -> Image.Image:
        self._prepare_image_context()
        if self._image is None or self._draw is None:
            raise VisualizationError(
                "Image context is not available for drawing. Preparation failed."
            )

        counts = self.slot_counts
        cfg = self.config
        max_frequency = max(counts) if counts else 0

        if not counts or max_frequency <= 0:
            return self._image

        num_slots = len(counts)
        bar_width = max(
            cfg.HISTOGRAM_BAR_MIN_WIDTH,
            cfg.BOARD_WIDTH // num_slots if num_slots > 0 else cfg.BOARD_WIDTH,
        )
        board_height_f = float(cfg.BOARD_HEIGHT)
        board_center_f = self._board_center
        height_scale = (
            board_height_f / max_frequency if max_frequency > 0 else 0
        )

        try:
            for i, frequency in enumerate(counts):
                if frequency <= 0:
                    continue

                bar_height = max(1, int(round(frequency * height_scale)))
                x0 = i * bar_width
                y0 = cfg.BOARD_HEIGHT - bar_height
                x1 = x0 + bar_width
                y1 = cfg.BOARD_HEIGHT

                bar_center_x = x0 + bar_width / 2.0
                bar_color = (
                    cfg.LEFT_COLOR
                    if bar_center_x < board_center_f
                    else cfg.RIGHT_COLOR
                )

                self._draw.rectangle((x0, y0, x1, y1), fill=bar_color)

        except Exception as e:
            raise VisualizationError(
                f"Failed to draw histogram bars: {e}"
            ) from e

        return self._image

    def save_image(self, filename: str | Path | None = None) -> str:
        output_path_arg = filename or (
            DEFAULT_OUTPUT_DIR / self.config.DEFAULT_IMAGE_FILENAME
        )
        output_path = Path(output_path_arg)

        resolved_path = _ensure_output_dir(output_path)
        if resolved_path is None:
            raise IOError(
                f"Invalid output path or directory creation failed for '{output_path}'. Image not saved."
            )

        try:
            image_to_save = self.generate_image()
            image_to_save.save(resolved_path)
            return str(resolved_path)
        except (OSError, IOError, VisualizationError) as e:
            raise IOError(
                f"Failed to save Galton image to '{resolved_path}': {e}"
            ) from e
        except Exception as e:
            raise IOError(
                f"Unexpected error saving Galton image to '{resolved_path}': {e}"
            ) from e


class Visualizer:
    def __init__(self, config: VisConfig) -> None:
        self.config = config

    def _save_or_show(
        self,
        fig: matplotlib.figure.Figure,
        show_plot: bool,
        save_path: Path | None,
    ) -> None:
        try:
            if save_path:
                target_path = _ensure_output_dir(save_path)
                if target_path:
                    try:
                        fig.savefig(
                            target_path,
                            dpi=self.config.DPI,
                            bbox_inches="tight",
                        )
                    except Exception as e:
                        print(
                            f"Warning: Failed to save plot to {target_path}: {e}",
                            file=sys.stderr,
                        )
                else:
                    print(
                        f"Warning: Plot not saved due to directory issue for path: {save_path}",
                        file=sys.stderr,
                    )

            if show_plot:
                try:
                    plt.show()
                except Exception as e:
                    print(
                        f"Warning: Failed to display plot interactively: {e}",
                        file=sys.stderr,
                    )
        finally:
            plt.close(fig)

    def plot_random_walks(
        self, walks: NDArrayF64, show_plot: bool = True, save_path: Path | None = None
    ) -> None:
        num_walks, num_steps_plus_1 = walks.shape
        if num_walks == 0:
            print("Info: No random walks data provided to plot.")
            return

        fig: matplotlib.figure.Figure | None = None
        try:
            fig, ax = plt.subplots(figsize=self.config.FIGSIZE)
            steps_axis = np.arange(num_steps_plus_1)

            ax.plot(
                steps_axis,
                walks.T,
                alpha=self.config.RANDOM_WALK_ALPHA,
                linewidth=self.config.RANDOM_WALK_LINEWIDTH,
            )

            title = f"{num_walks} Random Walk{'s' if num_walks != 1 else ''}"
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Step Number")
            ax.set_ylabel("Position")
            ax.grid(
                True, alpha=self.config.DIST_GRID_ALPHA, linestyle=":"
            )
            ax.margins(x=0.02)

            self._save_or_show(fig, show_plot, save_path)
            fig = None

        except Exception as e:
            if fig:
                plt.close(fig)
            raise VisualizationError(
                f"Failed to plot random walks: {e}"
            ) from e

    def plot_distribution_snapshots(
        self,
        walks: NDArrayF64,
        snapshot_times: list[int] | None = None,
        show_plot: bool = True,
        save_path: Path | None = None,
    ) -> None:
        num_walks, num_steps_plus_1 = walks.shape
        max_step = num_steps_plus_1 - 1

        if num_walks == 0 or max_step <= 0:
            print(
                "Info: No walks data or insufficient steps (<1) for distribution snapshots."
            )
            return

        valid_times = self._get_valid_snapshot_times(
            snapshot_times, max_step
        )
        if not valid_times:
            print(
                "Info: No valid snapshot times selected or available for plotting."
            )
            return

        num_plots = len(valid_times)
        fig: matplotlib.figure.Figure | None = None
        try:
            base_width, base_height = self.config.DIST_SNAPSHOT_FIGSIZE
            adjusted_width = max(base_width, base_width * num_plots / 4)

            fig, axes = plt.subplots(
                1,
                num_plots,
                figsize=(adjusted_width, base_height),
                sharex=True,
                sharey=True,
            )
            axes_list = np.atleast_1d(axes)

            plot_x_range, pdf_x_points = self._get_distribution_plot_range(
                walks, valid_times
            )

            for i, time_step in enumerate(valid_times):
                self._plot_single_distribution_snapshot(
                    ax=axes_list[i],
                    positions_at_time=walks[:, time_step],
                    time_step=time_step,
                    x_range=plot_x_range,
                    pdf_x_points=pdf_x_points,
                    is_first_plot=(i == 0),
                )

            fig.suptitle(
                "Distribution of Random Walkers Over Time", fontsize=16
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            self._save_or_show(fig, show_plot, save_path)
            fig = None

        except Exception as e:
            if fig:
                plt.close(fig)
            raise VisualizationError(
                f"Failed to plot distribution snapshots: {e}"
            ) from e

    def _get_valid_snapshot_times(
        self, requested_times: list[int] | None, max_step: int
    ) -> list[int]:
        if max_step <= 0:
            return []

        times: list[int]

        if requested_times:
            times = sorted(
                list(set(t for t in requested_times if 0 < t <= max_step))
            )
        else:
            num_snapshots = min(5, max_step)

            if max_step < 10:
                indices = np.linspace(
                    1, max_step, num=num_snapshots, dtype=int, endpoint=True
                )
            else:
                log_start = np.log10(max(1, max_step // 10))
                log_end = np.log10(max(1, max_step))
                if log_start >= log_end - PRECISION_TOLERANCE:
                    indices = np.array([max_step], dtype=int)
                else:
                    indices = np.logspace(
                        log_start,
                        log_end,
                        num=num_snapshots,
                        dtype=int,
                        endpoint=True,
                    )

            times = sorted(list(set(t for t in indices if t > 0)))
            if max_step > 0 and max_step not in times:
                times = sorted(list(set(times + [max_step])))

        max_plots = 10
        if len(times) > max_plots:
            indices = np.linspace(
                0, len(times) - 1, max_plots, dtype=int
            )
            times = [times[i] for i in indices]

        return times

    def _get_distribution_plot_range(
        self, walks: NDArrayF64, times: list[int]
    ) -> Tuple[tuple[float, float], NDArrayF64]:
        cfg = self.config
        default_range = cfg.DIST_X_RANGE
        x_range = default_range

        if times and walks.size > 0:
            try:
                positions_at_times = walks[:, times]
                if positions_at_times.size > 0:
                    q_low, q_high = np.percentile(
                        positions_at_times, [1, 99]
                    )
                    data_range = q_high - q_low
                    padding = max(data_range * 0.1, 1.0)
                    x_min = min(q_low - padding, default_range[0])
                    x_max = max(q_high + padding, default_range[1])
                    x_range = (x_min, x_max)
            except (IndexError, ValueError) as e:
                print(
                    f"Warning: Could not determine plot range from data percentiles ({e}). Using default range.",
                    file=sys.stderr,
                )
                pass

        pdf_x_points = np.linspace(
            x_range[0], x_range[1], cfg.DIST_X_POINTS
        )
        return x_range, pdf_x_points

    def _plot_single_distribution_snapshot(
        self,
        ax: plt.Axes,
        positions_at_time: NDArrayF64,
        time_step: int,
        x_range: tuple[float, float],
        pdf_x_points: NDArrayF64,
        is_first_plot: bool,
    ) -> None:
        cfg = self.config

        ax.hist(
            positions_at_time,
            bins=cfg.DIST_HIST_BINS,
            density=True,
            alpha=cfg.DIST_HIST_ALPHA,
            color="skyblue",
            label=f"Data (t={time_step})",
        )

        try:
            std_dev = math.sqrt(float(time_step))
        except ValueError:
            std_dev = 0.0

        if std_dev > PRECISION_TOLERANCE:
            theoretical_pdf = stats.norm.pdf(
                pdf_x_points, loc=0, scale=std_dev
            )
            ax.plot(
                pdf_x_points,
                theoretical_pdf,
                "r-",
                linewidth=cfg.RANDOM_WALK_LINEWIDTH * 1.5,
                label=f"Theory N(0, {std_dev**2:.1f})",
            )
        else:
            ax.axvline(
                0, color="red", linestyle="--", label="Theory (Dirac at 0)"
            )

        ax.set_title(f"Time t = {time_step}")
        ax.set_xlabel("Position")
        ax.grid(True, alpha=cfg.DIST_GRID_ALPHA, linestyle=":")
        ax.legend(fontsize="small")
        ax.set_xlim(x_range)

        if is_first_plot:
            ax.set_ylabel("Probability Density")


class CityMigrationSimulation:
    DEFAULT_CITIES: Final[list[str]] = ["Raleigh", "Chapel Hill", "Durham"]
    DEFAULT_TM_DATA: Final[NDArrayF64] = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.04, 0.01, 0.95],
        ]
    )
    DEFAULT_INITIAL_POPULATION: Final[NDArrayF64] = np.array(
        [300 * KILO, 300 * KILO, 300 * KILO], dtype=float
    )

    def __init__(
        self,
        transition_matrix: TransitionMatrix,
        initial_population: NDArrayF64,
        city_names: list[str],
        simulator: MarkovChainSimulator,
    ) -> None:
        num_states = transition_matrix.num_states
        population_vector = initial_population.flatten()

        if len(city_names) != num_states:
            raise ValueError(
                f"Number of city names ({len(city_names)}) must match the "
                f"number of states in the transition matrix ({num_states})."
            )
        if population_vector.size != num_states:
            raise ValueError(
                f"Initial population size ({population_vector.size}) must match the "
                f"number of states in the transition matrix ({num_states})."
            )
        if np.any(population_vector < 0):
            raise ValueError("Initial populations cannot be negative.")

        self.transition_matrix = transition_matrix
        self.city_names = list(city_names)
        self.simulator = simulator
        self.total_population = max(0.0, population_vector.sum())

        if self.total_population > PRECISION_TOLERANCE and num_states > 0:
            self.initial_distribution = _normalize_probability_vector(
                population_vector, num_states
            )
        elif num_states > 0:
            print(
                "Warning: Total initial population is zero or negligible. "
                "Using uniform initial distribution for analysis.",
                file=sys.stderr,
            )
            self.initial_distribution = np.ones(num_states) / num_states
        else:
            self.initial_distribution = np.array([], dtype=float)

    @classmethod
    def create_default(
        cls, simulator: MarkovChainSimulator
    ) -> CityMigrationSimulation:
        try:
            default_tm = TransitionMatrix(cls.DEFAULT_TM_DATA)
        except InvalidTransitionMatrixError as e:
            raise ConfigError(
                f"Default transition matrix data is invalid: {e}"
            ) from e

        return cls(
            transition_matrix=default_tm,
            initial_population=cls.DEFAULT_INITIAL_POPULATION.copy(),
            city_names=list(cls.DEFAULT_CITIES),
            simulator=simulator,
        )

    def run_and_display(self) -> dict[MarkovMethod, NDArrayF64]:
        print("\n--- Starting City Migration Steady-State Analysis ---")
        try:
            steady_state_results = self.simulator.analyze_steady_state(
                self.transition_matrix, self.initial_distribution
            )
            self._display_results_table(steady_state_results)
            print("--- City Migration Analysis Complete ---")
            return steady_state_results
        except SimulationError as e:
            print(
                f"ERROR: City migration analysis failed during computation: {e}",
                file=sys.stderr,
            )
            raise
        except Exception as e:
            print(
                f"ERROR: An unexpected error occurred during city migration analysis: {e}",
                file=sys.stderr,
            )
            raise

    def _display_results_table(
        self, results: dict[MarkovMethod, NDArrayF64]
    ) -> None:
        valid_methods = sorted(
            [
                method
                for method, vector in results.items()
                if vector.size > 0 and not np.any(np.isnan(vector))
            ]
        )

        max_table_width = 78
        title = "City Migration Steady-State Population Estimates"
        header_separator = "=" * max_table_width
        print(
            f"\n{header_separator}\n{title:^{max_table_width}}\n{header_separator}"
        )

        if not valid_methods:
            print(
                "\nNo valid steady-state results were computed or available for display."
            )
            print(header_separator + "\n")
            return

        max_city_name_len = (
            max(len(city) for city in self.city_names)
            if self.city_names
            else 0
        )
        city_col_width = max(15, max_city_name_len + 2)
        num_data_cols = len(valid_methods)
        data_col_width = max(
            18,
            (max_table_width - city_col_width) // num_data_cols
            if num_data_cols > 0
            else 18,
        )

        header_parts = [f"{'City':<{city_col_width}}"] + [
            f"{method.replace('_', ' ').title():>{data_col_width}}"
            for method in valid_methods
        ]
        header_line = "".join(header_parts)
        table_separator = "-" * len(header_line)
        print(f"\n{header_line}\n{table_separator}")

        scaled_populations: dict[MarkovMethod, NDArrayF64] = {}
        display_probs = self.total_population <= PRECISION_TOLERANCE
        if not display_probs:
            scaled_populations = {
                method: results[method] * self.total_population
                for method in valid_methods
            }
        else:
            scaled_populations = {
                method: results[method] for method in valid_methods
            }
            print(
                "(Displaying probabilities as total population is zero or negligible)"
            )

        for i, city in enumerate(self.city_names):
            row_parts = [f"{city:<{city_col_width}}"]
            if not display_probs:
                row_parts.extend(
                    [
                        f"{scaled_populations[method][i]:>{data_col_width},.0f}"
                        for method in valid_methods
                    ]
                )
            else:
                row_parts.extend(
                    [
                        f"{scaled_populations[method][i]:>{data_col_width}.4f}"
                        for method in valid_methods
                    ]
                )
            print("".join(row_parts))

        print(table_separator)
        total_row_parts = [f"{'Total':<{city_col_width}}"]
        if not display_probs:
            total_row_parts.extend(
                [
                    f"{np.sum(scaled_populations[method]):>{data_col_width},.0f}"
                    for method in valid_methods
                ]
            )
        else:
            total_row_parts.extend(
                [
                    f"{np.sum(scaled_populations[method]):>{data_col_width}.4f}"
                    for method in valid_methods
                ]
            )
        print("".join(total_row_parts))
        print(table_separator)
        if not display_probs:
            print(
                "(Estimates based on calculated steady-state probabilities and total initial population)"
            )
        print(header_separator + "\n")


class SimulationRunner:
    def __init__(
        self,
        galton_config: GaltonConfig | None = None,
        markov_config: MarkovConfig | None = None,
        vis_config: VisConfig | None = None,
    ) -> None:
        try:
            self.g_cfg = galton_config or GaltonConfig()
            self.m_cfg = markov_config or MarkovConfig()
            self.v_cfg = vis_config or VisConfig()
        except ConfigError as e:
            raise ConfigError(
                f"Configuration initialization failed: {e}"
            ) from e

        self._rng = np.random.default_rng(self.m_cfg.SEED)
        self.galton_board = GaltonBoard(self.g_cfg)
        self.galton_board.set_rng_seed(self.m_cfg.SEED)
        self.markov_sim = MarkovChainSimulator(self.m_cfg)
        self.visualizer = Visualizer(self.v_cfg)
        self.city_migration: CityMigrationSimulation | None = None

    def _run_task(
        self,
        task_name: str,
        task_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        print(f"\n--- Running {task_name} ---")
        start_time = time.monotonic()
        success = False
        try:
            task_func(*args, **kwargs)
            success = True
        except (
            SimulationError,
            VisualizationError,
            ConfigError,
            IOError,
            InvalidTransitionMatrixError,
        ) as e:
            print(
                f"ERROR during {task_name}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"UNEXPECTED ERROR during {task_name}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
        finally:
            elapsed_time = time.monotonic() - start_time
            status = "OK" if success else "FAILED"
            print(
                f"--- {task_name} {status} (Duration: {elapsed_time:.2f}s) ---"
            )
        return success

    def run_galton_simulation(self, save_image: bool = True) -> bool:
        def task(save: bool):
            print(
                f"Simulating Galton Board (mode: {self.g_cfg.MODE}, balls: {self.g_cfg.NUM_BALLS:,}, rows: {self.g_cfg.NUM_ROWS})..."
            )
            self.galton_board.simulate()
            print("Simulation complete.")
            if save:
                print("Generating and saving image...")
                saved_path = self.galton_board.save_image()
                print(f"Galton image saved: {saved_path}")
            else:
                print("Generating image (not saving)...")
                _ = self.galton_board.generate_image()
                print("Image generated.")

        return self._run_task("Galton Board Simulation", task, save_image)

    def run_random_walk_simulation(
        self, show_plots: bool = True, save_plots: bool = True
    ) -> bool:
        def task(show: bool, save: bool):
            print(
                f"Simulating {self.m_cfg.RANDOM_WALK_NUM:,} random walks ({self.m_cfg.RANDOM_WALK_STEPS:,} steps each)..."
            )
            walks = self.markov_sim.simulate_random_walks()
            num_walks, num_steps_plus_1 = walks.shape
            num_steps = num_steps_plus_1 - 1

            if num_walks == 0 or num_steps < 0:
                print(
                    "No random walks or steps generated. Skipping visualization."
                )
                return

            print(
                f"Generated {num_walks:,} walks with {num_steps:,} steps. Visualizing..."
            )

            walk_plot_path = (
                (DEFAULT_OUTPUT_DIR / self.v_cfg.DEFAULT_MARKOV_PLOT_FILENAME)
                if save
                else None
            )
            dist_plot_path = (
                (
                    DEFAULT_OUTPUT_DIR
                    / self.v_cfg.DEFAULT_DIST_PLOT_FILENAME
                )
                if save
                else None
            )

            self.visualizer.plot_random_walks(
                walks, show_plot=show, save_path=walk_plot_path
            )

            if num_steps > 0:
                self.visualizer.plot_distribution_snapshots(
                    walks, show_plot=show, save_path=dist_plot_path
                )
            else:
                print("Skipping distribution snapshots plot (0 steps).")

            if save:
                if walk_plot_path and walk_plot_path.exists():
                    print(
                        f"Random walk plot saved: {walk_plot_path.resolve()}"
                    )
                elif walk_plot_path:
                    print(
                        f"Random walk plot FAILED to save to: {walk_plot_path.resolve()}"
                    )

                if num_steps > 0:
                    if dist_plot_path and dist_plot_path.exists():
                        print(
                            f"Distribution plot saved: {dist_plot_path.resolve()}"
                        )
                    elif dist_plot_path:
                        print(
                            f"Distribution plot FAILED to save to: {dist_plot_path.resolve()}"
                        )

        return self._run_task(
            "Random Walk Simulation & Visualization", task, show_plots, save_plots
        )

    def setup_city_migration(self) -> bool:
        if self.city_migration is None:
            print("Setting up default City Migration simulation scenario...")
            try:
                self.city_migration = CityMigrationSimulation.create_default(
                    self.markov_sim
                )
                print("City Migration simulation initialized successfully.")
                return True
            except (
                ConfigError,
                ValueError,
                InvalidTransitionMatrixError,
            ) as e:
                print(
                    f"ERROR setting up City Migration simulation: {e}",
                    file=sys.stderr,
                )
                self.city_migration = None
                return False
        return True

    def run_city_migration_analysis(self) -> bool:
        def task():
            if self.city_migration is None:
                if not self.setup_city_migration():
                    raise SimulationError(
                        "City Migration simulation setup failed. Cannot run analysis."
                    )

            assert (
                self.city_migration is not None
            ), "City migration should be initialized here."
            self.city_migration.run_and_display()

        return self._run_task("City Migration Analysis", task)

    def run_all(
        self,
        run_galton: bool = True,
        run_random_walk: bool = True,
        run_city_migration: bool = True,
        show_plots: bool = True,
        save_outputs: bool = True,
    ) -> bool:
        max_width = 78
        title = "Integrated Stochastic Simulation Suite Run"
        print(
            f"\n{'*' * max_width}\n{title:^{max_width}}\n{'*' * max_width}"
        )
        overall_start_time = time.monotonic()
        task_results: list[bool] = []

        tasks_to_run = [
            (
                run_galton,
                self.run_galton_simulation,
                (save_outputs,),
            ),
            (
                run_random_walk,
                self.run_random_walk_simulation,
                (show_plots, save_outputs),
            ),
            (run_city_migration, self.run_city_migration_analysis, ()),
        ]

        for should_run, task_func, task_args in tasks_to_run:
            if should_run:
                task_results.append(task_func(*task_args))

        overall_elapsed_time = time.monotonic() - overall_start_time
        overall_success = all(task_results) if task_results else True

        print("\n--- Simulation Suite Summary ---")
        print(f"Total execution time: {overall_elapsed_time:.2f} seconds.")
        status_message = (
            "All selected tasks completed successfully"
            if overall_success
            else "One or more tasks FAILED"
        )
        print(f"Overall status: {status_message}")
        print("*" * max_width + "\n")

        return overall_success


def main_simulation_runner() -> int:
    plt.ioff()
    exit_code = 0
    runner: SimulationRunner | None = None

    try:
        print("Initializing Simulation Runner with default configurations...")
        runner = SimulationRunner()
        print("Initialization complete. Starting simulation suite execution...")
        success = runner.run_all(
            run_galton=True,
            run_random_walk=True,
            run_city_migration=True,
            show_plots=False,
            save_outputs=True,
        )
        exit_code = 0 if success else 1

    except ConfigError as e:
        print(
            f"\nCRITICAL CONFIGURATION ERROR: {e}\nAborting simulation.",
            file=sys.stderr,
        )
        exit_code = 2
    except Exception as e:
        print(
            f"\nCRITICAL UNHANDLED ERROR: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        exit_code = 3
    finally:
        plt.close("all")

    print(f"\nSimulation run finished. Exiting with code {exit_code}.")
    return exit_code


class TestSimulationSuite(unittest.TestCase):
    test_output_dir: ClassVar[Path]
    run_id: ClassVar[str]
    test_galton_config: ClassVar[GaltonConfig]
    test_markov_config: ClassVar[MarkovConfig]
    test_vis_config: ClassVar[VisConfig]

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_output_dir = DEFAULT_OUTPUT_DIR / "unit_tests"
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)
        cls.run_id = uuid.uuid4().hex[:8]
        print(
            f"\n[Test Setup] Using test directory: {cls.test_output_dir.resolve()}, Run ID: {cls.run_id}"
        )

        galton_fname = f"test_galton_{cls.run_id}.png"
        walk_fname = f"test_walks_{cls.run_id}.png"
        dist_fname = f"test_dist_{cls.run_id}.png"

        cls.test_galton_config = GaltonConfig(
            NUM_BALLS=100,
            NUM_ROWS=5,
            BOARD_WIDTH=50,
            BOARD_HEIGHT=40,
            SMOOTHING_WINDOW=1,
            DEFAULT_IMAGE_FILENAME=galton_fname,
        )
        cls.test_markov_config = MarkovConfig(
            RANDOM_WALK_NUM=10,
            RANDOM_WALK_STEPS=20,
            SEED=42,
            STEADY_STATE_ITERATIONS=500,
            STEADY_STATE_BURN_IN=50,
            ANALYSIS_METHODS=(
                "brute_force",
                "eigenvalue",
                "linear_system",
                "monte_carlo",
            ),
        )
        cls.test_vis_config = VisConfig(
            FIGSIZE=(4, 3),
            DPI=75,
            DEFAULT_MARKOV_PLOT_FILENAME=walk_fname,
            DEFAULT_DIST_PLOT_FILENAME=dist_fname,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        print(
            f"\n[Test Cleanup] Removing test output files for Run ID: {cls.run_id}..."
        )
        files_removed_count = 0
        try:
            if cls.test_output_dir.exists():
                for f in cls.test_output_dir.glob(f"*_{cls.run_id}.*"):
                    if f.is_file():
                        try:
                            f.unlink()
                            files_removed_count += 1
                        except OSError as e:
                            print(
                                f"[Test Cleanup] Warning: Could not remove file {f}: {e}",
                                file=sys.stderr,
                            )
                print(
                    f"[Test Cleanup] Removed {files_removed_count} test files."
                )
                try:
                    if not any(cls.test_output_dir.iterdir()):
                        cls.test_output_dir.rmdir()
                        print(
                            f"[Test Cleanup] Removed empty test directory: {cls.test_output_dir}"
                        )
                except OSError:
                    print(
                        f"[Test Cleanup] Info: Test directory {cls.test_output_dir} not removed (might not be empty).",
                        file=sys.stderr,
                    )
        except OSError as e:
            print(
                f"[Test Cleanup] Warning: Error during file/directory cleanup: {e}",
                file=sys.stderr,
            )

    def _get_test_file_path(self, filename: str) -> Path:
        return self.test_output_dir / filename

    def test_A01_default_configs_are_valid(self) -> None:
        try:
            g_cfg = GaltonConfig()
            m_cfg = MarkovConfig()
            v_cfg = VisConfig()
            self.assertIsInstance(g_cfg, GaltonConfig)
            self.assertIsInstance(m_cfg, MarkovConfig)
            self.assertIsInstance(v_cfg, VisConfig)
        except ConfigError as e:
            self.fail(
                f"Default configuration initialization failed unexpectedly: {e}"
            )

    def test_A02_invalid_config_parameters_raise_error(self) -> None:
        with self.assertRaisesRegex(
            ConfigError, "positive integer", msg="Negative NUM_BALLS"
        ):
            GaltonConfig(NUM_BALLS=-1)
        with self.assertRaisesRegex(
            ConfigError,
            "non-negative integer",
            msg="Negative RANDOM_WALK_NUM",
        ):
            MarkovConfig(RANDOM_WALK_NUM=-1)
        with self.assertRaisesRegex(
            ConfigError, "must be less than", msg="Burn-in >= Iterations"
        ):
            MarkovConfig(
                STEADY_STATE_BURN_IN=1000, STEADY_STATE_ITERATIONS=500
            )
        with self.assertRaisesRegex(
            ConfigError, "positive integer", msg="Zero FIGSIZE width"
        ):
            VisConfig(FIGSIZE=(0, 100))
        with self.assertRaisesRegex(
            ConfigError, "Invalid analysis methods", msg="Invalid method name"
        ):
            MarkovConfig(ANALYSIS_METHODS=("invalid_method",))
        with self.assertRaisesRegex(
            ConfigError,
            "At least one analysis method",
            msg="Empty methods tuple",
        ):
            MarkovConfig(ANALYSIS_METHODS=())
        with self.assertRaisesRegex(
            ConfigError, "positive integer", msg="Zero BOARD_WIDTH"
        ):
            GaltonConfig(BOARD_WIDTH=0)
        with self.assertRaisesRegex(
            ConfigError, "float between 0.0 and 1.0", msg="Damping > 1"
        ):
            GaltonConfig(DAMPING_FACTOR=1.1)
        with self.assertRaisesRegex(
            ConfigError,
            "must be a positive integer or a valid strategy",
            msg="Invalid bin string",
        ):
            VisConfig(DIST_HIST_BINS="invalid_bin_strategy")

    def test_B01_galton_board_physics_mode_simulation(self) -> None:
        config = dataclasses.replace(
            self.test_galton_config, MODE="physics_based"
        )
        board = GaltonBoard(config)
        board.set_rng_seed(0)
        board.simulate()
        self.assertEqual(
            len(board.slot_counts), config.BOARD_WIDTH, "Slot count length mismatch."
        )
        self.assertAlmostEqual(
            sum(board.slot_counts),
            config.NUM_BALLS,
            delta=1,
            msg="Ball count mismatch (physics).",
        )
        if config.NUM_BALLS > 0:
            self.assertTrue(
                sum(board.slot_counts) > 0,
                "Total count is zero after simulation.",
            )

    def test_B02_galton_board_simple_mode_simulation(self) -> None:
        config = dataclasses.replace(
            self.test_galton_config, MODE="simple_random_walk"
        )
        board = GaltonBoard(config)
        board.set_rng_seed(1)
        board.simulate()
        self.assertEqual(
            len(board.slot_counts), config.BOARD_WIDTH, "Slot count length mismatch."
        )
        self.assertEqual(
            sum(board.slot_counts),
            config.NUM_BALLS,
            "Ball count mismatch (simple).",
        )

    def test_B03_galton_board_image_saving(self) -> None:
        board = GaltonBoard(self.test_galton_config)
        board.simulate()
        output_file_path = self._get_test_file_path(
            board.config.DEFAULT_IMAGE_FILENAME
        )
        saved_path_str = ""
        try:
            saved_path_str = board.save_image(output_file_path)
            saved_path = Path(saved_path_str)
            self.assertTrue(
                saved_path.exists(), f"Image file was not created at {saved_path}"
            )
            self.assertEqual(
                saved_path.resolve(),
                output_file_path.resolve(),
                "Saved path mismatch.",
            )
            with Image.open(saved_path) as img:
                self.assertEqual(
                    img.size,
                    (board.config.BOARD_WIDTH, board.config.BOARD_HEIGHT),
                    "Image dimensions mismatch.",
                )
                self.assertEqual(img.mode, "RGB", "Image mode is not RGB.")
        except (IOError, VisualizationError) as e:
            self.fail(f"Galton board image saving failed with error: {e}")
        finally:
            if saved_path_str and Path(saved_path_str).exists():
                Path(saved_path_str).unlink()

    def test_B04_galton_board_smoothing_logic(self) -> None:
        config = dataclasses.replace(
            self.test_galton_config, SMOOTHING_WINDOW=3, BOARD_WIDTH=7
        )
        board = GaltonBoard(config)
        input_counts = np.array([0, 0, 10, 50, 10, 0, 0])
        board._apply_smoothing(input_counts)
        self.assertAlmostEqual(
            sum(board.slot_counts),
            sum(input_counts),
            delta=1,
            msg="Smoothing changed total count.",
        )
        self.assertTrue(
            board.slot_counts[1] > input_counts[1],
            "Smoothing did not affect left neighbor.",
        )
        self.assertTrue(
            board.slot_counts[5] > input_counts[5],
            "Smoothing did not affect right neighbor.",
        )
        self.assertTrue(
            board.slot_counts[3] < input_counts[3],
            "Smoothing did not reduce peak value.",
        )
        self.assertAlmostEqual(board.slot_counts[2], (0 + 10 + 50) / 3, delta=1)
        self.assertAlmostEqual(board.slot_counts[3], (10 + 50 + 10) / 3, delta=1)
        self.assertAlmostEqual(board.slot_counts[4], (50 + 10 + 0) / 3, delta=1)

    def test_C01_transition_matrix_validation_rules(self) -> None:
        try:
            TransitionMatrix(np.identity(2))
            tm_empty = TransitionMatrix(np.zeros((0, 0)))
            self.assertEqual(len(tm_empty), 0, "Empty matrix length incorrect.")
            TransitionMatrix([[0.1, 0.9], [0.5, 0.5]])
        except InvalidTransitionMatrixError as e:
            self.fail(
                f"Valid TransitionMatrix raised an unexpected error: {e}"
            )

        with self.assertRaisesRegex(
            InvalidTransitionMatrixError, "square", msg="Non-square matrix"
        ):
            TransitionMatrix([[0.1, 0.9], [0.2, 0.7, 0.1]])
        with self.assertRaisesRegex(
            InvalidTransitionMatrixError, "sum to 1", msg="Row sum != 1"
        ):
            TransitionMatrix([[0.1, 0.8], [0.8, 0.2]])
        with self.assertRaisesRegex(
            InvalidTransitionMatrixError, "negative", msg="Negative probability"
        ):
            TransitionMatrix([[-0.1, 1.1], [0.5, 0.5]])
        with self.assertRaisesRegex(
            InvalidTransitionMatrixError,
            "Invalid data type",
            msg="Non-numeric data",
        ):
            TransitionMatrix("this is not a matrix")

    def test_C02_random_walk_generation_output(self) -> None:
        simulator = MarkovChainSimulator(self.test_markov_config)
        walks = simulator.simulate_random_walks()
        expected_shape = (
            self.test_markov_config.RANDOM_WALK_NUM,
            self.test_markov_config.RANDOM_WALK_STEPS + 1,
        )
        self.assertEqual(
            walks.shape, expected_shape, "Generated walks array has incorrect shape."
        )
        self.assertTrue(
            np.all(walks[:, 0] == 0), "Walks do not start at position 0."
        )
        if walks.shape[0] > 0 and walks.shape[1] > 1:
            self.assertTrue(
                np.var(walks[:, -1]) > 0,
                "No variance in final positions, walks might not have moved.",
            )

    def test_C03_steady_state_computation_accuracy(self) -> None:
        tm_data = np.array([[0.8, 0.2], [0.3, 0.7]])
        transition_matrix = TransitionMatrix(tm_data)
        known_steady_state = np.array([0.6, 0.4])

        test_mc_config = dataclasses.replace(
            self.test_markov_config,
            STEADY_STATE_ITERATIONS=10000,
            STEADY_STATE_BURN_IN=1000,
            SEED=123,
        )
        simulator = MarkovChainSimulator(test_mc_config)
        results = simulator.analyze_steady_state(transition_matrix)

        self.assertEqual(
            set(results.keys()),
            set(test_mc_config.ANALYSIS_METHODS),
            "Not all analysis methods returned results.",
        )

        tolerances = {
            "brute_force": 1e-7,
            "eigenvalue": 1e-8,
            "linear_system": 1e-8,
            "monte_carlo": 5e-3,
        }

        for method in test_mc_config.ANALYSIS_METHODS:
            self.assertIn(
                method, results, f"Method '{method}' missing from results dictionary."
            )
            computed_vector = results[method]
            self.assertIsNotNone(
                computed_vector, f"Method '{method}' result is None."
            )
            self.assertFalse(
                np.isnan(computed_vector).any(),
                f"Method '{method}' result contains NaN values.",
            )
            self.assertEqual(
                computed_vector.shape,
                known_steady_state.shape,
                f"Method '{method}' result has incorrect shape.",
            )
            self.assertTrue(
                np.allclose(
                    computed_vector, known_steady_state, atol=tolerances[method]
                ),
                f"Method '{method}' failed accuracy test.\nExpected: {known_steady_state}\nGot:      {computed_vector}",
            )

    @unittest.mock.patch("matplotlib.pyplot.show")
    def test_D01_visualization_plot_saving_works(
        self, mock_plt_show
    ) -> None:
        simulator = MarkovChainSimulator(self.test_markov_config)
        visualizer = Visualizer(self.test_vis_config)
        walks = simulator.simulate_random_walks()

        walk_plot_path = self._get_test_file_path(
            self.test_vis_config.DEFAULT_MARKOV_PLOT_FILENAME
        )
        dist_plot_path = self._get_test_file_path(
            self.test_vis_config.DEFAULT_DIST_PLOT_FILENAME
        )

        try:
            visualizer.plot_random_walks(
                walks, show_plot=False, save_path=walk_plot_path
            )
            self.assertTrue(
                walk_plot_path.exists(),
                f"Walk plot file was not saved to {walk_plot_path}",
            )
        except VisualizationError as e:
            self.fail(
                f"plot_random_walks raised VisualizationError during saving: {e}"
            )
        finally:
            if walk_plot_path.exists():
                walk_plot_path.unlink()

        if self.test_markov_config.RANDOM_WALK_STEPS > 0:
            try:
                snapshot_times = [5, 10, 20]
                visualizer.plot_distribution_snapshots(
                    walks,
                    snapshot_times=snapshot_times,
                    show_plot=False,
                    save_path=dist_plot_path,
                )
                self.assertTrue(
                    dist_plot_path.exists(),
                    f"Distribution plot file was not saved to {dist_plot_path}",
                )
            except VisualizationError as e:
                self.fail(
                    f"plot_distribution_snapshots raised VisualizationError during saving: {e}"
                )
            finally:
                if dist_plot_path.exists():
                    dist_plot_path.unlink()
        else:
            self.skipTest(
                "Skipping distribution snapshot save test: RANDOM_WALK_STEPS is 0."
            )

    @unittest.mock.patch("matplotlib.pyplot.show")
    @unittest.mock.patch("sys.stdout")
    @unittest.mock.patch("sys.stderr")
    def test_D02_visualization_handles_empty_data_gracefully(
        self, mock_stderr, mock_stdout, mock_plt_show
    ) -> None:
        visualizer = Visualizer(self.test_vis_config)
        empty_walks_data = np.zeros(
            (0, self.test_markov_config.RANDOM_WALK_STEPS + 1)
        )
        zero_step_data = np.zeros(
            (self.test_markov_config.RANDOM_WALK_NUM, 1)
        )

        try:
            visualizer.plot_random_walks(
                empty_walks_data, show_plot=False, save_path=None
            )
            visualizer.plot_distribution_snapshots(
                empty_walks_data, show_plot=False, save_path=None
            )
            visualizer.plot_random_walks(
                zero_step_data, show_plot=False, save_path=None
            )
            visualizer.plot_distribution_snapshots(
                zero_step_data, show_plot=False, save_path=None
            )
        except Exception as e:
            self.fail(
                f"Visualization function failed unexpectedly with empty/zero-step data: {type(e).__name__}: {e}"
            )

    @unittest.mock.patch("matplotlib.pyplot.show")
    @unittest.mock.patch("sys.stdout")
    @unittest.mock.patch("sys.stderr")
    def test_E01_simulation_runner_full_execution(
        self, mock_stderr, mock_stdout, mock_plt_show
    ) -> None:
        runner = SimulationRunner(
            self.test_galton_config,
            self.test_markov_config,
            self.test_vis_config,
        )
        galton_path = self._get_test_file_path(
            self.test_galton_config.DEFAULT_IMAGE_FILENAME
        )
        walk_path = self._get_test_file_path(
            self.test_vis_config.DEFAULT_MARKOV_PLOT_FILENAME
        )
        dist_path = self._get_test_file_path(
            self.test_vis_config.DEFAULT_DIST_PLOT_FILENAME
        )

        success = runner.run_all(
            run_galton=True,
            run_random_walk=True,
            run_city_migration=True,
            show_plots=False,
            save_outputs=True,
        )

        self.assertTrue(
            success, "SimulationRunner.run_all reported failure, expected success."
        )
        self.assertTrue(
            galton_path.exists(), f"Galton output file missing: {galton_path}"
        )
        self.assertTrue(
            walk_path.exists(), f"Random walk plot file missing: {walk_path}"
        )
        if self.test_markov_config.RANDOM_WALK_STEPS > 0:
            self.assertTrue(
                dist_path.exists(),
                f"Distribution plot file missing: {dist_path}",
            )
        else:
             self.assertFalse(
                 dist_path.exists(),
                 f"Distribution plot file created unexpectedly for 0 steps: {dist_path}"
             )

        if galton_path.exists():
            galton_path.unlink()
        if walk_path.exists():
            walk_path.unlink()
        if dist_path.exists():
            dist_path.unlink()

    @unittest.mock.patch("sys.stdout")
    @unittest.mock.patch("sys.stderr")
    def test_E02_city_migration_setup_and_run_integration(
        self, mock_stderr, mock_stdout
    ) -> None:
        runner = SimulationRunner(markov_config=self.test_markov_config)

        setup_success = runner.setup_city_migration()
        self.assertTrue(
            setup_success, "Runner failed to set up City Migration simulation."
        )
        self.assertIsInstance(
            runner.city_migration,
            CityMigrationSimulation,
            "City Migration object not created in runner.",
        )
        self.assertEqual(
            runner.city_migration.city_names,
            CityMigrationSimulation.DEFAULT_CITIES,
            "Default city names mismatch.",
        )

        run_success = runner.run_city_migration_analysis()
        self.assertTrue(
            run_success, "Runner failed to run City Migration analysis."
        )


def run_tests(verbosity_level: int = 2) -> int:
    print("\n--- Running Unit Tests ---")
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    suite = loader.loadTestsFromTestCase(TestSimulationSuite)
    runner = unittest.TextTestRunner(
        verbosity=verbosity_level, failfast=False, buffer=True
    )
    result = runner.run(suite)

    print("\n--- Unit Tests Complete ---")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    return 0 if result.wasSuccessful() else 1


def display_help() -> None:
    try:
        script_name = Path(__file__).name
    except NameError:
        script_name = "mchs_prws_mntp.py"

    help_text = f"""
Usage: python {script_name} [options]

Stochastic Simulation Suite: Galton Board, Markov Random Walks, Steady State.

Options:
  --test [-v N] : Run the integrated unit test suite.
                  Optional verbosity level N can be 0 (quiet), 1 (default),
                  or 2 (verbose). Default is 2 if -v is omitted.
  --help, -h    : Display this help message and exit.
  (no options)  : Run the full simulation suite (Galton Board, Random Walks,
                  City Migration) with default parameters.

Description:
  This script simulates and visualizes several fundamental stochastic processes:
  - Galton Board: Demonstrates the Central Limit Theorem. Output: PNG image.
  - Markov Chain Random Walks: Simulates 1D walks, showing convergence to Normal.
    Output: PNG plots of paths and distributions.
  - Markov Chain Steady-State Analysis: Computes equilibrium probabilities using
    multiple methods (Power Iteration, Eigenvalue, Linear System, Monte Carlo).
    Output: Console table comparing results (e.g., for City Migration model).

Default Output Directory:
  Generated files are saved to: {DEFAULT_OUTPUT_DIR.resolve()}
"""
    print(help_text)


if __name__ == "__main__":
    exit_code: int = 0
    command_args = sys.argv[1:]

    if "--test" in command_args:
        test_verbosity = 2
        try:
            if "-v" in command_args:
                v_index = command_args.index("-v")
                if v_index + 1 < len(command_args):
                    level_str = command_args[v_index + 1]
                    if level_str.isdigit():
                        level = int(level_str)
                        if level in [0, 1, 2]:
                            test_verbosity = level
                        else:
                            print(
                                "Warning: Invalid verbosity level specified after -v. Must be 0, 1, or 2. Using default (2).",
                                file=sys.stderr,
                            )
                    else:
                        print(
                            "Warning: Non-integer verbosity level provided after -v. Using default (2).",
                            file=sys.stderr,
                        )
                else:
                    print(
                        "Warning: Missing verbosity level after -v argument. Using default (2).",
                        file=sys.stderr,
                    )
        except ValueError:
            print(
                "Warning: Error parsing verbosity argument '-v'. Using default verbosity (2).",
                file=sys.stderr,
            )
        exit_code = run_tests(verbosity_level=test_verbosity)

    elif "--help" in command_args or "-h" in command_args:
        display_help()
        exit_code = 0

    elif not command_args:
        exit_code = main_simulation_runner()

    else:
        print(
            f"Error: Unknown or invalid arguments provided: {' '.join(command_args)}",
            file=sys.stderr,
        )
        display_help()
        exit_code = 4

    sys.exit(exit_code)