from __future__ import annotations

import math
import os
import sys
import time
import unittest
import uuid
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
DEFAULT_MAX_WORKERS: Final[int] = 8
DEFAULT_SEED_FUNC: Callable[[], int] = (
    lambda: int(time.time() * 1_000_000) % (2**32)
)


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
    vec: NDArrayF64, n_states: int
) -> NDArrayF64:
    if n_states <= 0:
        return np.array([], dtype=np.float64)
    if np.isnan(vec).any():
        return vec

    vec = np.maximum(vec, 0.0)
    vec_sum = vec.sum()

    if abs(vec_sum) < PRECISION_TOLERANCE:
        return np.ones(n_states) / n_states

    if abs(vec_sum - 1.0) > 1e-6:
        vec = vec / vec_sum

    return np.clip(vec, 0.0, 1.0)


def _ensure_output_dir(file_path: Path) -> Path | None:
    resolved_path = file_path.resolve()
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        return resolved_path
    except OSError as e:
        print(
            f"Warning: Could not create/access directory "
            f"{resolved_path.parent}: {e}",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"Warning: Unexpected error checking/creating directory "
            f"{resolved_path.parent}: {e}",
            file=sys.stderr,
        )
        return None


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
        checks = {
            "positive integers": [
                ("NUM_ROWS", self.NUM_ROWS),
                ("NUM_BALLS", self.NUM_BALLS),
                ("BOARD_WIDTH", self.BOARD_WIDTH),
                ("BOARD_HEIGHT", self.BOARD_HEIGHT),
                ("PEG_RADIUS", self.PEG_RADIUS),
                ("HISTOGRAM_BAR_MIN_WIDTH", self.HISTOGRAM_BAR_MIN_WIDTH),
            ],
            "non-negative numbers": [
                ("INITIAL_VARIANCE", self.INITIAL_VARIANCE),
                ("SMOOTHING_WINDOW", self.SMOOTHING_WINDOW),
            ],
            "floats in [0, 1]": [
                ("DAMPING_FACTOR", self.DAMPING_FACTOR),
                ("ELASTICITY", self.ELASTICITY),
                ("MIN_BOUNCE_PROBABILITY", self.MIN_BOUNCE_PROBABILITY),
                ("MAX_BOUNCE_PROBABILITY", self.MAX_BOUNCE_PROBABILITY),
                ("BOUNCE_PROB_CENTER", self.BOUNCE_PROB_CENTER),
            ],
        }

        for name, val in checks["positive integers"]:
            if not isinstance(val, int) or val <= 0:
                raise ConfigError(f"{name} must be a positive integer.")
        for name, val in checks["non-negative numbers"]:
            if not isinstance(val, (int, float)) or val < 0:
                raise ConfigError(f"{name} must be non-negative.")
        for name, val in checks["floats in [0, 1]"]:
            if not isinstance(val, float) or not (0.0 <= val <= 1.0):
                raise ConfigError(f"{name} must be a float in [0, 1].")

        if self.MIN_BOUNCE_PROBABILITY > self.MAX_BOUNCE_PROBABILITY:
            raise ConfigError(
                "MIN_BOUNCE_PROBABILITY cannot exceed MAX_BOUNCE_PROBABILITY."
            )
        if self.BOARD_WIDTH <= 0:
            raise ConfigError("BOARD_WIDTH must be positive.")


@dataclass(frozen=True)
class MarkovConfig:
    RANDOM_WALK_NUM: int = 500
    RANDOM_WALK_STEPS: int = 150
    STEADY_STATE_ITERATIONS: int = 1_000
    STEADY_STATE_BURN_IN: int = 100
    ANALYSIS_METHODS: tuple[MarkovMethod, ...] = field(
        default=("brute_force", "eigenvalue", "linear_system", "monte_carlo")
    )
    SEED: int | None = field(default_factory=DEFAULT_SEED_FUNC)

    def __post_init__(self) -> None:
        if self.RANDOM_WALK_NUM < 0:
            raise ConfigError("RANDOM_WALK_NUM cannot be negative.")
        if self.RANDOM_WALK_STEPS < 0:
            raise ConfigError("RANDOM_WALK_STEPS cannot be negative.")
        if self.STEADY_STATE_ITERATIONS <= 0:
            raise ConfigError("STEADY_STATE_ITERATIONS must be positive.")
        if self.STEADY_STATE_BURN_IN < 0:
            raise ConfigError("STEADY_STATE_BURN_IN cannot be negative.")
        if self.STEADY_STATE_BURN_IN >= self.STEADY_STATE_ITERATIONS:
            raise ConfigError(
                "STEADY_STATE_BURN_IN must be less than STEADY_STATE_ITERATIONS."
            )

        valid_methods = {
            "brute_force", "eigenvalue", "linear_system", "monte_carlo"
        }
        invalid_methods = set(self.ANALYSIS_METHODS) - valid_methods
        if invalid_methods:
            raise ConfigError(
                f"Invalid analysis methods: {invalid_methods}. "
                f"Valid methods are: {valid_methods}"
            )
        if not self.ANALYSIS_METHODS:
            raise ConfigError("At least one analysis method must be specified.")


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
            "markov_plots", "png"
        )
    )
    DEFAULT_DIST_PLOT_FILENAME: Final[str] = field(
        default_factory=lambda: _create_unique_filename(
            "distribution_plots", "png"
        )
    )

    def __post_init__(self) -> None:
        dims = (
            *self.FIGSIZE, *self.DIST_SNAPSHOT_FIGSIZE, self.DPI, self.DIST_X_POINTS
        )
        alphas = (
            self.RANDOM_WALK_ALPHA, self.DIST_GRID_ALPHA, self.DIST_HIST_ALPHA
        )

        if any(x <= 0 for x in dims):
            raise ConfigError("Figure dimensions, DPI, DIST_X_POINTS must be positive.")
        if any(not (0.0 < alpha <= 1.0) for alpha in alphas):
            raise ConfigError("Alpha values must be in the range (0, 1].")

        is_int_bins = isinstance(self.DIST_HIST_BINS, int)
        is_auto_bins = self.DIST_HIST_BINS == "auto"
        if not (is_int_bins or is_auto_bins):
            raise ConfigError("DIST_HIST_BINS must be a positive integer or 'auto'.")
        if is_int_bins and self.DIST_HIST_BINS <= 0:
            raise ConfigError("Integer DIST_HIST_BINS must be positive.")


@runtime_checkable
class SteadyStateComputable(Protocol):
    def compute(
        self, matrix: NDArrayF64, init_state: NDArrayF64 | None = None
    ) -> NDArrayF64: ...


class SteadyStateComputer(ABC):
    @staticmethod
    def _validate_square_matrix(matrix: NDArrayF64) -> int:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise InvalidTransitionMatrixError(
                f"Matrix must be square, but has shape {matrix.shape}."
            )
        return matrix.shape[0]

    @abstractmethod
    def compute(
        self, matrix: NDArrayF64, init_state: NDArrayF64 | None = None
    ) -> NDArrayF64: ...


class BruteForceComputer(SteadyStateComputer):
    def __init__(self, iterations: int) -> None:
        if iterations <= 0:
            raise ValueError("Iterations must be positive.")
        self._iterations: Final = iterations

    def compute(
        self, matrix: NDArrayF64, init_state: NDArrayF64 | None = None
    ) -> NDArrayF64:
        dim = self._validate_square_matrix(matrix)
        if dim == 0:
            return np.array([], dtype=np.float64)

        if init_state is None:
            state = np.ones(dim) / dim
        else:
            if init_state.shape != (dim,):
                raise SimulationError(
                    f"Initial state shape {init_state.shape} mismatch "
                    f"with matrix dimension {dim}."
                )
            state = _normalize_probability_vector(init_state.copy(), dim)

        try:
            current_state = state
            for _ in range(self._iterations):
                current_state = current_state @ matrix
            return _normalize_probability_vector(current_state, dim)
        except Exception as e:
            raise SimulationError(
                "Brute force steady-state computation failed."
            ) from e


class EigenvalueComputer(SteadyStateComputer):
    def compute(
        self, matrix: NDArrayF64, _: NDArrayF64 | None = None
    ) -> NDArrayF64:
        dim = self._validate_square_matrix(matrix)
        if dim == 0:
            return np.array([], dtype=np.float64)

        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
            one_indices = np.isclose(eigenvalues, 1.0)

            if not np.any(one_indices):
                index_closest_to_one = np.argmin(np.abs(eigenvalues - 1.0))
                if not np.isclose(eigenvalues[index_closest_to_one], 1.0, atol=1e-5):
                    print(
                        f"Warning: Eigenvalue closest to 1 is "
                        f"{eigenvalues[index_closest_to_one]:.5f}. "
                        "Result might be approximate.", file=sys.stderr
                    )
            else:
                index_closest_to_one = np.where(one_indices)[0][0]

            steady_state_vector_t = eigenvectors[:, index_closest_to_one]
            steady_state_vector = np.real(steady_state_vector_t).flatten()
            return _normalize_probability_vector(steady_state_vector, dim)
        except np.linalg.LinAlgError as e:
            raise SimulationError("Eigenvalue computation failed.") from e
        except Exception as e:
            raise SimulationError(
                "Unexpected error during eigenvalue computation."
            ) from e


class LinearSystemComputer(SteadyStateComputer):
    def compute(
        self, matrix: NDArrayF64, _: NDArrayF64 | None = None
    ) -> NDArrayF64:
        n = self._validate_square_matrix(matrix)
        if n == 0:
            return np.array([], dtype=np.float64)

        try:
            a_system = matrix.T - np.identity(n)
            a_system[-1, :] = 1.0
            b_target = np.zeros(n)
            b_target[-1] = 1.0
            solution, residuals, _, _ = np.linalg.lstsq(
                a_system, b_target, rcond=None
            )

            if residuals.size > 0 and residuals[0] > 1e-6:
                print(
                    f"Warning: Linear system solution yielded "
                    f"residuals ({residuals[0]:.2e}).",
                    file=sys.stderr,
                )
            return _normalize_probability_vector(solution.flatten(), n)
        except np.linalg.LinAlgError as e:
            raise SimulationError("Linear system solution failed.") from e
        except Exception as e:
            raise SimulationError(
                "Unexpected error during linear system solution."
            ) from e


class MonteCarloComputer(SteadyStateComputer):
    def __init__(
        self, iterations: int, burn_in: int, rng: np.random.Generator
    ) -> None:
        if iterations <= 0:
            raise ValueError("Iterations must be positive.")
        if burn_in < 0:
            raise ValueError("Burn-in cannot be negative.")
        if burn_in >= iterations:
            raise ValueError("Burn-in must be less than iterations.")
        self._iterations: Final = iterations
        self._burn_in: Final = burn_in
        self._rng: Final = rng

    def compute(
        self, matrix: NDArrayF64, _: NDArrayF64 | None = None
    ) -> NDArrayF64:
        n = self._validate_square_matrix(matrix)
        if n == 0:
            return np.array([], dtype=np.float64)

        num_samples = self._iterations - self._burn_in
        if num_samples <= 0:
            return np.ones(n) / n if n > 0 else np.array([], dtype=float)

        try:
            current_state = self._rng.integers(0, n)
            state_counts = np.zeros(n, dtype=np.int64)

            for i in range(self._iterations):
                transition_probs = matrix[current_state]
                normalized_probs = _normalize_probability_vector(
                    transition_probs, n
                )

                if not np.any(normalized_probs > 0):
                    next_state = self._rng.integers(0, n)
                else:
                    next_state = self._rng.choice(n, p=normalized_probs)

                if i >= self._burn_in:
                    state_counts[current_state] += 1
                current_state = next_state

            steady_state_estimate = state_counts / num_samples
            return _normalize_probability_vector(steady_state_estimate, n)
        except Exception as e:
            raise SimulationError(
                "Monte Carlo steady-state computation failed."
            ) from e


class TransitionMatrix:
    _IVME = InvalidTransitionMatrixError

    def __init__(self, matrix_data: Any) -> None:
        try:
            matrix = np.array(matrix_data, dtype=np.float64)
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
                    "Transition matrix cannot contain negative probabilities."
                )
            matrix = np.maximum(matrix, 0.0)

            row_sums = matrix.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-8):
                problem_rows = np.where(
                    ~np.isclose(row_sums, 1.0, atol=1e-8)
                )[0]
                raise self._IVME(
                    f"Rows must sum to 1. Problem rows: {problem_rows} "
                    f"with sums: {row_sums[problem_rows]}"
                )
        self._matrix: Final[NDArrayF64] = matrix

    @property
    def matrix(self) -> NDArrayF64:
        return self._matrix

    @property
    def num_states(self) -> int:
        return self._matrix.shape[0]

    def __getitem__(self, index: int) -> NDArrayF64:
        return self._matrix[index]

    def __len__(self) -> int:
        return self.num_states


class MarkovChainSimulator:
    _SE = SimulationError

    def __init__(self, config: MarkovConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.SEED)
        self._computers = self._initialize_computers(config)

    def _initialize_computers(
        self, cfg: MarkovConfig
    ) -> dict[MarkovMethod, SteadyStateComputable]:
        computers: dict[MarkovMethod, SteadyStateComputable] = {}
        computer_class_map: dict[MarkovMethod, type[SteadyStateComputer]] = {
            "brute_force": BruteForceComputer,
            "eigenvalue": EigenvalueComputer,
            "linear_system": LinearSystemComputer,
            "monte_carlo": MonteCarloComputer,
        }

        for method_name in cfg.ANALYSIS_METHODS:
            ComputerClass = computer_class_map[method_name]
            try:
                if method_name == "brute_force":
                    computers[method_name] = ComputerClass(
                        cfg.STEADY_STATE_ITERATIONS
                    )
                elif method_name == "monte_carlo":
                    computers[method_name] = ComputerClass(
                        cfg.STEADY_STATE_ITERATIONS,
                        cfg.STEADY_STATE_BURN_IN,
                        self._rng,
                    )
                else:
                    computers[method_name] = ComputerClass()
            except ValueError as e:
                raise ConfigError(
                    f"Error initializing '{method_name}' computer: {e}"
                ) from e
        return computers

    def simulate_random_walks(self) -> NDArrayF64:
        num_walks = self.config.RANDOM_WALK_NUM
        num_steps = self.config.RANDOM_WALK_STEPS

        if num_walks <= 0 or num_steps < 0:
            return np.zeros(
                (max(0, num_walks), max(0, num_steps + 1)), dtype=np.float64
            )

        try:
            steps = self._rng.choice(
                [-1, 1], size=(num_walks, num_steps), p=[0.5, 0.5]
            )
            positions = np.empty(
                (num_walks, num_steps + 1), dtype=np.float64
            )
            positions[:, 0] = 0.0
            np.cumsum(steps, axis=1, out=positions[:, 1:])
            return positions
        except (ValueError, MemoryError) as e:
            raise self._SE(
                f"Failed to generate random walk array: {e}"
            ) from e
        except Exception as e:
            raise self._SE(
                f"Unexpected error during random walk simulation: {e}"
            ) from e

    def analyze_steady_state(
        self, tm: TransitionMatrix, init_state: NDArrayF64 | None = None
    ) -> dict[MarkovMethod, NDArrayF64]:
        matrix = tm.matrix
        num_states = tm.num_states
        methods_to_run = list(self._computers.keys())

        if num_states == 0 or not methods_to_run:
            return {
                method: np.array([], dtype=np.float64)
                for method in methods_to_run
            }

        norm_init_state: NDArrayF64 | None = None
        if init_state is not None:
            shape = init_state.shape
            size = init_state.size
            if (shape == (num_states,) or
                    (init_state.ndim == 1 and size == num_states) or
                    (size == num_states and (shape[0] == 1 or shape[1] == 1))):
                norm_init_state = _normalize_probability_vector(
                    init_state.flatten(), num_states
                )
            else:
                 raise ValueError(
                    f"Initial state shape {shape} incompatible "
                    f"with {num_states} states."
                 )

        results: dict[MarkovMethod, NDArrayF64] = {}
        max_workers = min(
            len(methods_to_run), os.cpu_count() or 1, DEFAULT_MAX_WORKERS
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_map = {
                executor.submit(
                    self._computers[method].compute,
                    matrix,
                    norm_init_state if method == "brute_force" else None,
                ): method
                for method in methods_to_run
            }

            for future in as_completed(futures_map):
                method = futures_map[future]
                try:
                    result_vector = future.result()
                    if not isinstance(result_vector, np.ndarray) or \
                       result_vector.shape != (num_states,) or \
                       np.any(~np.isfinite(result_vector)):
                        raise SimulationError(
                           f"Method '{method}' returned invalid result shape "
                           f"or non-finite values."
                        )
                    results[method] = _normalize_probability_vector(
                        result_vector, num_states
                    )
                except (
                    SimulationError, InvalidTransitionMatrixError, Exception
                ) as e:
                    print(
                        f"Warning: Steady-state computation failed for method "
                        f"'{method}': {e}", file=sys.stderr,
                    )
                    results[method] = np.full(
                        num_states, np.nan, dtype=np.float64
                    )
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
    @lru_cache(maxsize=None) # Use configured size conceptually, applied via config
    def _bounce_prob_calculator(
        normalized_distance: float,
        center_prob: float,
        factor: float,
        min_prob: float,
        max_prob: float,
    ) -> float:
        probability = center_prob + factor * normalized_distance
        return np.clip(probability, min_prob, max_prob)

    def __post_init__(self) -> None:
        GaltonBoard._bounce_prob_calculator.cache_clear() # Reset cache for potential size change
        # This static call to cache_clear() affects all instances if size is truly static.
        # A better approach might involve instance-specific caching if needed,
        # but lru_cache on static/class methods is shared.
        # Re-setting cache size requires more complex metaprogramming or wrappers.
        # For now, rely on the default or globally set cache size.
        # Let's assume BOUNCE_PROB_CACHE_SIZE is effectively a global hint.
        # If the cache size from config needs dynamic application *per instance*,
        # the static method cache won't work directly like this.

        self.set_rng_seed(DEFAULT_SEED_FUNC())
        if self.config.BOARD_WIDTH <= 0:
             raise ConfigError("GaltonConfig BOARD_WIDTH must be positive.")
        self.slot_counts = [0] * self.config.BOARD_WIDTH
        self._board_center = self.config.BOARD_WIDTH / 2.0
        peg_radius = float(self.config.PEG_RADIUS)
        board_width = float(self.config.BOARD_WIDTH)
        is_physics = self.config.MODE == "physics_based"
        min_bound = peg_radius if is_physics else 0.0
        max_bound = (
             max(min_bound, board_width - peg_radius) if is_physics
             else max(0.0, board_width - 1.0)
        )
        if min_bound > max_bound:
            raise ConfigError("Invalid Galton bounds calculated (min > max).")
        self._horizontal_bounds = (min_bound, max_bound)

    def set_rng_seed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def simulate(self) -> None:
        num_balls = self.config.NUM_BALLS
        board_width = self.config.BOARD_WIDTH
        self.slot_counts = [0] * board_width

        if num_balls <= 0 or board_width <= 0:
            self._invalidate_image_cache()
            return

        if self.config.MODE == "physics_based":
            path_generator = self._generate_physics_paths()
        else:
            path_generator = self._generate_simple_paths()

        try:
            final_positions = np.fromiter(
                path_generator, dtype=float, count=num_balls
            )
            slot_indices = np.clip(
                np.round(final_positions).astype(np.intp),
                0,
                max(0, board_width - 1),
            )
            raw_counts = np.bincount(slot_indices, minlength=board_width)
            self._apply_smoothing(raw_counts)
        except Exception as e:
            raise SimulationError(
                f"Error during Galton simulation or binning: {e}"
            ) from e
        self._invalidate_image_cache()

    def _generate_physics_paths(self) -> Iterator[Position]:
        cfg = self.config
        if cfg.PEG_RADIUS <= PRECISION_TOLERANCE:
            yield from self._generate_simple_paths()
            return

        peg_radius = float(cfg.PEG_RADIUS)
        peg_diameter = 2.0 * peg_radius
        inv_peg_radius = 1.0 / peg_radius
        bounce_prob_func = self._bounce_prob_calculator # Use static method
        gaussian_noise = self._rng.normal
        uniform_random = self._rng.random
        min_prob, max_prob = cfg.MIN_BOUNCE_PROBABILITY, cfg.MAX_BOUNCE_PROBABILITY
        bounce_center = cfg.BOUNCE_PROB_CENTER
        bounce_factor = cfg.BOUNCE_DISTANCE_FACTOR
        damping, elasticity = cfg.DAMPING_FACTOR, cfg.ELASTICITY
        num_rows = cfg.NUM_ROWS
        initial_variance = cfg.INITIAL_VARIANCE
        board_center = self._board_center

        for _ in range(cfg.NUM_BALLS):
            current_pos = self._constrain(
                board_center + gaussian_noise(0, initial_variance)
            )
            momentum = 0.0
            row_parity = 0

            for _ in range(num_rows):
                row_offset = row_parity * peg_radius
                peg_center_x = (
                    round((current_pos - row_offset) / peg_diameter)
                    * peg_diameter + row_offset
                )
                delta = np.clip(
                    (current_pos - peg_center_x) * inv_peg_radius, -1.0, 1.0
                )
                prob_right = bounce_prob_func(
                    delta, bounce_center, bounce_factor, min_prob, max_prob
                )
                direction = 1 if uniform_random() < prob_right else -1
                impact_force = (1.0 - abs(delta)) * elasticity
                momentum = momentum * damping + direction * impact_force * peg_diameter
                current_pos = self._constrain(current_pos + momentum)
                row_parity ^= 1

            yield current_pos

    def _constrain(self, position: Position) -> Position:
        return np.clip(
            position, self._horizontal_bounds[0], self._horizontal_bounds[1]
        )

    def _generate_simple_paths(self) -> Iterator[Position]:
        num_rows = self.config.NUM_ROWS
        board_width = self.config.BOARD_WIDTH
        num_balls = self.config.NUM_BALLS
        start_pos = board_width // 2
        max_index = max(0, board_width - 1)

        if num_rows == 0:
            yield from np.full(num_balls, float(start_pos))
            return

        net_shifts = self._rng.choice(
            [-1, 1], size=(num_balls, num_rows), p=[0.5, 0.5]
        ).sum(axis=1)
        final_positions = np.clip(start_pos + net_shifts, 0, max_index)
        yield from final_positions.astype(float)

    def _apply_smoothing(
        self, raw_counts: NDArrayInt | Sequence[Frequency]
    ) -> None:
        window_size = self.config.SMOOTHING_WINDOW
        num_slots = len(raw_counts)

        if window_size <= 1 or num_slots == 0:
            self.slot_counts = list(raw_counts)
            return

        kernel = np.ones(window_size) / window_size
        data = np.asarray(raw_counts, dtype=np.float64)
        smoothed_counts = np.convolve(data, kernel, mode="same")
        self.slot_counts = [
            int(round(x)) for x in np.maximum(0, smoothed_counts)
        ]

    def _invalidate_image_cache(self) -> None:
        self._image = None
        self._draw = None

    def _prepare_image_context(self) -> None:
        if self._image is None or self._draw is None:
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
                    f"Failed to initialize image context: {e}"
                ) from e

    def generate_image(self) -> Image.Image:
        self._prepare_image_context()
        assert self._image is not None, "Image context not initialized"
        assert self._draw is not None, "Draw context not initialized"

        counts = self.slot_counts
        cfg = self.config
        max_freq = max(counts, default=0)

        if not counts or max_freq <= 0:
            return self._image

        num_slots = len(counts)
        bar_pixel_width = max(
            cfg.HISTOGRAM_BAR_MIN_WIDTH,
            cfg.BOARD_WIDTH // num_slots if num_slots > 0 else cfg.BOARD_WIDTH,
        )
        board_height_f = float(cfg.BOARD_HEIGHT)
        board_center_f = self._board_center

        try:
            for i, freq in enumerate(counts):
                if freq <= 0:
                    continue

                bar_height = max(1, int(round((freq / max_freq) * board_height_f)))
                x0 = i * bar_pixel_width
                y0 = cfg.BOARD_HEIGHT - bar_height
                x1 = x0 + bar_pixel_width
                y1 = cfg.BOARD_HEIGHT
                bar_center_x = x0 + bar_pixel_width / 2.0
                color = (
                    cfg.LEFT_COLOR if bar_center_x < board_center_f
                    else cfg.RIGHT_COLOR
                )
                self._draw.rectangle([x0, y0, x1, y1], fill=color, outline=None)
        except Exception as e:
            raise VisualizationError(
                f"Failed to draw histogram bars: {e}"
            ) from e

        return self._image

    def save_image(self, filename: str | Path | None = None) -> str:
        output_path = Path(filename or self.config.DEFAULT_IMAGE_FILENAME)
        resolved_path = _ensure_output_dir(output_path)

        if resolved_path is None:
            raise IOError(
                f"Invalid output path or failed directory creation for '{output_path}'."
            )

        try:
            img = self.generate_image()
            img.save(resolved_path)
            return str(resolved_path)
        except (OSError, IOError, VisualizationError, Exception) as e:
            raise IOError(
                f"Failed to save Galton image to '{resolved_path}': {e}"
            ) from e


class Visualizer:
    def __init__(self, config: VisConfig) -> None:
        self.config = config

    def _save_or_show(
        self,
        fig: matplotlib.figure.Figure,
        show: bool,
        filepath: Path | None,
    ) -> None:
        if filepath:
            target_path = _ensure_output_dir(filepath)
            if target_path:
                try:
                    fig.savefig(
                        target_path, dpi=self.config.DPI, bbox_inches="tight"
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to save plot to {target_path}: {e}",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"Warning: Could not ensure output directory for "
                    f"{filepath}. Plot not saved.", file=sys.stderr
                )

        if show:
            try:
                plt.show()
            except Exception as e:
                print(
                    f"Warning: Failed to show plot interactively: {e}",
                    file=sys.stderr
                )

    def plot_random_walks(
        self, walks: NDArrayF64, show: bool = True, path: Path | None = None
    ) -> None:
        num_walks, num_steps_p1 = walks.shape
        cfg = self.config
        fig: matplotlib.figure.Figure | None = None

        if num_walks == 0:
            print("Info: No random walks data provided to plot.")
            return

        try:
            fig, ax = plt.subplots(figsize=cfg.FIGSIZE)
            steps_axis = np.arange(num_steps_p1)

            ax.plot(
                steps_axis,
                walks.T,
                alpha=cfg.RANDOM_WALK_ALPHA,
                linewidth=cfg.RANDOM_WALK_LINEWIDTH,
            )

            title = f"{num_walks} Random Walk{'s' if num_walks != 1 else ''}"
            ax.set(title=title, xlabel="Step Number", ylabel="Position")
            ax.grid(True, alpha=cfg.DIST_GRID_ALPHA, linestyle=":")
            ax.margins(x=0.02)

            self._save_or_show(fig, show, path)
        except Exception as e:
            raise VisualizationError(f"Failed to plot random walks: {e}") from e
        finally:
            if fig is not None:
                plt.close(fig)

    def plot_distribution_snapshots(
        self,
        walks: NDArrayF64,
        times: list[int] | None = None,
        show: bool = True,
        path: Path | None = None,
    ) -> None:
        num_walks, num_steps_p1 = walks.shape
        max_step = num_steps_p1 - 1
        cfg = self.config
        fig: matplotlib.figure.Figure | None = None

        if num_walks == 0 or max_step <= 0:
            print("Info: No walk data or insufficient steps for snapshots.")
            return

        valid_times = self._get_valid_times(times, max_step)
        if not valid_times:
            print("Info: No valid time steps selected for snapshots.")
            return

        num_plots = len(valid_times)
        try:
            fig_w, fig_h = cfg.DIST_SNAPSHOT_FIGSIZE
            adjusted_width = max(fig_w, fig_w * num_plots / 4)
            fig, axes = plt.subplots(
                1, num_plots, figsize=(adjusted_width, fig_h),
                sharex=True, sharey=True
            )
            axes_list = np.atleast_1d(axes)

            plot_range, x_pdf_pts = self._get_plot_range_and_pdf_points(
                walks, valid_times
            )

            for i, t in enumerate(valid_times):
                positions_at_t = walks[:, t]
                current_ax = axes_list[i]
                is_first_plot = (i == 0)
                self._plot_single_snapshot(
                    current_ax, positions_at_t, t, plot_range, x_pdf_pts, is_first_plot
                )

            fig.suptitle("Distribution of Random Walkers Over Time", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._save_or_show(fig, show, path)
        except Exception as e:
            raise VisualizationError(
                f"Failed to plot distribution snapshots: {e}"
            ) from e
        finally:
            if fig is not None:
                plt.close(fig)

    def _get_valid_times(
        self, requested_times: list[int] | None, max_step: int
    ) -> list[int]:
        if max_step <= 0: return []

        if requested_times:
            times_to_use = sorted([t for t in requested_times if 0 < t <= max_step])
        else:
            num_snapshots = min(5, max_step + 1)
            log_start = np.log10(1) if max_step < 10 else np.log10(max(1, max_step // 10))
            log_end = np.log10(max(1, max_step))

            if log_start > log_end + PRECISION_TOLERANCE:
                 log_indices = np.array([max_step], dtype=int)
            else:
                log_indices = np.logspace(
                    log_start, log_end, num=num_snapshots, dtype=int, endpoint=True
                )

            times_to_use = sorted(list(set(t for t in log_indices if t > 0)))
            if max_step not in times_to_use and max_step > 0:
                 times_to_use.append(max_step)
                 times_to_use = sorted(list(set(times_to_use)))

        max_plots = 10
        if len(times_to_use) > max_plots:
            indices = np.linspace(0, len(times_to_use) - 1, max_plots, dtype=int)
            times_to_use = [times_to_use[i] for i in indices]

        return times_to_use


    def _get_plot_range_and_pdf_points(
        self, walks: NDArrayF64, times: list[int]
    ) -> Tuple[tuple[float, float], NDArrayF64]:
        cfg = self.config
        if not times:
            x_range = cfg.DIST_X_RANGE
            x_pdf_points = np.linspace(x_range[0], x_range[1], cfg.DIST_X_POINTS)
            return x_range, x_pdf_points

        positions = walks[:, times]

        if positions.size == 0:
            x_range = cfg.DIST_X_RANGE
        else:
            q_low, q_high = np.percentile(positions, [1, 99])
            padding = max((q_high - q_low) * 0.1, 1.0)
            data_min = q_low - padding
            data_max = q_high + padding
            x_min = min(data_min, cfg.DIST_X_RANGE[0])
            x_max = max(data_max, cfg.DIST_X_RANGE[1])
            x_range = (x_min, x_max)

        x_pdf_points = np.linspace(
            x_range[0], x_range[1], cfg.DIST_X_POINTS
        )
        return x_range, x_pdf_points

    def _plot_single_snapshot(
        self,
        ax: plt.Axes,
        positions: NDArrayF64,
        time_step: int,
        x_range: tuple[float, float],
        x_pdf_points: NDArrayF64,
        is_first: bool = False
    ) -> None:
        cfg = self.config
        ax.hist(
            positions, bins=cfg.DIST_HIST_BINS, density=True,
            alpha=cfg.DIST_HIST_ALPHA, color="skyblue",
            label=f"Data (t={time_step})",
        )

        std_dev = math.sqrt(float(time_step))

        if std_dev > PRECISION_TOLERANCE:
            pdf = stats.norm.pdf(x_pdf_points, loc=0, scale=std_dev)
            ax.plot(
                x_pdf_points, pdf, "r-",
                linewidth=cfg.RANDOM_WALK_LINEWIDTH * 1.5,
                label=f"Theory N(0, {std_dev**2:.1f})"
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
        if is_first:
            ax.set_ylabel("Probability Density")


class CityMigrationSimulation:
    DEFAULT_CITIES: Final[list[str]] = ["Raleigh", "Chapel Hill", "Durham"]
    DEFAULT_TM_DATA: Final[NDArrayF64] = np.array(
        [[0.90, 0.05, 0.05], [0.10, 0.80, 0.10], [0.04, 0.01, 0.95]]
    )
    DEFAULT_INITIAL_POPULATION: Final[NDArrayF64] = np.array(
        [300 * KILO, 300 * KILO, 300 * KILO], dtype=float
    )

    def __init__(
        self,
        transition_matrix: TransitionMatrix,
        initial_population: NDArrayF64,
        city_names: list[str],
        markov_simulator: MarkovChainSimulator,
    ) -> None:
        num_states = transition_matrix.num_states
        pop_flat = initial_population.flatten()

        if len(city_names) != num_states:
            raise ValueError(
                f"Number of cities ({len(city_names)}) must match "
                f"number of states ({num_states})."
            )
        if len(pop_flat) != num_states:
            raise ValueError(
                f"Initial population size ({len(pop_flat)}) must match "
                f"number of states ({num_states})."
            )
        if np.any(pop_flat < 0):
            raise ValueError("Initial populations cannot be negative.")

        self.transition_matrix = transition_matrix
        self.city_names = city_names
        self.simulator = markov_simulator
        self.total_population = max(0.0, pop_flat.sum())

        if self.total_population > PRECISION_TOLERANCE:
            self.initial_distribution = _normalize_probability_vector(
                pop_flat, num_states
            )
        else:
            self.initial_distribution = (
                np.ones(num_states) / num_states
                if num_states > 0
                else np.array([], dtype=float)
            )

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
            initial_population=cls.DEFAULT_INITIAL_POPULATION,
            city_names=cls.DEFAULT_CITIES,
            markov_simulator=simulator,
        )

    def run_and_display(self) -> dict[MarkovMethod, NDArrayF64]:
        print("\n--- Starting City Migration Analysis ---")
        try:
            steady_state_results = self.simulator.analyze_steady_state(
                self.transition_matrix, self.initial_distribution
            )
            self._display_results(steady_state_results)
            print("--- City Migration Analysis Complete ---")
            return steady_state_results
        except SimulationError as e:
            print(
                f"ERROR: City migration simulation failed: {e}", file=sys.stderr
            )
            raise SimulationError("City migration run failed.") from e
        except Exception as e:
            print(
                f"ERROR: Unexpected error during city migration: {e}",
                file=sys.stderr
            )
            raise SimulationError("Unexpected city migration error.") from e

    def _display_results(self, results: dict[MarkovMethod, NDArrayF64]) -> None:
        valid_methods = sorted([
            method for method, vector in results.items()
            if vector.size > 0 and not np.isnan(vector).any()
        ])

        max_width = 78
        title = f"{'City Migration Steady-State Population Estimates':^{max_width}}"
        separator = "=" * max_width

        print("\n" + separator)
        print(title)
        print(separator)

        if not valid_methods:
            print("\nNo valid steady-state results were computed.")
            print(separator + "\n")
            return

        city_col_width = max(15, max((len(c) for c in self.city_names), default=0) + 2)
        num_data_cols = len(valid_methods)
        data_col_width = max(
            18, (max_width - city_col_width) // num_data_cols if num_data_cols > 0 else 18
        )

        header_parts = [f"{'City':<{city_col_width}}"]
        header_parts.extend([
            f"{method.replace('_', ' ').title():>{data_col_width}}"
            for method in valid_methods
        ])
        header_line = "".join(header_parts)
        table_separator = "-" * len(header_line)

        print(f"\n{header_line}")
        print(table_separator)

        scaled_results = {
            method: results[method] * self.total_population
            for method in valid_methods
        }
        for i, city_name in enumerate(self.city_names):
            row_parts = [f"{city_name:<{city_col_width}}"]
            row_parts.extend([
                f"{scaled_results[method][i]:>{data_col_width},.0f}"
                for method in valid_methods
            ])
            print("".join(row_parts))

        print(table_separator)
        total_row_parts = [f"{'Total':<{city_col_width}}"]
        total_row_parts.extend([
            f"{np.sum(scaled_results[method]):>{data_col_width},.0f}"
            for method in valid_methods
        ])
        print("".join(total_row_parts))
        print(table_separator)
        print("(Estimates based on steady-state probabilities and total population)")
        print(separator + "\n")


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
            raise ConfigError(f"Configuration initialization failed: {e}") from e

        self._master_rng = np.random.default_rng(self.m_cfg.SEED)
        self.galton_board = GaltonBoard(self.g_cfg)
        self.galton_board.set_rng_seed(self.m_cfg.SEED)
        self.markov_sim = MarkovChainSimulator(self.m_cfg)
        self.visualizer = Visualizer(self.v_cfg)
        self.city_migration: CityMigrationSimulation | None = None

    def _run_task(
        self, task_name: str, task_func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> bool:
        print(f"\n--- Running {task_name} ---")
        start_time = time.monotonic()
        success = False
        try:
            task_func(*args, **kwargs)
            success = True
        except (
            SimulationError, VisualizationError, ConfigError,
            IOError, InvalidTransitionMatrixError
        ) as e:
            print(
                f"ERROR in {task_name}: {type(e).__name__}: {e}", file=sys.stderr
            )
        except Exception as e:
            print(
                f"UNEXPECTED ERROR in {task_name}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
        finally:
            elapsed_time = time.monotonic() - start_time
            status = "Successfully Completed" if success else "Finished with ERRORS"
            print(f"--- {task_name} {status} (Duration: {elapsed_time:.2f}s) ---")
        return success

    def run_galton(self, save_image: bool = True) -> bool:
        def task(save: bool):
            print(
                f"Simulating Galton board ({self.g_cfg.MODE} mode, "
                f"{self.g_cfg.NUM_BALLS} balls)..."
            )
            self.galton_board.simulate()
            print("Simulation complete. Generating/Saving image...")
            if save:
                try:
                    saved_path = self.galton_board.save_image()
                    print(f"Galton board image saved to: {saved_path}")
                except IOError as e:
                    print(f"ERROR saving Galton image: {e}", file=sys.stderr)
                    raise
            else:
                _ = self.galton_board.generate_image()
                print("Galton board image generated (not saved).")

        return self._run_task("Galton Board Simulation", task, save_image)

    def run_random_walks(
        self, show_plots: bool = True, save_plots: bool = True
    ) -> bool:
        def task(show: bool, save: bool):
            print(
                f"Simulating {self.m_cfg.RANDOM_WALK_NUM} random walks "
                f"({self.m_cfg.RANDOM_WALK_STEPS} steps)..."
            )
            walks_data = self.markov_sim.simulate_random_walks()
            n_walks, n_steps_p1 = walks_data.shape
            n_steps = n_steps_p1 - 1

            if n_walks == 0 or n_steps < 0:
                print("No random walks were generated or steps <= 0.")
                return

            print(f"Generated {n_walks} walks, {n_steps} steps each.")
            print("Generating visualizations...")

            wp_path: Path | None = None
            dp_path: Path | None = None
            if save:
                wp_path = DEFAULT_OUTPUT_DIR / self.v_cfg.DEFAULT_MARKOV_PLOT_FILENAME
                dp_path = DEFAULT_OUTPUT_DIR / self.v_cfg.DEFAULT_DIST_PLOT_FILENAME

            self.visualizer.plot_random_walks(
                walks_data, show=show, path=wp_path
            )

            if n_steps > 0:
                self.visualizer.plot_distribution_snapshots(
                    walks_data, show=show, path=dp_path
                )
            else:
                print("Skipping distribution snapshots (0 steps).")

            if save:
                if wp_path:
                    print(f"RW plot saved/attempted: {wp_path.resolve()}")
                if dp_path and n_steps > 0:
                    print(f"Dist plot saved/attempted: {dp_path.resolve()}")

        return self._run_task(
            "Random Walk Simulation & Visualization", task, show_plots, save_plots
        )

    def setup_city_migration(self) -> bool:
        if self.city_migration is None:
            print("Setting up default City Migration simulation...")
            try:
                self.city_migration = CityMigrationSimulation.create_default(
                    simulator=self.markov_sim
                )
                print("City Migration simulation initialized.")
                return True
            except Exception as e:
                print(
                    f"ERROR setting up City Migration simulation: {e}",
                    file=sys.stderr
                )
                self.city_migration = None
                return False
        return True

    def run_city_migration(self) -> bool:
        def task():
            if self.city_migration is None:
                if not self.setup_city_migration():
                    raise SimulationError(
                        "City Migration setup failed, cannot run analysis."
                    )
            assert self.city_migration is not None
            self.city_migration.run_and_display()

        return self._run_task("City Migration Analysis", task)

    def run_all(
        self,
        run_galton_sim: bool = True,
        run_rw_sim: bool = True,
        run_city_sim: bool = True,
        show_plots: bool = True,
        save_outputs: bool = True,
    ) -> bool:
        max_width = 78
        print("\n" + "*" * max_width)
        print(f"{'Integrated Simulation Suite Run':^{max_width}}")
        print("*" * max_width)
        start_time = time.monotonic()

        results = []
        if run_galton_sim:
            results.append(self.run_galton(save_image=save_outputs))
        if run_rw_sim:
            results.append(
                self.run_random_walks(
                    show_plots=show_plots, save_plots=save_outputs
                )
            )
        if run_city_sim:
            results.append(self.run_city_migration())

        elapsed_time = time.monotonic() - start_time
        overall_success = all(r is True for r in results)

        print("\n--- Simulation Suite Summary ---")
        print(f"Total execution time: {elapsed_time:.2f} seconds.")
        status_msg = "SUCCESS" if overall_success else "COMPLETED WITH ERRORS"
        print(f"Overall status: {status_msg}")
        print("*" * max_width + "\n")

        return overall_success


def main_simulation_runner() -> int:
    plt.ioff()
    exit_code = 0

    try:
        print("Initializing Simulation Runner...")
        runner = SimulationRunner()
        print("Initialization complete. Starting simulation suite...")
        all_success = runner.run_all(show_plots=False, save_outputs=True)
        exit_code = 0 if all_success else 1
    except ConfigError as e:
        print(f"\nCRITICAL CONFIGURATION ERROR: {e}", file=sys.stderr)
        print("Aborting due to invalid configuration.", file=sys.stderr)
        exit_code = 2
    except Exception as e:
        print(
            f"\nCRITICAL UNHANDLED ERROR: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()
        exit_code = 3
    finally:
        plt.close("all")

    print(f"\nSimulation run finished. Exiting with code {exit_code}.")
    return exit_code


class TestSimulationSuite(unittest.TestCase):
    test_output_dir: ClassVar[Path]
    run_id: ClassVar[str]
    default_g_cfg: ClassVar[GaltonConfig]
    default_m_cfg: ClassVar[MarkovConfig]
    default_v_cfg: ClassVar[VisConfig]

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_output_dir = DEFAULT_OUTPUT_DIR / "unit_tests"
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)
        cls.run_id = uuid.uuid4().hex[:8]
        print(f"\n[Test Setup] Test Output Directory: {cls.test_output_dir}")
        print(f"[Test Setup] Test Run ID: {cls.run_id}")

        g_fname = f"test_galton_{cls.run_id}.png"
        m_fname = f"test_markov_plots_{cls.run_id}.png"
        d_fname = f"test_dist_plots_{cls.run_id}.png"

        cls.default_g_cfg = GaltonConfig(
            NUM_BALLS=100, NUM_ROWS=5, BOARD_WIDTH=50, BOARD_HEIGHT=40,
            SMOOTHING_WINDOW=1,
            DEFAULT_IMAGE_FILENAME=g_fname
        )
        cls.default_m_cfg = MarkovConfig(
            RANDOM_WALK_NUM=10, RANDOM_WALK_STEPS=20, SEED=42,
            STEADY_STATE_ITERATIONS=500, STEADY_STATE_BURN_IN=50,
            ANALYSIS_METHODS=("brute_force", "eigenvalue", "linear_system", "monte_carlo")
        )
        cls.default_v_cfg = VisConfig(
            FIGSIZE=(3, 2), DPI=75,
            DEFAULT_MARKOV_PLOT_FILENAME=m_fname,
            DEFAULT_DIST_PLOT_FILENAME=d_fname
        )

    @classmethod
    def tearDownClass(cls) -> None:
        print("\n[Test Cleanup] Removing test output files...")
        removed_count = 0
        try:
            if cls.test_output_dir.exists():
                for f in cls.test_output_dir.glob(f"*_{cls.run_id}.*"):
                    if f.is_file():
                        f.unlink(missing_ok=True)
                        removed_count += 1
                print(f"[Test Cleanup] Removed {removed_count} test files.")
        except OSError as e:
            print(f"[Test Cleanup] Warning: Error during cleanup: {e}", file=sys.stderr)

    def _get_test_filepath(self, filename: str) -> Path:
        return self.test_output_dir / filename

    def test_A01_default_configs_valid(self) -> None:
        try:
            g, m, v = GaltonConfig(), MarkovConfig(), VisConfig()
            self.assertIsInstance(g, GaltonConfig)
            self.assertIsInstance(m, MarkovConfig)
            self.assertIsInstance(v, VisConfig)
        except ConfigError as e:
            self.fail(f"Default configuration initialization failed: {e}")

    def test_A02_invalid_configs_raise_error(self) -> None:
        with self.assertRaisesRegex(ConfigError, "positive integer"):
            GaltonConfig(NUM_BALLS=-1)
        with self.assertRaisesRegex(ConfigError, "cannot be negative"):
            MarkovConfig(RANDOM_WALK_NUM=-1)
        with self.assertRaisesRegex(ConfigError, "must be less than"):
            MarkovConfig(STEADY_STATE_BURN_IN=1000, STEADY_STATE_ITERATIONS=500)
        with self.assertRaisesRegex(ConfigError, "must be positive"):
            VisConfig(FIGSIZE=(0, 1))
        with self.assertRaisesRegex(ConfigError, "Invalid analysis methods"):
            MarkovConfig(ANALYSIS_METHODS=("invalid_method",))
        with self.assertRaisesRegex(ConfigError, "at least one analysis method"):
            MarkovConfig(ANALYSIS_METHODS=())
        with self.assertRaisesRegex(ConfigError, "must be positive"):
             GaltonConfig(BOARD_WIDTH=0)

    def test_B01_galton_physics_mode_simulation(self) -> None:
        cfg = self.default_g_cfg._replace(MODE="physics_based")
        gb = GaltonBoard(cfg)
        gb.set_rng_seed(0)
        gb.simulate()
        self.assertEqual(len(gb.slot_counts), cfg.BOARD_WIDTH)
        self.assertTrue(sum(gb.slot_counts) > 0, "No balls counted")
        self.assertAlmostEqual(sum(gb.slot_counts), cfg.NUM_BALLS, delta=1)

    def test_B02_galton_simple_random_walk_mode_simulation(self) -> None:
        cfg = self.default_g_cfg._replace(MODE="simple_random_walk")
        gb = GaltonBoard(cfg)
        gb.set_rng_seed(1)
        gb.simulate()
        self.assertEqual(len(gb.slot_counts), cfg.BOARD_WIDTH)
        self.assertEqual(sum(gb.slot_counts), cfg.NUM_BALLS)

    def test_B03_galton_image_saving(self) -> None:
        gb = GaltonBoard(self.default_g_cfg)
        gb.simulate()
        test_img_path = self._get_test_filepath(gb.config.DEFAULT_IMAGE_FILENAME)
        try:
            saved_path_str = gb.save_image(test_img_path)
            saved_path = Path(saved_path_str)
            self.assertTrue(saved_path.exists(), "Image file was not created.")
            self.assertEqual(saved_path, test_img_path.resolve())
            with Image.open(saved_path) as img:
                self.assertEqual(img.size, (gb.config.BOARD_WIDTH, gb.config.BOARD_HEIGHT))
                self.assertEqual(img.mode, "RGB")
        except (IOError, VisualizationError) as e:
            self.fail(f"Galton image saving failed: {e}")
        finally:
            test_img_path.unlink(missing_ok=True)

    def test_B04_galton_smoothing_effect(self) -> None:
        gb = GaltonBoard(GaltonConfig(SMOOTHING_WINDOW=3, BOARD_WIDTH=7))
        input_counts = np.array([0, 0, 0, 99, 0, 0, 0], dtype=int)
        gb._apply_smoothing(input_counts)
        expected_val = 33
        self.assertEqual(gb.slot_counts[2], expected_val)
        self.assertEqual(gb.slot_counts[3], expected_val)
        self.assertEqual(gb.slot_counts[4], expected_val)
        self.assertEqual(gb.slot_counts[0], 0)
        self.assertEqual(gb.slot_counts[6], 0)
        self.assertEqual(sum(gb.slot_counts), sum(input_counts))

    def test_C01_transition_matrix_validation(self) -> None:
        try:
            TransitionMatrix(np.identity(2))
            tm_empty = TransitionMatrix(np.zeros((0, 0)))
            self.assertEqual(len(tm_empty), 0)
        except InvalidTransitionMatrixError as e:
            self.fail(f"Valid TransitionMatrix raised error: {e}")

        with self.assertRaisesRegex(InvalidTransitionMatrixError, "must be square"):
            TransitionMatrix([[0.1, 0.9], [0.2, 0.7, 0.1]])
        with self.assertRaisesRegex(InvalidTransitionMatrixError, "Rows must sum to 1"):
            TransitionMatrix([[0.1, 0.8], [0.8, 0.2]])
        with self.assertRaisesRegex(InvalidTransitionMatrixError, "cannot contain negative"):
            TransitionMatrix([[-0.1, 1.1], [0.5, 0.5]])
        with self.assertRaisesRegex(InvalidTransitionMatrixError, "Invalid data type"):
            TransitionMatrix("not a valid matrix input")

    def test_C02_random_walk_generation(self) -> None:
        sim = MarkovChainSimulator(self.default_m_cfg)
        walks = sim.simulate_random_walks()
        expected_shape = (self.default_m_cfg.RANDOM_WALK_NUM, self.default_m_cfg.RANDOM_WALK_STEPS + 1)
        self.assertEqual(walks.shape, expected_shape)
        self.assertTrue(np.all(walks[:, 0] == 0))
        if walks.shape[1] > 1 and walks.shape[0] > 1:
             self.assertTrue(np.var(walks[:, -1]) > 0, "Walks show no variance at end.")

    def test_C03_steady_state_computation(self) -> None:
        tm_data = np.array([[0.8, 0.2], [0.3, 0.7]])
        tm = TransitionMatrix(tm_data)
        known_steady_state = np.array([0.6, 0.4])
        test_m_cfg = self.default_m_cfg._replace(
            STEADY_STATE_ITERATIONS=10000,
            STEADY_STATE_BURN_IN=1000,
            SEED=123
        )
        sim = MarkovChainSimulator(test_m_cfg)
        results = sim.analyze_steady_state(tm)

        self.assertEqual(set(results.keys()), set(test_m_cfg.ANALYSIS_METHODS))
        tolerances = {
            "brute_force": 1e-7, "eigenvalue": 1e-8,
            "linear_system": 1e-8, "monte_carlo": 5e-3
        }

        for method in test_m_cfg.ANALYSIS_METHODS:
            self.assertIn(method, results, f"Method {method} missing.")
            computed_vector = results[method]
            self.assertIsNotNone(computed_vector, f"{method} result is None.")
            self.assertFalse(np.isnan(computed_vector).any(), f"{method} result has NaN.")
            self.assertEqual(computed_vector.shape, known_steady_state.shape)
            self.assertTrue(
                np.allclose(computed_vector, known_steady_state, atol=tolerances[method]),
                f"Method '{method}' failed. Expected {known_steady_state}, "
                f"got {computed_vector}"
            )

    def test_D01_visualization_plot_creation(self) -> None:
        sim = MarkovChainSimulator(self.default_m_cfg)
        vis = Visualizer(self.default_v_cfg)
        walks_data = sim.simulate_random_walks()
        rw_path = self._get_test_filepath(self.default_v_cfg.DEFAULT_MARKOV_PLOT_FILENAME)
        ds_path = self._get_test_filepath(self.default_v_cfg.DEFAULT_DIST_PLOT_FILENAME)

        try:
            vis.plot_random_walks(walks_data, show=False, path=rw_path)
            self.assertTrue(rw_path.exists(), "Random walk plot file not created.")
        except VisualizationError as e:
            self.fail(f"plot_random_walks failed: {e}")
        finally:
            rw_path.unlink(missing_ok=True)

        try:
            vis.plot_distribution_snapshots(
                walks_data, times=[5, 10, 20], show=False, path=ds_path
            )
            self.assertTrue(ds_path.exists(), "Distribution snapshot plot not created.")
        except VisualizationError as e:
            self.fail(f"plot_distribution_snapshots failed: {e}")
        finally:
            ds_path.unlink(missing_ok=True)

    def test_D02_visualization_with_empty_data(self) -> None:
        vis = Visualizer(self.default_v_cfg)
        empty_0_walks = np.zeros((0, self.default_m_cfg.RANDOM_WALK_STEPS + 1))
        empty_0_steps = np.zeros((self.default_m_cfg.RANDOM_WALK_NUM, 0))

        try:
            vis.plot_random_walks(empty_0_walks, show=False, path=None)
            vis.plot_random_walks(empty_0_steps, show=False, path=None)
            vis.plot_distribution_snapshots(empty_0_walks, show=False, path=None)
            vis.plot_distribution_snapshots(empty_0_steps, show=False, path=None)
        except Exception as e:
            self.fail(f"Visualization method failed unexpectedly with empty data: {e}")

    def test_E01_simulation_runner_execution(self) -> None:
        runner = SimulationRunner(
            galton_config=self.default_g_cfg,
            markov_config=self.default_m_cfg,
            vis_config=self.default_v_cfg
        )
        g_path = self._get_test_filepath(self.default_g_cfg.DEFAULT_IMAGE_FILENAME)
        rw_path = self._get_test_filepath(self.default_v_cfg.DEFAULT_MARKOV_PLOT_FILENAME)
        ds_path = self._get_test_filepath(self.default_v_cfg.DEFAULT_DIST_PLOT_FILENAME)

        overall_success = runner.run_all(show_plots=False, save_outputs=True)

        self.assertTrue(overall_success, "SimulationRunner.run_all reported failure.")
        self.assertTrue(g_path.exists(), "Runner failed to create Galton image.")
        self.assertTrue(rw_path.exists(), "Runner failed to create random walk plot.")
        if self.default_m_cfg.RANDOM_WALK_STEPS > 0:
            self.assertTrue(ds_path.exists(), "Runner failed to create distribution plot.")

    def test_E02_city_migration_integration(self) -> None:
        runner = SimulationRunner(markov_config=self.default_m_cfg)
        self.assertTrue(runner.setup_city_migration(), "City migration setup failed.")
        self.assertIsInstance(runner.city_migration, CityMigrationSimulation)
        success = runner.run_city_migration()
        self.assertTrue(success, "City migration run reported failure.")


def run_tests(verbosity: int = 2) -> int:
    print("\n--- Running Unit Tests ---")
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    suite = loader.loadTestsFromTestCase(TestSimulationSuite)
    runner = unittest.TextTestRunner(
        verbosity=verbosity, failfast=False, buffer=True
    )
    result = runner.run(suite)
    print("--- Unit Tests Complete ---")
    return 0 if result.wasSuccessful() else 1


def display_help() -> None:
    script_name = Path(__file__).name
    print(f"\nUsage: python {script_name} [options]")
    print("\nOptions:")
    print("  --test [-v N] : Run internal unit tests (verbosity N=0,1,2 default=2).")
    print("  --help, -h    : Display this help message and exit.")
    print("  (no options)  : Run the full simulation suite with default settings.")
    print("                  Outputs are saved to './simulation_outputs/'.")
    print("\nDescription:")
    print("  Simulates various stochastic processes including:")
    print("    - Galton Board (Physics-based or Simple Random Walk)")
    print("    - Markov Chain Random Walks")
    print("    - Steady-State Analysis of Markov Chains (various methods)")
    print("    - Example: City Population Migration Model")


if __name__ == "__main__":
    exit_code: int = 0
    args = sys.argv[1:]

    if "--test" in args:
        test_verbosity = 2
        try:
            if "-v" in args:
                v_index = args.index("-v")
                if v_index + 1 < len(args):
                    level = int(args[v_index + 1])
                    if level in [0, 1, 2]:
                        test_verbosity = level
                    else:
                        print("Warning: Verbosity level must be 0, 1, or 2. Using 2.", file=sys.stderr)
        except (ValueError, IndexError):
            print(
                "Warning: Invalid verbosity level provided after -v. Using default (2).",
                file=sys.stderr
            )
            test_verbosity = 2
        exit_code = run_tests(verbosity=test_verbosity)

    elif "--help" in args or "-h" in args:
        display_help()
        exit_code = 0

    else:
        exit_code = main_simulation_runner()

    sys.exit(exit_code)
