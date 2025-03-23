import logging
import sys
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, cast, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import unittest

NDArrayF64: TypeAlias = npt.NDArray[np.float64]
Method: TypeAlias = Literal["brute_force", "eigenvalue", "linear_system"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SimulationError(Exception): pass
class VisualizationError(Exception): pass
class InvalidTransitionMatrixError(SimulationError): pass

@dataclass(frozen=True)
class SimulationConfig:
    num_walks: int = 1000
    num_steps: int = 100
    burn_in: int = 1000
    iterations: int = 1000000
    methods: tuple[Method, ...] = field(default=("brute_force", "eigenvalue", "linear_system"))
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_walks <= 0 or self.num_steps <= 0:
            raise ValueError("Number of walks/steps must be positive")
        if self.burn_in >= self.iterations:
            raise ValueError("Burn-in period must be less than iterations")

@dataclass(frozen=True)
class VisualizationConfig:
    figsize: tuple[int, int] = (10, 6)
    dpi: int = 300
    alpha: float = 0.3
    linewidth: float = 0.8
    grid_alpha: float = 0.3
    hist_alpha: float = 0.4
    hist_bins: int = 30
    x_range: tuple[float, float] = (-25, 25)
    x_points: int = 100
    subplot_figsize: tuple[int, int] = (15, 3)

    def __post_init__(self) -> None:
        if any(x <= 0 for x in self.figsize + (self.dpi,)):
            raise ValueError("Figure size/dpi must be positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("Alpha must be in (0, 1]")

@runtime_checkable
class StateComputable(Protocol):
    def compute(self, matrix: NDArrayF64, init_state: NDArrayF64) -> NDArrayF64:
        ...

class StateComputer(ABC):
    @abstractmethod
    def compute(self, matrix: NDArrayF64, init_state: NDArrayF64) -> NDArrayF64:
        ...

class BruteForceComputer(StateComputer):
    def __init__(self, iterations: int) -> None:
        if iterations <= 0:
            raise ValueError("Iterations must be positive")
        self.iterations: Final = iterations

    def compute(self, matrix: NDArrayF64, init_state: NDArrayF64) -> NDArrayF64:
        state = init_state.copy()
        for _ in range(self.iterations):
            state = matrix.T @ state
        return cast(NDArrayF64, state.flatten())

class EigenvalueComputer(StateComputer):
    def compute(self, matrix: NDArrayF64, _: NDArrayF64) -> NDArrayF64:
        vals, vecs = np.linalg.eig(matrix.T)
        mask = np.isclose(vals, 1.0)
        steady = vecs[:, mask].real
        steady /= steady.sum()
        return cast(NDArrayF64, steady.flatten())

class LinearSystemComputer(StateComputer):
    def compute(self, matrix: NDArrayF64, _: NDArrayF64) -> NDArrayF64:
        dim = matrix.shape[0]
        a = np.vstack([matrix.T - np.eye(dim), np.ones(dim)])
        b = np.zeros(dim + 1)
        b[-1] = 1.0
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        return cast(NDArrayF64, solution)

class TransitionMatrix:
    def __init__(self, matrix: NDArrayF64) -> None:
        if not np.allclose(matrix.sum(axis=1), 1):
            logger.error("Invalid transition matrix: rows do not sum to 1")
            raise InvalidTransitionMatrixError("Rows of the transition matrix must sum to 1")
        self._matrix: Final[NDArrayF64] = matrix

    @property
    def matrix(self) -> NDArrayF64:
        return self._matrix

    def __getitem__(self, idx: int) -> NDArrayF64:
        return self._matrix[idx]

class MarkovChainSimulator:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._computers = {
            "brute_force": BruteForceComputer(self.config.iterations),
            "eigenvalue": EigenvalueComputer(),
            "linear_system": LinearSystemComputer(),
        }

    def simulate_random_walks(self) -> NDArrayF64:
        dims = (self.config.num_walks, self.config.num_steps)
        steps = self._rng.choice([-1, 1], size=dims)
        starts = np.zeros((self.config.num_walks, 1))
        walks = np.hstack((starts, steps))
        try:
            return np.cumsum(walks, axis=1)
        except ValueError as exc:
            logger.exception("Random walk simulation failed")
            raise SimulationError("Failed to simulate random walks") from exc

    def analyze_steady_state(
        self, transition_matrix: TransitionMatrix, initial_state: NDArrayF64
    ) -> dict[Method, NDArrayF64]:
        results: dict[Method, NDArrayF64] = {}
        matrix = transition_matrix.matrix
        workers = min(len(self.config.methods), 3)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._computers[method].compute, matrix, initial_state
                ): method for method in self.config.methods
            }
            for future in as_completed(futures):
                method = futures[future]
                try:
                    results[method] = future.result()
                except Exception as err:
                    logger.exception(f"Method {method} computation failed")
                    raise SimulationError(
                        f"Failed to compute steady state using {method}"
                    ) from err
        return results

    def monte_carlo_estimate(
        self, transition_matrix: TransitionMatrix, initial_state: int | None = None
    ) -> NDArrayF64:
        matrix = transition_matrix.matrix
        n_states = matrix.shape[0]
        n_samples = self.config.iterations + 1
        states = self._rng.integers(0, n_states, size=n_samples, dtype=np.int64)
        if initial_state is not None:
            states[0] = initial_state
        samples = states[self.config.burn_in:]
        counts = np.bincount(samples, minlength=n_states)
        probs = counts / len(samples)
        return cast(NDArrayF64, probs)

class MarkovChainVisualizer:
    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()

    def plot_walks(
        self, walks: NDArrayF64, show: bool = True, save_path: str | None = None
    ) -> None:
        fig = None
        try:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            steps = np.arange(walks.shape[1])
            for walk in walks:
                ax.plot(steps, walk, alpha=self.config.alpha,
                        linewidth=self.config.linewidth)
            ax.set_title("Random Walk Trajectories")
            ax.set_xlabel("Step")
            ax.set_ylabel("Displacement")
            ax.grid(True, alpha=self.config.grid_alpha)
            if save_path:
                out_path = Path(save_path)
                plt.savefig(out_path, dpi=self.config.dpi, bbox_inches="tight")
                logger.info(f"Saved plot to {out_path}")
            if show:
                plt.show()
            plt.close(fig)
        except Exception as err:
            logger.exception("Plot visualization failed")
            raise VisualizationError("Failed to generate plot") from err
        finally:
            if fig:
                plt.close(fig)

    def plot_distribution_snapshots(
        self, walks: NDArrayF64, times: list[int] | None = None, show: bool = True
    ) -> None:
        fig = None
        try:
            t_points = times or list(range(20, 101, 20))
            fig, axes = plt.subplots(
                1, len(t_points),
                figsize=self.config.subplot_figsize,
                sharex=True
            )
            x_vals = np.linspace(
                self.config.x_range[0], self.config.x_range[1],
                self.config.x_points
            )
            for i, t in enumerate(t_points):
                axes[i].hist(
                    walks[:, t],
                    bins=self.config.hist_bins,
                    density=True,
                    alpha=self.config.hist_alpha,
                    color="skyblue"
                )
                axes[i].set_title(f"t = {t}")
                std = np.sqrt(t)
                pdf = stats.norm(loc=0, scale=std).pdf(x_vals)
                axes[i].plot(x_vals, pdf, "r-", linewidth=self.config.linewidth)
            plt.tight_layout()
            if show:
                plt.show()
            plt.close(fig)
        except Exception as err:
            logger.exception("Distribution plot failed")
            raise VisualizationError("Failed to generate distribution snapshots") from err
        finally:
            if fig:
                plt.close(fig)

class CityMigrationSimulation:
    def __init__(
        self, transition_matrix: TransitionMatrix, population: NDArrayF64,
        cities: list[str], simulator: MarkovChainSimulator
    ) -> None:
        self.transition_matrix = transition_matrix
        self.population = population
        self.cities = cities
        self.simulator = simulator

    def run(self) -> None:
        try:
            results = self.simulator.analyze_steady_state(
                self.transition_matrix, self.population
            )
            self._display_results(results)
        except Exception as exc:
            logger.exception("City migration simulation failed")
            raise SimulationError("Failed to run city migration simulation") from exc

    def _display_results(self, results: dict[Method, NDArrayF64]) -> None:
        print("\nCity Migration Simulation Results:")
        print("=" * 40)
        for method, dist in results.items():
            title = method.replace("_", " ").title()
            print(f"\n{title} Method:")
            for city, pop in zip(self.cities, dist):
                print(f"  {city}: {pop:.0f}")
        mc = self.simulator.monte_carlo_estimate(self.transition_matrix)
        print("\nMonte Carlo Estimation:")
        for city, prob in zip(self.cities, mc):
            print(f"  {city}: {prob:.5f}")
        total = self.population.sum()
        scaled = mc * total
        print("\nScaled Monte Carlo Estimation (Population):")
        for city, pop in zip(self.cities, scaled):
            print(f"  {city}: {pop:.0f}")

class SimulationRunner:
    def __init__(
        self, sim_config: SimulationConfig, vis_config: VisualizationConfig,
        transition_data: NDArrayF64, population: NDArrayF64, cities: list[str]
    ) -> None:
        self.sim_config = sim_config
        self.vis_config = vis_config
        self.transition_matrix = TransitionMatrix(transition_data)
        self.population = population
        self.cities = cities
        self.simulator = MarkovChainSimulator(self.sim_config)
        self.visualizer = MarkovChainVisualizer(self.vis_config)
        self.city_simulation = CityMigrationSimulation(
            self.transition_matrix, self.population, self.cities, self.simulator
        )

    def run_random_walk_simulation(self) -> None:
        try:
            logger.info("Running Random Walk Simulation...")
            walks = self.simulator.simulate_random_walks()
            filename = f"markov_walks_{uuid.uuid4().hex[:8]}.png"
            self.visualizer.plot_walks(walks, save_path=filename)
            self.visualizer.plot_distribution_snapshots(walks)
            logger.info("Random Walk Simulation completed successfully")
        except Exception as err:
            logger.exception("Random walk simulation failed")
            raise

    def run_city_migration_simulation(self) -> None:
        try:
            logger.info("Running City Migration Markov Chain Simulation...")
            self.city_simulation.run()
            logger.info("City Migration Simulation completed successfully")
        except Exception as err:
            logger.exception("City migration simulation failed")
            raise

    def run(self) -> None:
        try:
            self.run_random_walk_simulation()
            self.run_city_migration_simulation()
        except Exception as err:
            logger.error(f"Simulation failed: {err}")
            raise
        finally:
            plt.close("all")

def main() -> None:
    plt.ioff()
    try:
        sim_config = SimulationConfig()
        vis_config = VisualizationConfig()
        transition_data = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.04, 0.01, 0.95]
        ])
        population = np.array([300000, 300000, 300000]).reshape(-1, 1)
        cities = ["Raleigh", "Chapel Hill", "Durham"]
        runner = SimulationRunner(
            sim_config, vis_config, transition_data, population, cities
        )
        runner.run()
    except Exception as err:
        logger.error(f"Main function failed: {err}")
    finally:
        plt.close("all")

class TestMarkovChain(unittest.TestCase):
    def setUp(self) -> None:
        self.sim_config = SimulationConfig(
            num_walks=10, num_steps=50, iterations=1000, burn_in=100
        )
        self.vis_config = VisualizationConfig()
        self.transition_data = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.04, 0.01, 0.95]
        ])
        self.population = np.array([1000, 1500, 2000]).reshape(-1, 1)
        self.cities = ["A", "B", "C"]
        self.runner = SimulationRunner(
            self.sim_config, self.vis_config,
            self.transition_data, self.population, self.cities
        )
        self.simulator = MarkovChainSimulator(self.sim_config)
        self.transition_matrix = TransitionMatrix(self.transition_data)

    def test_random_walk_shape(self) -> None:
        walks = self.simulator.simulate_random_walks()
        self.assertEqual(
            walks.shape,
            (self.sim_config.num_walks, self.sim_config.num_steps + 1)
        )

    def test_steady_state_methods(self) -> None:
        init_state = self.population
        results = self.simulator.analyze_steady_state(self.transition_matrix, init_state)
        self.assertTrue(all(isinstance(val, np.ndarray) for val in results.values()))

    def test_monte_carlo_estimate(self) -> None:
        probs = self.simulator.monte_carlo_estimate(self.transition_matrix)
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.argv.pop(1)
        unittest.main()
    else:
        main()