from __future__ import annotations
import logging, uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, cast, runtime_checkable
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from numpy.random import Generator

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

class SimulationMethod(Enum):
    BRUTE_FORCE = auto()
    EIGENVALUE = auto()
    LINEAR_SYSTEM = auto()
    MONTE_CARLO = auto()

@runtime_checkable
class StateComputable(Protocol):
    def compute(
        self, 
        matrix: NDArrayF64, 
        init_state: NDArrayF64
    ) -> NDArrayF64:
        ...

@dataclass(frozen=True)
class SimulationConfig:
    num_walks: int = field(default=1000)
    num_steps: int = field(default=100)
    burn_in: int = field(default=1000)
    iterations: int = field(default=1000000)
    methods: tuple[Method, ...] = field(
        default=("brute_force", "eigenvalue", "linear_system")
    )
    seed: int | None = field(default=None)
    def __post_init__(self) -> None:
        if self.num_walks <= 0 or self.num_steps <= 0:
            raise ValueError("Number of walks/steps must be positive.")
        if self.burn_in >= self.iterations:
            raise ValueError("Burn-in period must be less than iterations.")

@dataclass(frozen=True)
class VisualizationConfig:
    figsize: tuple[int, int] = field(default=(10, 6))
    dpi: int = field(default=300)
    alpha: float = field(default=0.3)
    linewidth: float = field(default=0.8)
    grid_alpha: float = field(default=0.3)
    hist_alpha: float = field(default=0.4)
    hist_bins: int = field(default=30)
    x_range: tuple[float, float] = field(default=(-25, 25))
    x_points: int = field(default=100)
    subplot_figsize: tuple[int, int] = field(default=(15, 3))
    def __post_init__(self) -> None:
        if any(x <= 0 for x in self.figsize + (self.dpi,)):
            raise ValueError("Figure size/dpi must be positive.")
        if not 0 < self.alpha <= 1:
            raise ValueError("Alpha must be in (0, 1].")

class StateComputer(ABC):
    @abstractmethod
    def compute(
        self,
        matrix: NDArrayF64, 
        init_state: NDArrayF64
    ) -> NDArrayF64:
        ...

class BruteForceComputer(StateComputer):
    def __init__(self, iterations: int) -> None:
        if iterations <= 0:
            raise ValueError("Iterations must be positive.")
        self.iterations: Final = iterations
    def compute(
        self, 
        matrix: NDArrayF64, 
        init_state: NDArrayF64
    ) -> NDArrayF64:
        state = init_state.copy()
        for _ in range(self.iterations):
            state = matrix.T @ state
        return cast(NDArrayF64, state.flatten())

class EigenvalueComputer(StateComputer):
    def compute(
        self, 
        matrix: NDArrayF64, 
        _: NDArrayF64
    ) -> NDArrayF64:
        ev, evec = np.linalg.eig(matrix.T)
        msk = np.isclose(ev, 1.0)
        steady = evec[:, msk].real
        steady /= steady.sum()
        return cast(NDArrayF64, steady.flatten())

class LinearSystemComputer(StateComputer):
    def compute(
        self, 
        matrix: NDArrayF64, 
        _: NDArrayF64
    ) -> NDArrayF64:
        n = matrix.shape[0]
        sys_mat = np.vstack([matrix.T - np.eye(n), np.ones(n)])
        sys_vec = np.zeros(n + 1)
        sys_vec[-1] = 1.0
        sol, *_ = np.linalg.lstsq(sys_mat, sys_vec, rcond=None)
        return cast(NDArrayF64, sol)

class TransitionMatrix:
    def __init__(self, matrix: NDArrayF64) -> None:
        if not np.allclose(matrix.sum(axis=1), 1):
            logger.error("Invalid transition matrix: rows do not sum to 1.")
            raise InvalidTransitionMatrixError(
                "Rows of the transition matrix must sum to 1."
            )
        self._matrix: Final[NDArrayF64] = matrix
    @property
    def matrix(self) -> NDArrayF64:
        return self._matrix
    def __getitem__(self, idx: int) -> NDArrayF64:
        return self._matrix[idx]

class MarkovChainSimulator:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self._rng: Generator = np.random.default_rng(self.config.seed)
        self._computers = {
            "brute_force": BruteForceComputer(self.config.iterations),
            "eigenvalue": EigenvalueComputer(),
            "linear_system": LinearSystemComputer()
        }
    def simulate_random_walks(self) -> NDArrayF64:
        steps = self._rng.choice(
            [-1, 1],
            size=(
                self.config.num_walks,
                self.config.num_steps
            )
        )
        starts = np.zeros((self.config.num_walks, 1))
        traj = np.hstack((starts, steps))
        try:
            return np.cumsum(traj, axis=1)
        except ValueError as err:
            logger.exception("Random walk simulation failed.")
            raise SimulationError("Failed to simulate random walks.") from err
    def analyze_steady_state(
        self,
        transition_matrix: TransitionMatrix,
        initial_state: NDArrayF64
    ) -> dict[Method, NDArrayF64]:
        res: dict[Method, NDArrayF64] = {}
        for method in self.config.methods:
            comp = self._computers[method]
            try:
                res[method] = comp.compute(
                    transition_matrix.matrix, initial_state
                )
            except Exception as err:
                logger.exception(
                    f"Failed to compute steady state using {method} method."
                )
                raise SimulationError(
                    f"Failed to compute steady state using {method} method."
                ) from err
        return res
    def monte_carlo_estimate(
        self,
        transition_matrix: TransitionMatrix,
        initial_state: int | None = None
    ) -> NDArrayF64:
        mat = transition_matrix.matrix
        n = mat.shape[0]
        traj = self._rng.integers(
            0, n,
            size=self.config.iterations + 1,
            dtype=np.int64
        )
        if initial_state is not None:
            traj[0] = initial_state
        effective = traj[self.config.burn_in:]
        cnt = np.bincount(effective, minlength=n)
        return cast(NDArrayF64, cnt / len(effective))

class MarkovChainVisualizer:
    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()
    def plot_walks(
        self, 
        walks: NDArrayF64, 
        show: bool = True,
        save_path: str | None = None
    ) -> None:
        fig = None
        try:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            t = np.arange(walks.shape[1])
            for w in walks:
                ax.plot(
                    t, w,
                    alpha=self.config.alpha,
                    linewidth=self.config.linewidth
                )
            ax.set_title("Random Walk Trajectories")
            ax.set_xlabel("Step")
            ax.set_ylabel("Displacement")
            ax.grid(True, alpha=self.config.grid_alpha)
            if save_path:
                fp = Path(save_path)
                plt.savefig(fp, dpi=self.config.dpi, bbox_inches="tight")
                logger.info(f"Saved plot to {fp}")
            if show:
                plt.show()
            plt.close(fig)
        except Exception as err:
            logger.exception("Plot generation failed.")
            raise VisualizationError("Failed to generate plot.") from err
        finally:
            if fig:
                plt.close(fig)
    def plot_distribution_snapshots(
        self, 
        walks: NDArrayF64, 
        times: list[int] | None = None,
        show: bool = True
    ) -> None:
        fig = None
        try:
            pts = times or list(range(20, 101, 20))
            fig, axes = plt.subplots(
                1, len(pts),
                figsize=self.config.subplot_figsize,
                sharex=True
            )
            x_vals = np.linspace(
                self.config.x_range[0],
                self.config.x_range[1],
                self.config.x_points
            )
            for i, t_val in enumerate(pts):
                axes[i].hist(
                    walks[:, t_val],
                    bins=self.config.hist_bins,
                    density=True,
                    alpha=self.config.hist_alpha,
                    color="skyblue"
                )
                axes[i].set_title(f"t = {t_val}")
                pdf = stats.norm(loc=0, scale=np.sqrt(t_val)).pdf(x_vals)
                axes[i].plot(
                    x_vals, pdf,
                    "r-", linewidth=self.config.linewidth
                )
            plt.tight_layout()
            if show:
                plt.show()
            plt.close(fig)
        except Exception as err:
            logger.exception("Distribution plot failed.")
            raise VisualizationError("Failed to generate distribution plot.") from err
        finally:
            if fig:
                plt.close(fig)

class CityMigrationSimulation:
    def __init__(
        self,
        transition_matrix: TransitionMatrix,
        population: NDArrayF64,
        cities: list[str],
        simulator: MarkovChainSimulator
    ) -> None:
        self.transition_matrix = transition_matrix
        self.population = population
        self.cities = cities
        self.simulator = simulator
    def run(self) -> None:
        try:
            res = self.simulator.analyze_steady_state(
                self.transition_matrix, self.population
            )
            print("\nCity Migration Simulation Results:")
            print("=" * 40)
            for m, dist in res.items():
                print(f"\n{m.replace('_', ' ').title()} Method:")
                for city, val in zip(self.cities, dist):
                    print(f"  {city}: {val:.0f}")
            mc = self.simulator.monte_carlo_estimate(
                self.transition_matrix
            )
            print("\nMonte Carlo Estimation:")
            for city, freq in zip(self.cities, mc):
                print(f"  {city}: {freq:.5f}")
            tot = self.population.sum()
            scaled = mc * tot
            print("\nScaled Monte Carlo Estimation (Population):")
            for city, val in zip(self.cities, scaled):
                print(f"  {city}: {val:.0f}")
        except Exception as err:
            logger.exception("City migration simulation failed.")
            raise SimulationError("Failed to run city migration simulation.") from err

class SimulationRunner:
    def __init__(
        self,
        simulation_config: SimulationConfig,
        visualization_config: VisualizationConfig,
        transition_matrix_data: NDArrayF64,
        population: NDArrayF64,
        cities: list[str]
    ) -> None:
        self.simulation_config = simulation_config
        self.visualization_config = visualization_config
        self.transition_matrix = TransitionMatrix(transition_matrix_data)
        self.population = population
        self.cities = cities
        self.simulator = MarkovChainSimulator(self.simulation_config)
        self.visualizer = MarkovChainVisualizer(self.visualization_config)
        self.city_simulation = CityMigrationSimulation(
            self.transition_matrix, 
            self.population, 
            self.cities,
            self.simulator
        )
    def run_random_walk_simulation(self) -> None:
        try:
            logger.info("Running Random Walk Simulation...")
            walks = self.simulator.simulate_random_walks()
            out_file = f"markov_walks_{uuid.uuid4().hex}.png"
            self.visualizer.plot_walks(walks, save_path=out_file)
            self.visualizer.plot_distribution_snapshots(walks)
            logger.info("Random Walk Simulation completed successfully.")
        except Exception as err:
            logger.exception("Random walk simulation failed.")
            raise
    def run_city_migration_simulation(self) -> None:
        try:
            logger.info("Running City Migration Markov Chain Simulation...")
            self.city_simulation.run()
            logger.info("City Migration Simulation completed successfully.")
        except Exception as err:
            logger.exception("City migration simulation failed.")
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
        sim_cfg = SimulationConfig()
        vis_cfg = VisualizationConfig()
        trans_data = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.04, 0.01, 0.95]
        ])
        pop = np.array([300000, 300000, 300000]).reshape(-1, 1)
        cities = ["Raleigh", "Chapel Hill", "Durham"]
        runner = SimulationRunner(sim_cfg, vis_cfg, trans_data, pop, cities)
        runner.run()
    except Exception as err:
        logger.error(f"Main function failed: {err}")
    finally:
        plt.close("all")

if __name__ == "__main__":
    main()