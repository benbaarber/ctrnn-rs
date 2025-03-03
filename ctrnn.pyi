import numpy as np
from numpy.typing import NDArray

class CTRNN:
    taus: NDArray[np.float64]
    weights: NDArray[np.float64]
    biases: NDArray[np.float64]
    gains: NDArray[np.float64]
    states: NDArray[np.float64]
    outputs: NDArray[np.float64]
    input_size: int
    hidden_size: int
    output_size: int
    total_size: int
    def __new__(
        cls, input_size: int, hidden_size: int, output_size: int, step_size: float = 0.1
    ) -> CTRNN: ...
    def euler_step(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform an Euler integration step with external inputs
        """
    def output(self) -> NDArray[np.float64]:
        """
        Get CTRNN output
        """
    def to_file(self, filename: str) -> None:
        """
        Save parameters to a json file
        """
    @classmethod
    def from_file(cls, filename: str) -> CTRNN:
        """
        Load parameters from a json file
        """
    def load_genome_json(self, genome: str) -> None:
        """
        Load a squid genome from a json string
        """
    def burn_in(self, steps: int) -> None:
        """
        Apply `steps` number of Euler steps with no external input
        """
    def reset(self) -> None:
        """
        Reset CTRNN state
        """
    def clone(self) -> CTRNN:
        """
        Clone this instance
        """
    def __getstate__(self) -> bytes: ...
    def __setstate__(self, state: bytes) -> None: ...
