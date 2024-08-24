from __future__ import annotations
from pathlib import Path
import os


class State:
    """Class for storing the state of Run"""

    run_dir: Path
    """The directory to save the run"""

    state: str
    """The state of the Run"""

    COMPLETE = "COMPLETE"
    """The state of the Run when it is complete"""

    RUNNING = "RUNNING"
    """The state of the Run when it is running"""

    CRASHED = "CRASHED"
    """The state of the Run when it has crashed"""

    def __init__(
        self,
        run_dir: Path
    ) -> None:
        self.run_dir = run_dir
        self.set_state(self.RUNNING)

    def set_state(self, state: str) -> None:
        """Set the state of the Run"""
        self.state = state
        self.save()

    def get_state(self) -> str:
        """Get the state of the Run"""
        return self.state

    def save(self) -> None:
        """Save the state of the Run"""
        for file in os.listdir(self.run_dir):
            if file.endswith(".state"):
                os.remove(self.run_dir / file)
        with open(self.run_dir / f"{self.state}.state", "w") as f:
            f.write(self.state)