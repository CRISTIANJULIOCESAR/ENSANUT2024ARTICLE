"""Project-root discovery and standardized input/output paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_MARKERS = ("src/ensanut_hba1c", "notebooks", "data/input")


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).expanduser().resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if all((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        "The project root was not found. Open Jupyter from the project folder "
        "or from its notebooks/ folder."
    )


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "ProjectPaths":
        return cls(find_project_root(start))

    @property
    def input_dir(self) -> Path:
        return self.root / "data" / "input"

    @property
    def notebook1_results(self) -> Path:
        return self.root / "results" / "01_modeling_clustering"

    @property
    def notebook2_results(self) -> Path:
        return self.root / "results" / "02_naive_bayes"

    @property
    def handoff_dir(self) -> Path:
        return self.notebook1_results / "handoff_to_notebook2"

    def ensure(self) -> "ProjectPaths":
        for path in (
            self.input_dir,
            self.notebook1_results,
            self.notebook2_results,
            self.handoff_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self
