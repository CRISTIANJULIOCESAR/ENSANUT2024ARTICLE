"""Compact progress utilities for notebook and command-line workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class NullProgressBar:
    """No-op replacement used when progress display is disabled."""

    total: int | None = None
    n: int = 0

    def update(self, value: int = 1) -> None:
        self.n += int(value)

    def set_postfix_str(self, value: str, refresh: bool = True) -> None:
        del value, refresh

    def set_description_str(self, value: str, refresh: bool = True) -> None:
        del value, refresh

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None


def create_progress_bar(
    *,
    total: int,
    description: str,
    initial: int = 0,
    position: int = 0,
    leave: bool = True,
    enabled: bool = True,
    unit: str = "step",
) -> Any:
    """Create a tqdm bar that works in Jupyter and standard terminals."""

    if not enabled:
        bar = NullProgressBar()
        bar.total = int(total)
        bar.n = int(initial)
        return bar

    try:
        from tqdm.auto import tqdm

        return tqdm(
            total=int(total),
            initial=int(initial),
            desc=str(description),
            position=int(position),
            leave=bool(leave),
            dynamic_ncols=True,
            unit=str(unit),
        )
    except Exception:
        bar = NullProgressBar()
        bar.total = int(total)
        bar.n = int(initial)
        return bar


@dataclass
class TaskProgress:
    """Single compact task bar for a multi-stage notebook workflow."""

    tasks: Iterable[str]
    description: str = "Workflow"
    enabled: bool = True
    leave: bool = True
    position: int = 0
    _task_list: list[str] = field(init=False, repr=False)
    _bar: Any = field(init=False, repr=False)
    _completed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._task_list = [str(task) for task in self.tasks]
        self._bar = create_progress_bar(
            total=len(self._task_list),
            description=self.description,
            position=self.position,
            leave=self.leave,
            enabled=self.enabled,
            unit="task",
        )

    def set_stage(self, stage: str) -> None:
        self._bar.set_postfix_str(str(stage), refresh=True)

    def complete(self, stage: str | None = None) -> None:
        if stage is not None:
            self.set_stage(stage)
        if self._completed < len(self._task_list):
            self._bar.update(1)
            self._completed += 1

    def close(self, stage: str = "Complete") -> None:
        self.set_stage(stage)
        self._bar.close()
