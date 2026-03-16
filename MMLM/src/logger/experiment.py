class NullLogger:
    """No-op experiment logger."""

    def start(self) -> None:
        """Open the logging session."""

    def log(self, payload: dict, step: int) -> None:
        """Log a payload."""

    def finish(self) -> None:
        """Close the logging session."""


class SwanLabLogger:
    """Thin wrapper around swanlab with lazy import."""

    def __init__(self, project: str, experiment_name: str, config: dict | None = None, enabled: bool = True):
        self.project = project
        self.experiment_name = experiment_name
        self.config = config or {}
        self.enabled = enabled
        self.run = None

    def start(self) -> None:
        """Initialize a swanlab run when enabled."""
        if not self.enabled:
            return
        import swanlab

        self.run = swanlab.init(
            project=self.project,
            experiment_name=self.experiment_name,
            config=self.config,
        )

    def log(self, payload: dict, step: int) -> None:
        """Log one step to swanlab."""
        if self.run is None:
            return
        self.run.log(payload, step=step)

    def finish(self) -> None:
        """Finish the swanlab run."""
        if self.run is not None:
            self.run.finish()
            self.run = None
