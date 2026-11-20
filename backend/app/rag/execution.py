"""Short-lived in-process progress for deterministic workflow runs."""

from threading import RLock


class ExecutionTracker:
    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, dict] = {}

    def start(self, execution_id: str, node_ids: list[str], completed: list[str] | None = None) -> None:
        statuses = {node_id: "queued" for node_id in node_ids}
        for node_id in completed or []:
            if node_id in statuses:
                statuses[node_id] = "completed"
        with self._lock:
            self._runs[execution_id] = {
                "execution_id": execution_id,
                "status": "running",
                "node_statuses": statuses,
                "error": None,
            }
            while len(self._runs) > 64:
                oldest = next(iter(self._runs))
                if oldest == execution_id:
                    break
                self._runs.pop(oldest, None)

    def node(self, execution_id: str, node_id: str, status: str) -> None:
        with self._lock:
            run = self._runs.get(execution_id)
            if run and node_id in run["node_statuses"]:
                run["node_statuses"][node_id] = status

    def finish(self, execution_id: str) -> None:
        with self._lock:
            run = self._runs.get(execution_id)
            if run:
                run["status"] = "succeeded"
                for node_id, status in run["node_statuses"].items():
                    if status in {"queued", "running"}:
                        run["node_statuses"][node_id] = "skipped"

    def fail(self, execution_id: str, message: str) -> None:
        with self._lock:
            run = self._runs.get(execution_id)
            if run:
                run["status"] = "failed"
                run["error"] = message
                running = [
                    node_id
                    for node_id, status in run["node_statuses"].items()
                    if status == "running"
                ]
                if running:
                    run["node_statuses"][running[-1]] = "failed"
                for node_id, status in run["node_statuses"].items():
                    if status == "queued":
                        run["node_statuses"][node_id] = "skipped"

    def get(self, execution_id: str) -> dict | None:
        with self._lock:
            run = self._runs.get(execution_id)
            if run is None:
                return None
            return {**run, "node_statuses": dict(run["node_statuses"])}


execution_tracker = ExecutionTracker()
