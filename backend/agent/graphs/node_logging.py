"""Small logging seam for graph node timing and failures."""

from __future__ import annotations

import inspect
import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger("backend.agent.graphs.nodes")


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": sorted(state.keys()),
        "document_count": len(state.get("document_ids", [])),
        "history_count": len(state.get("history", [])),
        "retry_count": state.get("retry_count"),
    }


def log_node(graph_name: str, node_name: str) -> Callable:
    """Log start, duration, output keys, and traceback for one graph node."""

    def decorator(function: Callable) -> Callable:
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapper(state: dict[str, Any]) -> Any:
                started = time.perf_counter()
                logger.info(
                    "[graph_node:start] graph=%s node=%s state=%s",
                    graph_name,
                    node_name,
                    _state_summary(state),
                )
                try:
                    result = await function(state)
                    logger.info(
                        "[graph_node:success] graph=%s node=%s duration_ms=%.1f output_keys=%s",
                        graph_name,
                        node_name,
                        (time.perf_counter() - started) * 1000,
                        sorted(result.keys()) if isinstance(result, dict) else [],
                    )
                    return result
                except Exception:
                    logger.exception(
                        "[graph_node:error] graph=%s node=%s duration_ms=%.1f",
                        graph_name,
                        node_name,
                        (time.perf_counter() - started) * 1000,
                    )
                    raise

            return async_wrapper

        @wraps(function)
        def sync_wrapper(state: dict[str, Any]) -> Any:
            started = time.perf_counter()
            logger.info(
                "[graph_node:start] graph=%s node=%s state=%s",
                graph_name,
                node_name,
                _state_summary(state),
            )
            try:
                result = function(state)
                logger.info(
                    "[graph_node:success] graph=%s node=%s duration_ms=%.1f output_keys=%s",
                    graph_name,
                    node_name,
                    (time.perf_counter() - started) * 1000,
                    sorted(result.keys()) if isinstance(result, dict) else [],
                )
                return result
            except Exception:
                logger.exception(
                    "[graph_node:error] graph=%s node=%s duration_ms=%.1f",
                    graph_name,
                    node_name,
                    (time.perf_counter() - started) * 1000,
                )
                raise

        return sync_wrapper

    return decorator