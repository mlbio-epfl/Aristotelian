"""Centralized logging configuration for experiments.

This module provides:
- Structured logging setup with loguru
- tqdm integration without visual conflicts
- Timing utilities for operations
- Uniform experiment state definitions
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from enum import Enum
from functools import partial

from loguru import logger
from tqdm import tqdm as tqdm_original


class ExperimentState(str, Enum):
    """Standard experiment states for uniform logging."""

    INITIALIZING = "Initializing"
    PARSING_ARGS = "Parsing arguments"
    SETTING_UP = "Setting up environment"
    LOADING_DATA = "Loading data"
    GENERATING_DATA = "Generating synthetic data"
    COMPUTING = "Computing"
    COMPUTING_PERMUTATIONS = "Computing permutation null"
    COMPUTING_SIMILARITIES = "Computing similarity matrix"
    COMPUTING_METRICS = "Computing metrics"
    AGGREGATING = "Aggregating results"
    SAVING_RESULTS = "Saving results"
    ANALYZING = "Analyzing results"
    GENERATING_PLOTS = "Generating plots"
    COMPLETED = "Completed"
    FAILED = "Failed"
    SKIPPED = "Skipped"


def setup_experiment_logging(name: str, level: str = "INFO") -> None:
    """
    Configure loguru for console-only experiment logging.

    This function sets up structured logging with:
    - Console output (stderr) with colors
    - Clean, human-readable format
    - Configurable log level

    Args:
        name: Experiment name (for identification in logs)
        level: Minimum log level ("TRACE"|"DEBUG"|"INFO"|"WARNING"|"ERROR")

    Example:
        >>> setup_experiment_logging("my_exp", level="INFO")
        >>> logger.info("Experiment started")

    Note:
        - Call this ONCE at the start of your experiment
        - All subsequent logger.* calls will use this configuration
        - Workers in multiprocessing will inherit this configuration
    """
    # Remove default handler to avoid duplicates
    logger.remove()

    # Console handler with color and clean format
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
        colorize=True,
    )

    logger.debug(f"Logging configured for experiment: {name}")


def get_loguru_safe_tqdm():
    """
    Returns a tqdm wrapper compatible with loguru.

    This ensures tqdm progress bars don't conflict with logger output by:
    - Forcing tqdm to use stderr (same as loguru console handler)
    - Preventing progress bar corruption

    Returns:
        Partial function that wraps tqdm with file=sys.stderr

    Example:
        >>> tqdm = get_loguru_safe_tqdm()
        >>> for item in tqdm(items, desc="Processing"):
        ...     process(item)

    Note:
        Replace standard tqdm imports with this in experiment scripts:
        from aristotelian.utils.logging import get_loguru_safe_tqdm
        tqdm = get_loguru_safe_tqdm()
    """
    return partial(tqdm_original, file=sys.stderr)


@contextmanager
def log_timing(operation: str, level: str = "INFO"):
    """
    Context manager for timing operations with automatic logging.

    Logs:
    - Start of operation
    - Completion with elapsed time
    - Failure with elapsed time (if exception occurs)

    Args:
        operation: Description of the operation being timed
        level: Log level for start message ("INFO"|"DEBUG")

    Example:
        >>> with log_timing("Data generation"):
        ...     X, Y = generate_data(n=1000, d=100)
        # Logs: "Starting: Data generation"
        #       "Completed: Data generation (took 1.23s)"

    Note:
        - Exceptions are re-raised after logging
        - Use level="DEBUG" for frequent/minor operations
    """
    start = time.perf_counter()
    logger.log(level.upper(), f"Starting: {operation}")
    try:
        yield
        elapsed = time.perf_counter() - start
        logger.success(f"Completed: {operation} (took {elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Failed: {operation} after {elapsed:.2f}s - {e}")
        raise
