"""Tests for logging configuration module."""

import time

import pytest
from loguru import logger

from aristotelian.utils.logging import (
    ExperimentState,
    get_loguru_safe_tqdm,
    log_timing,
    setup_experiment_logging,
)


def test_setup_logging(capsys):
    """Test basic logging setup writes to stderr."""
    setup_experiment_logging("test", level="DEBUG")
    logger.info("test message")
    logger.debug("debug message")
    captured = capsys.readouterr()
    assert "test message" in captured.err
    assert "debug message" in captured.err


def test_setup_logging_info_level(capsys):
    """Test INFO level filters DEBUG messages."""
    setup_experiment_logging("test_info", level="INFO")
    logger.debug("hidden debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.success("success message")
    captured = capsys.readouterr()
    assert "info message" in captured.err
    assert "warning message" in captured.err
    assert "error message" in captured.err
    assert "success message" in captured.err
    assert "hidden debug message" not in captured.err


def test_log_timing_success(capsys):
    """Test timing context manager logs start and completion."""
    setup_experiment_logging("test_timing", level="INFO")
    with log_timing("test operation"):
        time.sleep(0.01)
    captured = capsys.readouterr()
    assert "Starting: test operation" in captured.err
    assert "Completed: test operation" in captured.err


def test_log_timing_failure(capsys):
    """Test timing context manager logs failure and re-raises."""
    setup_experiment_logging("test_timing_fail", level="INFO")
    with pytest.raises(ValueError):
        with log_timing("failing operation"):
            raise ValueError("intentional error")
    captured = capsys.readouterr()
    assert "Starting: failing operation" in captured.err
    assert "Failed: failing operation" in captured.err


def test_experiment_state_enum():
    """Test experiment state enum."""
    assert ExperimentState.INITIALIZING == "Initializing"
    assert ExperimentState.COMPLETED == "Completed"
    assert ExperimentState.COMPUTING == "Computing"
    assert ExperimentState.SAVING_RESULTS == "Saving results"
    assert ExperimentState.FAILED == "Failed"
    assert ExperimentState.SKIPPED == "Skipped"

    # Test that all states are strings
    for state in ExperimentState:
        assert isinstance(state.value, str)


def test_tqdm_wrapper():
    """Test tqdm wrapper."""
    setup_experiment_logging("test_tqdm", level="INFO")
    tqdm = get_loguru_safe_tqdm()

    items = list(range(10))
    result = []

    for item in tqdm(items, desc="Test"):
        result.append(item)

    assert result == items


def test_tqdm_wrapper_with_logging():
    """Test tqdm wrapper works alongside logging."""
    setup_experiment_logging("test_tqdm_logging", level="INFO")
    tqdm = get_loguru_safe_tqdm()

    items = list(range(5))
    result = []

    for item in tqdm(items, desc="Processing"):
        logger.info(f"Processing item {item}")
        result.append(item)

    assert result == items


def test_log_timing_with_debug_level(capsys):
    """Test timing with DEBUG level writes to stderr."""
    setup_experiment_logging("test_timing_debug", level="DEBUG")
    with log_timing("debug operation", level="DEBUG"):
        time.sleep(0.01)
    captured = capsys.readouterr()
    assert "Starting: debug operation" in captured.err
    assert "Completed: debug operation" in captured.err


def test_multiple_setup_calls():
    """Test that calling setup_experiment_logging replaces handlers."""
    setup_experiment_logging("test1", level="INFO")
    logger.info("test1 message")

    setup_experiment_logging("test2", level="DEBUG")
    logger.debug("test2 message")
    assert len(logger._core.handlers) == 1
