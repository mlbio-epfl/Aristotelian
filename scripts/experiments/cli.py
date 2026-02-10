#!/usr/bin/env python
"""CLI for running experiment sections."""

from __future__ import annotations

import argparse
import inspect
import itertools
import time
from multiprocessing import get_all_start_methods
from pathlib import Path

from loguru import logger

from aristotelian.utils.logging import log_timing, setup_experiment_logging

from .infra.device import mp_context, parse_devices, resolve_device
from .infra.parallel import PARALLEL_SECTIONS
from .registry import SECTION_FUNCS, parse_sections
from .sections import prh as prh_module


def _run_section_task(
    section: str,
    assets_dir_str: str,
    device: str,
    force: bool,
    force_features: bool,
    seed: int | None,
    num_workers: int,
    start_method: str | None,
    prh_metrics: tuple[str, ...],
    prh_q_outlier: float,
    v2t_video_modelset: str,
    v2t_text_modelset: str,
    v2t_num_frames: int,
    v2t_num_captions: int,
) -> tuple[str, float]:
    """Execute a single experiment section."""
    assets_dir = Path(assets_dir_str)
    setup_experiment_logging(f"experiments.section.{section}", level="INFO")
    fn = SECTION_FUNCS[section]
    start_time = time.perf_counter()
    with log_timing(f"Section: {section}"):
        if section in PARALLEL_SECTIONS and num_workers > 1:
            logger.info(f"Using {num_workers} parallel workers")
        sig = inspect.signature(fn)
        kwargs = dict(assets_dir=assets_dir, device=device, force=force, seed=seed)
        if "force_features" in sig.parameters:
            kwargs["force_features"] = force_features
        if "num_workers" in sig.parameters:
            kwargs["num_workers"] = num_workers
        if "start_method" in sig.parameters:
            kwargs["start_method"] = start_method
        if "prh_metrics" in sig.parameters:
            kwargs["prh_metrics"] = prh_metrics
        if "prh_q_outlier" in sig.parameters:
            kwargs["prh_q_outlier"] = prh_q_outlier
        # V2T arguments
        if "v2t_metrics" in sig.parameters:
            kwargs["v2t_metrics"] = prh_metrics  # reuse PRH metrics
        if "v2t_video_modelset" in sig.parameters:
            kwargs["v2t_video_modelset"] = v2t_video_modelset
        if "v2t_text_modelset" in sig.parameters:
            kwargs["v2t_text_modelset"] = v2t_text_modelset
        if "v2t_q_outlier" in sig.parameters:
            kwargs["v2t_q_outlier"] = prh_q_outlier  # reuse PRH outlier quantile
        if "num_frames" in sig.parameters:
            kwargs["num_frames"] = v2t_num_frames
        if "num_captions" in sig.parameters:
            kwargs["num_captions"] = v2t_num_captions
        fn(**kwargs)
    return section, time.perf_counter() - start_time


def _section_worker(
    result_queue,
    section: str,
    assets_dir_str: str,
    device: str,
    force: bool,
    force_features: bool,
    seed: int | None,
    num_workers: int,
    start_method: str | None,
    prh_metrics: tuple[str, ...],
    prh_q_outlier: float,
    v2t_video_modelset: str,
    v2t_text_modelset: str,
    v2t_num_frames: int,
    v2t_num_captions: int,
) -> None:
    """Worker function for parallel section execution."""
    try:
        section_name, duration = _run_section_task(
            section,
            assets_dir_str,
            device,
            force,
            force_features,
            seed,
            num_workers,
            start_method,
            prh_metrics,
            prh_q_outlier,
            v2t_video_modelset,
            v2t_text_modelset,
            v2t_num_frames,
            v2t_num_captions,
        )
        result_queue.put((section_name, duration, None))
    except Exception as exc:
        result_queue.put((section, 0.0, repr(exc)))


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run sgCKA experiment sections and save outputs."
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=["all"],
        help="Sections to run (comma-separated or space-separated).",
    )
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="Directory to store experiment outputs.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute outputs even if files exist."
    )
    parser.add_argument(
        "--force-prh-features",
        action="store_true",
        help="Recompute PRH feature caches even if files exist.",
    )
    parser.add_argument("--device", default=None, help="Torch device (e.g. cpu, cuda).")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for supported sections (default: 1=sequential, 6-8 recommended)",
    )
    parser.add_argument(
        "--mp-start-method",
        choices=sorted(get_all_start_methods()),
        default=None,
        help="Multiprocessing start method. Default uses fork if available; set explicitly for spawn/forkserver.",
    )
    parser.add_argument(
        "--parallel-sections",
        action="store_true",
        help="Run multiple experiment sections in parallel processes.",
    )
    parser.add_argument(
        "--section-workers",
        type=int,
        default=1,
        help="Number of concurrent section workers when --parallel-sections is set.",
    )
    parser.add_argument(
        "--section-devices",
        default=None,
        help="Comma-separated list of devices to assign to sections (round-robin).",
    )
    parser.add_argument(
        "--prh-metrics",
        default=",".join(prh_module.PRH_METRICS),
        help=(
            "Comma-separated PRH alignment metrics (e.g. mutual_knn,cka_lin,"
            "cka_rbf,unbiased_cka,svcca)."
        ),
    )
    parser.add_argument(
        "--prh-q-outlier",
        type=float,
        default=prh_module.PRH_Q_OUTLIER,
        help="Outlier quantile for PRH alignment feature clamping.",
    )
    # V2T (video-to-text) experiment arguments
    parser.add_argument(
        "--v2t-video-modelset",
        default="videoprh",
        choices=[
            "default",
            "small",
            "videomae",
            "dinov2",
            "clip",
            "extended",
            "videoprh",
        ],
        help="Video model set for V2T experiment (default: videoprh).",
    )
    parser.add_argument(
        "--v2t-text-modelset",
        default="videoprh",
        help="Text model set for V2T experiment (default: videoprh).",
    )
    parser.add_argument(
        "--v2t-num-frames",
        type=int,
        default=16,
        help="Number of frames to extract per video for V2T experiment (default: 16).",
    )
    parser.add_argument(
        "--v2t-num-captions",
        type=int,
        default=1,
        help="Number of captions to use per video for V2T experiment (default: 1).",
    )
    args = parser.parse_args()

    prh_metrics = tuple(m.strip() for m in args.prh_metrics.split(",") if m.strip())
    prh_q_outlier = float(args.prh_q_outlier)

    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    # Setup logging
    setup_experiment_logging("experiments.cli", level="INFO")

    # Log experiment start with configuration
    logger.info("=" * 80)
    logger.info("Experiment Suite: scripts/experiments/cli.py")
    logger.info("=" * 80)
    logger.info(f"Assets directory: {assets_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Force recompute: {args.force}")
    logger.info(f"Force PRH features: {args.force_prh_features}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info(f"MP start method: {args.mp_start_method or 'auto'}")
    logger.info("=" * 80)

    sections = parse_sections(args.sections)
    logger.info(f"Running {len(sections)} sections: {', '.join(sections)}")

    start_time = time.perf_counter()

    if args.parallel_sections and args.section_workers > 1 and len(sections) > 1:
        devices = parse_devices(args.section_devices, device)
        if args.section_devices and devices == [device]:
            logger.warning(
                "Requested section devices were unavailable; "
                f"falling back to {device}"
            )
        logger.info(
            "Parallel sections enabled "
            f"(workers={args.section_workers}, devices={', '.join(devices)})"
        )
        ctx = mp_context(device, args.mp_start_method)
        device_cycle = itertools.cycle(devices)
        jobs = [
            (
                section,
                str(assets_dir),
                next(device_cycle),
                args.force,
                args.force_prh_features,
                args.seed,
                args.num_workers,
                args.mp_start_method,
                prh_metrics,
                prh_q_outlier,
                args.v2t_video_modelset,
                args.v2t_text_modelset,
                args.v2t_num_frames,
                args.v2t_num_captions,
            )
            for section in sections
        ]
        result_queue = ctx.Queue()
        active = []
        errors = []
        job_iter = iter(jobs)

        def _start_next() -> None:
            try:
                job = next(job_iter)
            except StopIteration:
                return
            proc = ctx.Process(target=_section_worker, args=(result_queue, *job))
            proc.daemon = False
            proc.start()
            active.append(proc)

        for _ in range(min(args.section_workers, len(jobs))):
            _start_next()

        completed = 0
        while active:
            section, duration, error = result_queue.get()
            completed += 1
            if error:
                logger.error(f"Section failed: {section} ({error})")
                errors.append((section, error))
            else:
                logger.success(f"Section finished: {section} ({duration:.2f}s)")
            for proc in list(active):
                if not proc.is_alive():
                    proc.join()
                    active.remove(proc)
            _start_next()

        if errors:
            raise RuntimeError(
                "One or more sections failed: "
                + ", ".join(section for section, _ in errors)
            )
    else:
        for idx, section in enumerate(sections, 1):
            fn = SECTION_FUNCS[section]
            logger.info(f"[{idx}/{len(sections)}] Running section: {section}")

            with log_timing(f"Section: {section}"):
                if section in PARALLEL_SECTIONS and args.num_workers > 1:
                    logger.info(f"Using {args.num_workers} parallel workers")
                kwargs = dict(
                    device=device,
                    force=args.force,
                    seed=args.seed,
                )
                sig = inspect.signature(fn)
                if "force_features" in sig.parameters:
                    kwargs["force_features"] = args.force_prh_features
                if "num_workers" in sig.parameters:
                    kwargs["num_workers"] = args.num_workers
                if "start_method" in sig.parameters:
                    kwargs["start_method"] = args.mp_start_method
                if "prh_metrics" in sig.parameters:
                    kwargs["prh_metrics"] = prh_metrics
                if "prh_q_outlier" in sig.parameters:
                    kwargs["prh_q_outlier"] = prh_q_outlier
                # V2T arguments
                if "v2t_metrics" in sig.parameters:
                    kwargs["v2t_metrics"] = prh_metrics  # reuse PRH metrics
                if "v2t_video_modelset" in sig.parameters:
                    kwargs["v2t_video_modelset"] = args.v2t_video_modelset
                if "v2t_text_modelset" in sig.parameters:
                    kwargs["v2t_text_modelset"] = args.v2t_text_modelset
                if "v2t_q_outlier" in sig.parameters:
                    kwargs["v2t_q_outlier"] = prh_q_outlier  # reuse PRH outlier
                if "num_frames" in sig.parameters:
                    kwargs["num_frames"] = args.v2t_num_frames
                if "num_captions" in sig.parameters:
                    kwargs["num_captions"] = args.v2t_num_captions
                fn(assets_dir, **kwargs)

    elapsed = time.perf_counter() - start_time
    logger.success(f"All {len(sections)} sections completed in {elapsed:.2f}s")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
