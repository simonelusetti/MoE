import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm


def configure_runtime(cfg):
    runtime = cfg.runtime
    num_threads = runtime.num_threads
    if not num_threads:
        return
    try:
        num_threads = int(num_threads)
    except (TypeError, ValueError):
        return
    if num_threads <= 0:
        return
    value = str(num_threads)
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = value
    try:
        import torch

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except Exception:
        pass


def should_disable_tqdm(*, metrics_only=False):
    """Return True when tqdm progress bars should be disabled."""
    if metrics_only:
        return True

    override = os.environ.get("EXPERT_MOE_DISABLE_TQDM")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}

    try:
        return not sys.stderr.isatty()
    except Exception:
        return True


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


def get_logger(logfile="train.log"):
    logger = logging.getLogger("expert_moe")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    stream_handler = TqdmLoggingHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    file_handler = logging.FileHandler(Path(logfile))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
