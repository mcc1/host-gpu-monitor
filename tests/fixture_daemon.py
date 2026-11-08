"""Use the production CLI options with a synthetic NVIDIA backend."""

import sys
from pathlib import Path
from unittest.mock import patch

from host_gpu_monitor import daemon
from host_gpu_monitor.__main__ import main
from host_gpu_monitor.sampler import Sampler


async def read_xml():
    return Path(__file__).with_name("fixture.xml").read_bytes()


if __name__ == "__main__":
    with patch.object(daemon, "Sampler", lambda: Sampler(read_xml)):
        raise SystemExit(main(sys.argv[1:]))
