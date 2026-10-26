"""An engine worker's process: it ends with the process that started it, however that ends."""

import os
import signal
import subprocess
import sys
import time

PARENT = """
import os, sys
from eicvibe.engines.worker import EngineWorker

if __name__ == "__main__":
    worker = EngineWorker("xsuite")
    print(worker._submit(os.getpid).result(), flush=True)
    sys.stdin.read()  # until it is killed
"""


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_worker_ends_when_the_process_that_started_it_is_killed(tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    parent = subprocess.Popen(
        [sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    worker = None
    try:
        worker = int(parent.stdout.readline())
        assert not gone(worker)
        parent.send_signal(signal.SIGKILL)  # no exit handlers: as a kernel killed on restart
        parent.wait(10)
        for _ in range(100):
            if gone(worker):
                break
            time.sleep(0.05)
        assert gone(worker)
    finally:
        parent.kill()
        parent.wait(10)
        if worker is not None and not gone(worker):  # the test failed: leave nothing behind
            os.kill(worker, signal.SIGKILL)
