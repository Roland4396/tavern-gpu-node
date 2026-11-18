#!/home/worker/.venvs/deep-learning/bin/python
import os
import runpy
import stat
import sys
from pathlib import Path


REPO = Path(os.environ.get("GRAPH_REPO", "/home/worker/.local/share/graph-stack/core"))
SOCKET_PATH = Path(os.environ["GRAPH_SOCKET"])


SOCKET_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
if SOCKET_PATH.exists():
    if not stat.S_ISSOCK(SOCKET_PATH.lstat().st_mode):
        raise RuntimeError(f"refusing to replace non-socket path: {SOCKET_PATH}")
    SOCKET_PATH.unlink()
os.umask(0o077)

# Native MP4/ZIP writers need POSIX seek/reopen semantics. The original nodes
# run unchanged; completed files are published before their UI events.
spool = os.environ.get("GRAPH_OUTPUT_SPOOL", "")
if spool:
    Path(spool).mkdir(mode=0o700, parents=True, exist_ok=True)
    if "--output-directory" in sys.argv:
        sys.argv[sys.argv.index("--output-directory") + 1] = spool
    else:
        sys.argv.extend(["--output-directory", spool])

from aiohttp import web


class PrivateUnixSite(web.UnixSite):
    def __init__(self, runner, host=None, port=None, *, shutdown_timeout=60.0, ssl_context=None, backlog=128, **_ignored):
        super().__init__(runner, str(SOCKET_PATH), shutdown_timeout=shutdown_timeout, ssl_context=ssl_context, backlog=backlog)

    async def start(self):
        await super().start()
        SOCKET_PATH.chmod(0o600)


web.TCPSite = PrivateUnixSite
os.chdir(REPO)
sys.path.insert(0, str(REPO))
if os.environ.get("GRAPH_H3_ORIGINAL_MANIFEST"):
    from h3_original_bootstrap import install_graph_startup
    install_graph_startup(os.environ["GRAPH_H3_ORIGINAL_MANIFEST"])
runpy.run_path(str(REPO / "main.py"), run_name="__main__")
