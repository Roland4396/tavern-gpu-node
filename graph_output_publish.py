"""Publish completed native Graph files from POSIX spool to object storage.

No encoding/remuxing, workflow changes or replacement save-node algorithms.
UI events are forwarded only after their files are published. A final success
publishes intermediate outputs as well. Existing image bytes stay unchanged.
"""
import functools
import os
from pathlib import Path
import shutil
import threading
import uuid


class OutputPublisher:
    def __init__(self, spool, destination):
        self.spool=Path(spool).resolve();self.destination=Path(destination).resolve()
        if self.spool==self.destination:raise ValueError('Native spool must be separate from shared output')
        self.spool.mkdir(parents=True,exist_ok=True)
        self.destination.mkdir(parents=True,exist_ok=True)
        self.published={};self.lock=threading.Lock()

    def _copy(self, path):
        path=Path(path)
        relative=path.resolve().relative_to(self.spool)
        if not path.is_file() or path.is_symlink():return
        st=path.stat();version=(st.st_mtime_ns,st.st_size)
        if self.published.get(relative)==version:return
        target=self.destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.partial')
        try:
            shutil.copyfile(path,temporary)
            os.replace(temporary,target)
            self.published[relative]=version
        finally:
            temporary.unlink(missing_ok=True)

    def ui(self, value):
        with self.lock:
            def visit(v):
                if isinstance(v,dict):
                    if v.get('type')=='output' and isinstance(v.get('filename'),str):
                        self._copy(self.spool/v.get('subfolder','')/v['filename'])
                        if isinstance(v.get('workflow'),str):
                            self._copy(self.spool/v.get('subfolder','')/v['workflow'])
                    for x in v.values():visit(x)
                elif isinstance(v,(tuple,list)):
                    for x in v:visit(x)
            visit(value)

    def all(self):
        with self.lock:
            for p in self.spool.rglob('*'):
                if p.is_file() and not any(x.startswith('.') for x in p.relative_to(self.spool).parts):
                    self._copy(p)


def install_output_publisher(server, spool, destination):
    publisher=OutputPublisher(spool,destination)
    original=server.send_sync
    @functools.wraps(original)
    def send(event,data,*args,**kwargs):
        if event=='executed':publisher.ui(data.get('output',{}))
        elif event=='execution_success':publisher.all()
        return original(event,data,*args,**kwargs)
    server.send_sync=send
    return publisher
