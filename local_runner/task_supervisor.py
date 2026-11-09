"""Wall-clock supervision of an agent process tree, with a startup handshake.

Only controlled bootstrap programs which wait for stdin are supported. No model
or tool work begins until containment is installed. Deadline + OS cleanup latency
is not a hard-real-time guarantee. Remote API billing cannot be cancelled locally.
"""
from __future__ import annotations
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

class WindowsJob:
    def __init__(self):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),
                      ('flags',w.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),
                      ('active_limit',w.DWORD),('affinity',ctypes.c_size_t),
                      ('priority',w.DWORD),('scheduling',w.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(n,ctypes.c_ulonglong) for n in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_=[('basic',Basic),('io',IO),('process_mem',ctypes.c_size_t),
                      ('job_mem',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype=w.HANDLE
        self.kernel.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
        self.kernel.SetInformationJobObject.restype=w.BOOL
        self.kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD]
        self.kernel.OpenProcess.restype=w.HANDLE
        self.kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
        self.kernel.AssignProcessToJobObject.restype=w.BOOL
        self.kernel.CloseHandle.argtypes=[w.HANDLE]
        self.kernel.CloseHandle.restype=w.BOOL
        self.kernel.TerminateJobObject.argtypes=[w.HANDLE,w.UINT]
        self.kernel.TerminateJobObject.restype=w.BOOL
        self.kernel.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype=w.BOOL
        self.kernel.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD]
        self.kernel.WaitForSingleObject.restype=w.DWORD
        self.handle=self.kernel.CreateJobObjectW(None,None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info=Extended()
        info.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):
            error=ctypes.get_last_error(); self.close()
            raise ctypes.WinError(error)

    def assign(self,pid):
        process=self.kernel.OpenProcess(0x0101,False,pid)  # SET_QUOTA | TERMINATE
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle,process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(process)

    def close(self):
        from ctypes import wintypes as w
        class Accounting(ctypes.Structure):
            _fields_=[('user',ctypes.c_longlong),('kernel',ctypes.c_longlong),
                      ('period_user',ctypes.c_longlong),('period_kernel',ctypes.c_longlong),
                      ('faults',w.DWORD),('total',w.DWORD),('active',w.DWORD),('terminated',w.DWORD)]
        confirmed=True
        if self.handle:
            class Pids(ctypes.Structure):
                _fields_=[('assigned',w.DWORD),('count',w.DWORD),('ids',ctypes.c_size_t*256)]
            pids=Pids(); process_handles=[]
            enumerated=bool(self.kernel.QueryInformationJobObject(self.handle,3,ctypes.byref(pids),ctypes.sizeof(pids),None))
            if enumerated:
                for pid in list(pids.ids)[:pids.count]:
                    process=self.kernel.OpenProcess(0x100000,False,pid)  # SYNCHRONIZE
                    if process:
                        process_handles.append(process)
            self.kernel.TerminateJobObject(self.handle,1)
            until=time.monotonic()+5
            confirmed=False
            try:
                waited=enumerated
                for process in process_handles:
                    remaining=max(0,int((until-time.monotonic())*1000))
                    if self.kernel.WaitForSingleObject(process,remaining)!=0:
                        waited=False
                while time.monotonic()<until:
                    info=Accounting()
                    if not self.kernel.QueryInformationJobObject(self.handle,1,ctypes.byref(info),ctypes.sizeof(info),None):
                        break
                    if info.active==0:
                        confirmed=waited
                        break
                    time.sleep(.01)
            finally:
                for process in process_handles:
                    self.kernel.CloseHandle(process)
                self.kernel.CloseHandle(self.handle)
                self.handle=None
        return confirmed

def supervise(command: list[str], request: dict, *, timeout_s: float,
              log_path: Path, cwd: Path | None=None) -> dict:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('positive finite task deadline required')
    # Exclusive creation: previous traces/logs are never silently reused.
    started=time.monotonic()
    job=None; child=None; state='runtime_error'; cleanup_confirmed=False
    with log_path.open('x',encoding='utf-8') as log:
        try:
            if os.name=='nt':
                job=WindowsJob()
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log,
                text=True,encoding='utf-8',cwd=cwd,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                start_new_session=os.name!='nt')
            if job:
                job.assign(child.pid)
            remaining=timeout_s-(time.monotonic()-started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command,timeout_s)
            # Controlled child must not perform work before this message.
            child.communicate(json.dumps(request),timeout=remaining)
            state='process_completed' if child.returncode==0 else 'runtime_error'
        except subprocess.TimeoutExpired:
            state='task_deadline_exceeded'
        finally:
            # Containment is closed even after successful parent exit: any
            # surviving descendants are part of this task, not background work.
            if job:
                cleanup_confirmed=job.close()
            elif child is not None:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.communicate()
    if os.name=='nt' and not cleanup_confirmed:
        state='cleanup_unconfirmed'
    return {'supervisor_status':state,'exit_code':child.returncode if child else None,
            'deadline_s':timeout_s,'elapsed_s':time.monotonic()-started,
            'process_tree_cleanup_requested':True,
            'process_tree_cleanup_confirmed':cleanup_confirmed,
            'billing_note':'Local termination cannot revoke remote requests already accepted.'}
