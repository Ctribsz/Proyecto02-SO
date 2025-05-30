from dataclasses import dataclass
from collections import deque
import pandas as pd

@dataclass
class Event:
    pid: str
    time: int
    action: str      
    resource: str

class Mutex:
    def __init__(self, name):
        self.name = name
        self.locked_by = None
        self.queue = deque()

    def acquire(self, pid, now, timeline):
        if self.locked_by is None:
            self.locked_by = pid
            timeline.append((now, pid, "acquired", self.name))
        else:
            self.queue.append(pid)
            timeline.append((now, pid, "waiting", self.name))

    def release(self, pid, now, timeline):
        if self.locked_by != pid:
            raise RuntimeError(f"{pid} libera un mutex que no posee")
        if self.queue:
            nxt = self.queue.popleft()
            self.locked_by = nxt
            timeline.append((now, nxt, "acquired", self.name))
        else:
            self.locked_by = None
        timeline.append((now, pid, "released", self.name))

class Semaphore:
    def __init__(self, name, initial):
        self.name = name
        self.count = initial
        self.queue = deque()

    def acquire(self, pid, now, timeline):
        if self.count > 0:
            self.count -= 1
            timeline.append((now, pid, "acquired", self.name))
        else:
            self.queue.append(pid)
            timeline.append((now, pid, "waiting", self.name))

    def release(self, pid, now, timeline):
        # para semáforos generalmente no se valida owner
        if self.queue:
            nxt = self.queue.popleft()
            timeline.append((now, nxt, "acquired", self.name))
        else:
            self.count += 1
        timeline.append((now, pid, "released", self.name))

def simulate_sync(df_events, mode, sem_init=None):
    """
    df_events: DataFrame con columnas pid,time,action,resource,[initial]
    mode: "Mutex" o "Semaphore"
    sem_init: dict{name:count} para semáforos
    """
    # 1) crear instancias por recurso
    resources = {}
    timeline = []   # lista de (time,pid,status,resource)

    # si es semáforo, leemos conteos iniciales
    if mode == "Semaphore":
        for name, cnt in sem_init.items():
            resources[name] = Semaphore(name, cnt)

    # 2) ordenar eventos
    evs = df_events.sort_values("time").itertuples()

    # 3) procesar uno a uno
    for ev in evs:
        rsrc = ev.resource
        if rsrc not in resources:
            # creamos mutex o semáforo según mode
            if mode == "Mutex":
                resources[rsrc] = Mutex(rsrc)
            else:
                raise ValueError(f"Falta count para semáforo {rsrc}")
        res = resources[rsrc]

        if ev.action == "acquire":
            res.acquire(ev.pid, ev.time, timeline)
        else:
            res.release(ev.pid, ev.time, timeline)

    # devolvemos el timeline de sincronización
    return pd.DataFrame(timeline, columns=["time","pid","status","resource"])