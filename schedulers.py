from dataclasses import dataclass
from collections import deque
from typing import List, Tuple

@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    priority: int = 0
    remaining: int = None

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.burst

def fifo_scheduler(processes):
    time_ptr = 0
    tl = []
    for p in sorted(processes, key=lambda x: x.arrival):
        start = max(time_ptr, p.arrival)
        end = start + p.burst
        tl.append((p.pid, start, end))
        time_ptr = end
    return tl

def sjf_scheduler(procs: list[Process]) -> list[tuple]:
    """SJF no-preemptivo."""
    time = 0
    timeline = []
    remaining = {p.pid: p for p in procs}
    done = set()
    while len(done) < len(procs):
        ready = [p for p in procs if p.arrival <= time and p.pid not in done]
        if ready:
            p = min(ready, key=lambda x: x.burst)
            start = max(time, p.arrival)
            end = start + p.burst
            timeline.append((p.pid, start, end))
            time = end
            done.add(p.pid)
        else:
            time += 1
    return timeline

def srt_scheduler(procs: list[Process]) -> list[tuple]:
    """SRT preemptivo."""
    time = 0
    rem = {p.pid: p.burst for p in procs}
    timeline = []
    current, start = None, 0

    while True:
        arrived = [p for p in procs if p.arrival <= time and rem[p.pid] > 0]
        if arrived:
            p = min(arrived, key=lambda x: rem[x.pid])
            if current != p.pid:
                if current is not None:
                    timeline.append((current, start, time))
                current, start = p.pid, time
            rem[p.pid] -= 1
        else:
            # si no hay ready y todo está hecho, cortamos
            if all(v == 0 for v in rem.values()):
                break
        time += 1

    if current is not None:
        timeline.append((current, start, time))
    return timeline

def rr_scheduler(procs: list[Process], quantum: int) -> list[tuple]:
    """Round Robin preemptivo."""
    time = 0
    queue = deque()
    timeline = []
    rem = {p.pid: p.burst for p in procs}
    sorted_procs = sorted(procs, key=lambda x: x.arrival)
    idx = 0

    while True:
        # encolar llegadas
        while idx < len(sorted_procs) and sorted_procs[idx].arrival <= time:
            queue.append(sorted_procs[idx].pid)
            idx += 1

        if not queue:
            if idx < len(sorted_procs):
                time = sorted_procs[idx].arrival
                continue
            else:
                break

        pid = queue.popleft()
        slice_len = min(quantum, rem[pid])
        start, end = time, time + slice_len
        timeline.append((pid, start, end))
        rem[pid] -= slice_len
        time = end

        # encolar nuevas llegadas durante el slice
        while idx < len(sorted_procs) and sorted_procs[idx].arrival <= time:
            queue.append(sorted_procs[idx].pid)
            idx += 1
        if rem[pid] > 0:
            queue.append(pid)

    return timeline

def priority_scheduler(procs: list[Process], preemptive: bool) -> list[tuple]:
    """Priority, preemptivo o no."""
    time = 0
    rem = {p.pid: p.burst for p in procs}
    prio = {p.pid: p.priority for p in procs}
    timeline = []
    current, start = None, 0

    while True:
        ready = [p for p in procs if p.arrival <= time and rem[p.pid] > 0]
        if ready:
            p = min(ready, key=lambda x: prio[x.pid])
            if preemptive:
                if current != p.pid:
                    if current is not None:
                        timeline.append((current, start, time))
                    current, start = p.pid, time
                rem[p.pid] -= 1
                time += 1
            else:
                # run to completion
                st = max(time, p.arrival)
                en = st + rem[p.pid]
                timeline.append((p.pid, st, en))
                time = en
                rem[p.pid] = 0
        else:
            if all(v == 0 for v in rem.values()):
                break
            # avanzar al siguiente arribo
            next_arrival = min(p.arrival for p in procs if rem[p.pid] > 0)
            time = next_arrival

    if preemptive and current is not None:
        timeline.append((current, start, time))
    return timeline