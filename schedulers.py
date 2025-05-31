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
    """FIFO/FCFS - First In First Out"""
    time_ptr = 0
    timeline = []
    for p in sorted(processes, key=lambda x: x.arrival):
        start = max(time_ptr, p.arrival)
        end = start + p.burst
        timeline.append((p.pid, start, end))
        time_ptr = end
    return timeline

def sjf_scheduler(procs: list[Process]) -> list[tuple]:
    """SJF no-preemptivo - Shortest Job First"""
    time = 0
    timeline = []
    remaining = procs.copy()  # Procesos pendientes
    
    while remaining:
        # Filtrar procesos que ya han llegado
        ready = [p for p in remaining if p.arrival <= time]
        
        if ready:
            # Seleccionar el de menor burst time
            shortest = min(ready, key=lambda x: x.burst)
            start = max(time, shortest.arrival)
            end = start + shortest.burst
            timeline.append((shortest.pid, start, end))
            time = end
            remaining.remove(shortest)
        else:
            # Si no hay procesos listos, avanzar al siguiente arrival
            next_arrival = min(p.arrival for p in remaining)
            time = next_arrival
    
    return timeline

def srt_scheduler(procs: list[Process]) -> list[tuple]:
    """SRT preemptivo - Shortest Remaining Time"""
    time = 0
    remaining = {p.pid: p.burst for p in procs}
    timeline = []
    current_process = None
    current_start = 0

    while any(remaining.values()):
        # Procesos que han llegado y aún tienen tiempo restante
        ready = [p for p in procs if p.arrival <= time and remaining[p.pid] > 0]
        
        if ready:
            # Seleccionar el de menor tiempo restante
            shortest = min(ready, key=lambda x: remaining[x.pid])
            
            # Si cambia el proceso en ejecución
            if current_process != shortest.pid:
                if current_process is not None:
                    timeline.append((current_process, current_start, time))
                current_process = shortest.pid
                current_start = time
            
            # Ejecutar por una unidad de tiempo
            remaining[shortest.pid] -= 1
            time += 1
        else:
            # Si no hay procesos listos, avanzar al siguiente arrival
            if any(remaining.values()):
                next_arrival = min(p.arrival for p in procs if remaining[p.pid] > 0)
                time = next_arrival

    # Agregar el último segmento si existe
    if current_process is not None:
        timeline.append((current_process, current_start, time))
    
    return timeline

def rr_scheduler(procs: list[Process], quantum: int) -> list[tuple]:
    """Round Robin preemptivo"""
    time = 0
    queue = deque()
    timeline = []
    remaining = {p.pid: p.burst for p in procs}
    sorted_procs = sorted(procs, key=lambda x: x.arrival)
    next_arrival_idx = 0
    in_queue = set()  # Para evitar duplicados en la cola

    while any(remaining.values()) or queue:
        # Agregar procesos que han llegado a la cola
        while (next_arrival_idx < len(sorted_procs) and 
               sorted_procs[next_arrival_idx].arrival <= time):
            proc = sorted_procs[next_arrival_idx]
            if proc.pid not in in_queue and remaining[proc.pid] > 0:
                queue.append(proc.pid)
                in_queue.add(proc.pid)
            next_arrival_idx += 1

        if not queue:
            # Si no hay procesos en cola, avanzar al siguiente arrival
            if next_arrival_idx < len(sorted_procs):
                time = sorted_procs[next_arrival_idx].arrival
                continue
            else:
                break

        # Ejecutar el proceso al frente de la cola
        pid = queue.popleft()
        in_queue.remove(pid)
        
        if remaining[pid] > 0:
            execution_time = min(quantum, remaining[pid])
            start_time = time
            end_time = time + execution_time
            timeline.append((pid, start_time, end_time))
            remaining[pid] -= execution_time
            time = end_time

            # Agregar nuevos arrivals durante la ejecución
            while (next_arrival_idx < len(sorted_procs) and 
                   sorted_procs[next_arrival_idx].arrival <= time):
                proc = sorted_procs[next_arrival_idx]
                if proc.pid not in in_queue and remaining[proc.pid] > 0:
                    queue.append(proc.pid)
                    in_queue.add(proc.pid)
                next_arrival_idx += 1

            # Si el proceso no terminó, regresarlo a la cola
            if remaining[pid] > 0:
                queue.append(pid)
                in_queue.add(pid)

    return timeline

def priority_scheduler(procs: list[Process], preemptive: bool) -> list[tuple]:
    """Priority Scheduling - preemptivo o no preemptivo"""
    time = 0
    remaining = {p.pid: p.burst for p in procs}
    timeline = []
    current_process = None
    current_start = 0

    if preemptive:
        # Priority preemptivo
        while any(remaining.values()):
            # Procesos listos (han llegado y tienen tiempo restante)
            ready = [p for p in procs if p.arrival <= time and remaining[p.pid] > 0]
            
            if ready:
                # Seleccionar el de mayor prioridad (menor número = mayor prioridad)
                highest_priority = min(ready, key=lambda x: x.priority)
                
                # Si cambia el proceso en ejecución
                if current_process != highest_priority.pid:
                    if current_process is not None:
                        timeline.append((current_process, current_start, time))
                    current_process = highest_priority.pid
                    current_start = time
                
                # Ejecutar por una unidad de tiempo
                remaining[highest_priority.pid] -= 1
                time += 1
            else:
                # Si no hay procesos listos, avanzar al siguiente arrival
                if any(remaining.values()):
                    next_arrival = min(p.arrival for p in procs if remaining[p.pid] > 0)
                    time = next_arrival

        # Agregar el último segmento
        if current_process is not None:
            timeline.append((current_process, current_start, time))
    
    else:
        # Priority no preemptivo
        remaining_procs = procs.copy()
        
        while remaining_procs:
            # Procesos que han llegado
            ready = [p for p in remaining_procs if p.arrival <= time]
            
            if ready:
                # Seleccionar el de mayor prioridad
                highest_priority = min(ready, key=lambda x: x.priority)
                start_time = max(time, highest_priority.arrival)
                end_time = start_time + highest_priority.burst
                timeline.append((highest_priority.pid, start_time, end_time))
                time = end_time
                remaining_procs.remove(highest_priority)
            else:
                # Si no hay procesos listos, avanzar al siguiente arrival
                next_arrival = min(p.arrival for p in remaining_procs)
                time = next_arrival

    return timeline

# Función auxiliar para mostrar resultados
def print_timeline(timeline, algorithm_name):
    print(f"\n{algorithm_name}:")
    print("PID\tStart\tEnd")
    for pid, start, end in timeline:
        print(f"{pid}\t{start}\t{end}")