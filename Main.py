import time
import streamlit as st
import pandas as pd
import plotly.express as px
from dataclasses import dataclass
from collections import deque

@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    priority: int = None    
    remaining: int = None

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.burst

@dataclass
class Event:
    pid: str
    time: int
    action: str       # "acquire" o "release"
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

import pandas as pd

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

def fifo_scheduler(processes):
    time_ptr = 0
    tl = []
    for p in sorted(processes, key=lambda x: x.arrival):
        start = max(time_ptr, p.arrival)
        end = start + p.burst
        tl.append((p.pid, start, end))
        time_ptr = end
    return tl

def timeline_to_df(tl):
    return pd.DataFrame(tl, columns=['Process','start','end'])

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

def animate_by_process(tl, cmax, placeholder, delay=0.7):
    shown = []
    for pid, s, e in tl:
        shown.append((pid, s, e))
        df = pd.DataFrame(shown, columns=['Process','start','end'])
        fig = px.bar(df, x='end', base='start', y='Process',
                     orientation='h', color='Process',
                     title=f"Procesos completados: {len(shown)}/{len(tl)}")
        fig.update_yaxes(autorange='reversed')
        fig.update_xaxes(title='Tiempo', range=[0, cmax+1])
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(delay)

def animate_by_cycle(tl, cmax, placeholder, delay=0.3):
    df_all = pd.DataFrame(tl, columns=['Process','start','end'])
    for t in range(0, cmax+1):
        vis = df_all.assign(
            ds = df_all['start'].clip(upper=t),
            de = df_all['end'].clip(upper=t)
        ).query("ds < de")
        fig = px.bar(vis, x='de', base='ds', y='Process',
                     orientation='h', color='Process',
                     title=f"Ciclo {t}/{cmax}")
        fig.update_yaxes(autorange='reversed')
        fig.update_xaxes(title='Tiempo', range=[0, cmax+1])
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(delay)

def build_sync_gantt(df_tl: pd.DataFrame) -> pd.DataFrame:
    """
    Toma el DataFrame con columnas [time, pid, status, resource]
    y devuelve otro con columnas [resource, pid, start, end],
    donde cada fila es un intervalo en que pid mantuvo el recurso.
    """
    segments = []
    # clave temporal para “acquired” pendientes
    start_times: dict[tuple[str,str], int] = {}

    for _, row in df_tl.iterrows():
        key = (row["resource"], row["pid"])
        t, status = row["time"], row["status"]
        if status == "acquired":
            start_times[key] = t
        elif status == "released" and key in start_times:
            segments.append({
                "resource": row["resource"],
                "pid":      row["pid"],
                "start":    start_times[key],
                "end":      t
            })
            del start_times[key]

    return pd.DataFrame(segments)

def animate_sync_visual(df_tl: pd.DataFrame, sync_type: str, sem_init: dict[str,int] = None, delay: float = 0.7):
    visual_ph = st.empty()
    status_ph = st.empty()
    event_ph = st.empty()
    history_ph = st.empty()
    
    segments = []
    start_times = {}
    
    # Estado inicial
    if sync_type == "Mutex":
        unique_resources = df_tl["resource"].unique()
        resource_state = {r: None for r in unique_resources}
    else:
        if sem_init is None:
            raise ValueError("sem_init es requerido para semáforos")
        resource_state = sem_init.copy()
    
    df_sorted = df_tl.sort_values("time").reset_index(drop=True)
    
    for idx, row in df_sorted.iterrows():
        t, pid, status, rsrc = row["time"], row["pid"], row["status"], row["resource"]
        key = (rsrc, pid)
        
        # Lógica acquire/release
        if status == "acquired":
            start_times[key] = t
            if sync_type == "Mutex":
                resource_state[rsrc] = pid
            else:
                resource_state[rsrc] -= 1
        else:
            if key in start_times:
                segments.append({
                    "resource": rsrc, 
                    "pid": str(pid),
                    "start": start_times[key],
                    "end": t,
                    "duration": t - start_times[key]
                })
                del start_times[key]
                
            if sync_type == "Mutex":
                resource_state[rsrc] = None
            else:
                resource_state[rsrc] += 1
        
        # VISUALIZACIÓN PRINCIPAL: Crear representación visual
        with visual_ph.container():
            st.markdown(f"### Estado de {sync_type} en tiempo t={t}")
            
            # Crear columnas para cada recurso
            resources = list(resource_state.keys())
            if len(resources) <= 4:
                cols = st.columns(len(resources))
            else:
                # Si hay muchos recursos, hacer filas
                cols = st.columns(min(4, len(resources)))
            
            for i, (resource, state) in enumerate(resource_state.items()):
                col_idx = i % len(cols)
                
                with cols[col_idx]:
                    if sync_type == "Mutex":
                        if state is None:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; border: 2px solid green; border-radius: 10px; background-color: #d4f4dd;">
                                <h4>{resource}</h4>
                                <div style="font-size: 30px;">🟢</div>
                                <p><b>LIBRE</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; border: 2px solid red; border-radius: 10px; background-color: #fdd4d4;">
                                <h4>{resource}</h4>
                                <div style="font-size: 30px;">🔴</div>
                                <p><b>Ocupado por:</b><br>{state}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:  # Semáforos
                        total = sem_init[resource]
                        available = state
                        used = total - available
                        
                        # Crear círculos para representar el semáforo
                        circles = ""
                        for j in range(total):
                            if j < used:
                                circles += "🔴 "  # Ocupado
                            else:
                                circles += "🟢 "  # Disponible
                        
                        color = "green" if available > 0 else "red"
                        bg_color = "#d4f4dd" if available > 0 else "#fdd4d4"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px; background-color: {bg_color};">
                            <h4>{resource}</h4>
                            <div style="font-size: 20px;">{circles}</div>
                            <p><b>Disponible:</b> {available}/{total}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Historial de uso
        if segments:
            with history_ph.container():
                st.markdown("### Historial de uso")
                df_segments = pd.DataFrame(segments)
                
                # Mostrar como tabla simple
                for _, seg in df_segments.iterrows():
                    st.markdown(f"• **{seg['pid']}** usó **{seg['resource']}** desde t={seg['start']} hasta t={seg['end']} (duración: {seg['duration']})")
        
        # Estado en tabla
        if sync_type == "Mutex":
            df_state = pd.DataFrame([
                {"Resource": rsrc, "Owner": owner if owner else "🟢 LIBRE"}
                for rsrc, owner in resource_state.items()
            ])
        else:
            df_state = pd.DataFrame([
                {"Resource": rsrc, "Available": f"{count}/{sem_init[rsrc]}"}
                for rsrc, count in resource_state.items()
            ])
        
        status_ph.table(df_state)
        
        # Evento actual
        verb = "adquirió" if status == "acquired" else "liberó"
        color = "🟢" if status == "acquired" else "🔴"
        event_ph.markdown(f"**t={t}**: {color} `{pid}` **{verb}** `{rsrc}`")
        
        time.sleep(delay)
    
    st.success(f"🔒 Animación de {sync_type} completada")

def run_animation(tl, cmax, algo, placeholder):
    if algo in ["FIFO", "SJF (no-preempt)"]:
        animate_by_process(tl, cmax, placeholder)
    else:
        animate_by_cycle(tl, cmax, placeholder)
def main():
    st.set_page_config(layout="wide")
    st.sidebar.header("Modo de simulación")
    mode = st.sidebar.radio("¿Qué quieres simular?", ["Calendarización", "Sincronización"])

    if mode == "Calendarización":
        # —– Sidebar calendarización —
        algo = st.sidebar.selectbox(
            "Algoritmo",
            ["FIFO", "SJF (no-preempt)", "SRT", "Round Robin", "Priority"]
        )
        quantum = st.sidebar.slider("Quantum", 1, 10, 2) if algo == "Round Robin" else None
        preempt = st.sidebar.checkbox("Preemptivo", True) if algo == "Priority" else None

        st.header(f"🏷️ Calendarización — {algo}")
        uploaded = st.file_uploader(
            "Sube CSV con pid,arrival,burst,priority",
            type=['csv']
        )

        if uploaded and st.button("▶️ Ejecutar calendarización"):
            df = pd.read_csv(uploaded)
            required_cols = {'pid', 'arrival', 'burst', 'priority'}
            if not required_cols.issubset(df.columns):
                st.error(f"El CSV debe contener columnas: {', '.join(required_cols)}")
                return

            procs = []
            for _, row in df.iterrows():
                procs.append(
                    Process(
                        pid      = str(row["pid"]),
                        arrival  = int(row["arrival"]),
                        burst    = int(row["burst"]),
                        priority = int(row["priority"])
                    )
                )

            # Selección del scheduler
            if algo == "FIFO":
                tl = fifo_scheduler(procs)
            elif algo == "SJF (no-preempt)":
                tl = sjf_scheduler(procs)
            elif algo == "SRT":
                tl = srt_scheduler(procs)
            elif algo == "Round Robin":
                tl = rr_scheduler(procs, quantum)
            else:  # Priority
                tl = priority_scheduler(procs, preempt)

            # Animación y métricas
            cmax = max(end for _, _, end in tl)
            placeholder = st.empty()
            run_animation(tl, cmax, algo, placeholder)

            waiting_times = [
                start - next(p.arrival for p in procs if p.pid == pid)
                for pid, start, _ in tl
            ]
            st.success(f"⏱️ Tiempo de espera promedio: {sum(waiting_times)/len(waiting_times):.2f} ciclos")

    else:
        # —– Sidebar sincronización —
        sync_type = st.sidebar.selectbox("Tipo de sincronización", ["Mutex", "Semaphore"])
        st.header(f"🔒 Sincronización — {sync_type}")

        f_proc = st.file_uploader("Procesos (.txt)", type="txt")
        f_res  = st.file_uploader("Recursos (.txt)", type="txt")
        f_act  = st.file_uploader("Acciones (.txt)", type="txt")

        if st.button("▶️ Ejecutar sincronización"):
            if not (f_proc and f_res and f_act):
                st.error("Debes subir los 3 archivos: procesos, recursos y acciones.")
                return

            # 1) Procesos (solo usamos PID aquí, aunque leemos el resto)
            df_p = pd.read_csv(f_proc, sep=",", header=None,
                               names=["pid","burst","arrival","priority"])
            # 2) Recursos
            df_r = pd.read_csv(f_res, sep=",", header=None,
                               names=["resource","initial_count"])
            # 3) Acciones
            df_a = pd.read_csv(f_act, sep=",", header=None,
                               names=["pid","action","resource","time"])

            sem_init = dict(zip(df_r.resource, df_r.initial_count))
            df_tl = simulate_sync(df_a, sync_type, sem_init)

            # Llamamos con sem_init en el caso de semáforos
            animate_sync_visual(df_tl, sync_type, sem_init=sem_init, delay=0.8)



if __name__ == "__main__":
    main()