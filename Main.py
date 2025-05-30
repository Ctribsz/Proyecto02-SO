import time
import streamlit as st
import pandas as pd
import plotly.express as px
from parsers import parse_processes_txt, parse_resources_txt, parse_actions_txt
from schedulers import (
    Process,
    fifo_scheduler, sjf_scheduler, srt_scheduler,
    rr_scheduler, priority_scheduler
)
from synchronization import simulate_sync
from visualization import animate_by_process, animate_by_cycle, animate_line_state

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css(".streamlit/style.css")  

def main():
    st.sidebar.header("🔧 Modo de simulación")
    mode = st.sidebar.radio("¿Qué quieres simular?", ["Calendarización", "Sincronización"])

    if mode == "Calendarización":
        # Selección múltiple de algoritmos
        algos = st.sidebar.multiselect(
            "Algoritmos",
            ["FIFO", "SJF (no-preempt)", "SRT", "Round Robin", "Priority"],
            default=["FIFO"]
        )
        # Controles condicionales
        quantum = st.sidebar.slider("Quantum (RR)", 1, 20, 2) if "Round Robin" in algos else None
        preempt = st.sidebar.checkbox("Priority preemptiva", True) if "Priority" in algos else None
        proc_file = st.sidebar.file_uploader("Procesos (.txt)", type="txt")
        run_btn = st.sidebar.button("▶️ Ejecutar Calendarización")

        if proc_file and run_btn:
            df_proc = parse_processes_txt(proc_file)
            procs = [Process(r.pid, r.arrival, r.burst, r.priority) for r in df_proc.itertuples()]

            # Simular cada algoritmo
            timelines = {}
            for algo in algos:
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
                df_tl = pd.DataFrame(tl, columns=["Process", "start", "end"])
                timelines[algo] = df_tl

            global_cmax = max(df["end"].max() for df in timelines.values())

            # Colores fijos por PID
            pids = [p.pid for p in procs]
            colors = px.colors.qualitative.Plotly
            color_map = {pid: colors[i % len(colors)] for i, pid in enumerate(pids)}

            # 6) Creamos un placeholder full-width por cada algoritmo
            placeholders = {
                algo: st.empty()
                for algo in algos
            }

            # 7) Animar ciclo a ciclo
            for t in range(0, int(global_cmax) + 1):
                for algo, df in timelines.items():
                    # filtrar la porción hasta el ciclo t
                    vis = (
                        df
                        .assign(
                            ds = df["start"].clip(upper=t),
                            de = df["end"].clip(upper=t)
                        )
                        .query("ds < de")
                    )
                    fig = px.bar(
                        vis,
                        x="de", base="ds",
                        y="Process",
                        orientation="h",
                        color="Process",
                        title=f"{algo} — Ciclo {t}/{int(global_cmax)}",
                        color_discrete_map=color_map  # tu mapeo PID→color
                    )
                    fig.update_yaxes(autorange="reversed")
                    fig.update_xaxes(range=[0, global_cmax], title="Ciclo")
                    fig.update_layout(showlegend=True, height=250)

                    # **Aquí** uso el placeholder correcto:
                    placeholders[algo].plotly_chart(fig, use_container_width=True)

                # pausa para animar
                time.sleep(0.4)
            
            arrival_map = {p.pid: p.arrival for p in procs}

            metrics = []
            for algo, df in timelines.items():
                # df tiene columnas ['Process','start','end']
                waits    = [row.start - arrival_map[row.Process]   for row in df.itertuples()]
                turnaround = [row.end   - arrival_map[row.Process] for row in df.itertuples()]

                metrics.append({
                    'Algoritmo'        : algo,
                    'Espera Media'     : sum(waits) / len(waits),
                    'Turnaround Medio' : sum(turnaround) / len(turnaround)
                })

            df_metrics = pd.DataFrame(metrics).set_index('Algoritmo')

            st.subheader("📊 Métricas de Scheduling")
            st.dataframe(df_metrics)

            # 9) Gráfica de barras comparativa
            fig_m = px.bar(
                df_metrics.reset_index(),
                x='Algoritmo',
                y=['Espera Media','Turnaround Medio'],
                barmode='group',
                labels={
                'value': 'Tiempo (ciclos)',
                'variable': 'Métrica'
                },
                title='Comparativa de Espera y Turnaround'
            )
            fig_m.update_layout(margin=dict(l=40,r=40,t=60,b=40))
            st.plotly_chart(fig_m, use_container_width=True)

            # 10) Mensaje de finalización
            st.success("✅ Animación de Calendarización completada y métricas mostradas")


    else:
        # Sincronización concurrente
        sync_modes = st.sidebar.multiselect(
            "Modos de sincronización",
            ["Mutex", "Semaphore"],
            default=["Mutex", "Semaphore"]
        )
        f_proc = st.sidebar.file_uploader("Procesos (.txt)", type="txt")
        f_res  = st.sidebar.file_uploader("Recursos (.txt)", type="txt")
        f_act  = st.sidebar.file_uploader("Acciones (.txt)", type="txt")
        run_sync = st.sidebar.button("▶️ Ejecutar Sincronización")

        if run_sync:
            if not (f_proc and f_res and f_act):
                st.error("Debes subir los 3 archivos: procesos, recursos y acciones")
                return

            # Parsear entradas
            df_p = parse_processes_txt(f_proc)
            df_r = parse_resources_txt(f_res)
            df_a = parse_actions_txt(f_act)

            sem_init = {r: c for r, c in zip(df_r.resource, df_r.initial_count)}

            # Simulación y construcción de df_state para cada modo
            sync_results = {}
            for mode in sync_modes:
                df_tl = simulate_sync(df_a, mode, sem_init if mode == "Semaphore" else None)

                # Construir df_state ciclo a ciclo
                times = sorted(df_tl['time'].unique())
                resources = sorted(df_tl['resource'].unique())
                state = {r: (1 if mode == "Mutex" else sem_init[r]) for r in resources}
                records = []
                for t in times:
                    for _, ev in df_tl[df_tl.time == t].iterrows():
                        if mode == "Mutex":
                            state[ev.resource] = 0 if ev.status == "acquired" else 1
                        else:
                            delta = -1 if ev.status == "acquired" else +1
                            state[ev.resource] += delta
                    rec = {'time': t}
                    rec.update(state)
                    records.append(rec)
                df_state = pd.DataFrame(records)

                sync_results[mode] = {
                    'df_tl': df_tl,
                    'df_state': df_state,
                    'resources': resources
                }

            # UI: columnas para cada modo
            cols = st.columns(len(sync_modes))
            line_ph = {}
            event_ph = {}
            for idx, mode in enumerate(sync_modes):
                cols[idx].subheader(mode)
                line_ph[mode] = cols[idx].empty()
                event_ph[mode] = cols[idx].empty()

            # Animar simultáneamente
            global_max = max(res['df_state']['time'].max() for res in sync_results.values())
            delay = 0.5
            for t in range(global_max + 1):
                for mode, res in sync_results.items():
                    df_state = res['df_state']
                    df_tl = res['df_tl']
                    resources = res['resources']

                    # Línea de estados
                    df_plot = df_state[df_state['time'] <= t]
                    fig = px.line(
                        df_plot,
                        x='time', y=resources,
                        labels={'time':'Ciclo','value':'Estado','variable':'Recurso'},
                        title=f"{mode} — Ciclo {t}/{global_max}"
                    )
                    fig.update_layout(height=300, margin=dict(l=40,r=20,t=40,b=20))
                    line_ph[mode].plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"{mode}_line_{t}"
                    )

                    # Eventos ocurridos en t
                    acts = df_tl[df_tl['time'] == t]
                    if not acts.empty:
                        textos = []
                        for _, ev in acts.iterrows():
                            icon = '🟢' if ev.status=='acquired' else '🔴'
                            verbo = 'adquirió' if ev.status=='acquired' else 'liberó'
                            textos.append(f"{icon} `{ev.pid}` **{verbo}** `{ev.resource}`")
                        event_ph[mode].markdown("**Eventos:**" + " ".join(textos))
                    else:
                        event_ph[mode].empty()

                time.sleep(delay)

            # Métricas de sincronización por recurso
            from collections import deque
            metrics_sync = []
            for mode, res in sync_results.items():
                df_tl = res['df_tl']
                resources = res['resources']
                queues = {r: deque() for r in resources}
                queue_sizes = {r: 0 for r in resources}
                pending_wait = {}
                pending_hold = {}
                wait_times = []
                hold_times = []
                for _, row in df_tl.sort_values('time').iterrows():
                    t,pid,status,rsrc = row['time'], row['pid'], row['status'], row['resource']
                    key = (pid, rsrc)
                    if status == 'waiting':
                        queues[rsrc].append(pid)
                        queue_sizes[rsrc] = max(queue_sizes[rsrc], len(queues[rsrc]))
                        pending_wait[key] = t
                    elif status == 'acquired':
                        if key in pending_wait:
                            wait_times.append(t - pending_wait.pop(key))
                        pending_hold[key] = t
                        if pid in queues[rsrc]: queues[rsrc].remove(pid)
                    else:
                        if key in pending_hold:
                            hold_times.append(t - pending_hold.pop(key))
                avg_wait = sum(wait_times)/len(wait_times) if wait_times else 0
                avg_hold = sum(hold_times)/len(hold_times) if hold_times else 0
                for r in resources:
                    metrics_sync.append({
                        'Modo': mode,
                        'Recurso': r,
                        'Espera Media': avg_wait,
                        'Uso Medio': avg_hold,
                        'Pico Cola': queue_sizes[r]
                    })

            df_ms = pd.DataFrame(metrics_sync).set_index(['Modo','Recurso'])
            st.subheader("📊 Métricas de Sincronización")
            st.dataframe(df_ms)
            fig_sync = px.bar(
                df_ms.reset_index(),
                x='Recurso', y=['Espera Media','Uso Medio','Pico Cola'],
                color='Modo', barmode='group',
                labels={'value':'Ciclos','variable':'Métrica'},
                title='Comparativa de Métricas de Sincronización'
            )
            fig_sync.update_layout(margin=dict(l=40,r=40,t=60,b=40))
            st.plotly_chart(fig_sync, use_container_width=True)
            st.success(f"✅ Sincronización completada para: {', '.join(sync_modes)}")


if __name__ == "__main__":
    main()