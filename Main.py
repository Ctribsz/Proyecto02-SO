import time
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import deque
from parsers import parse_processes_txt, parse_resources_txt, parse_actions_txt
from schedulers import (
    Process,
    fifo_scheduler, sjf_scheduler, srt_scheduler,
    rr_scheduler, priority_scheduler
)
from synchronization import simulate_sync
from visualization import animate_by_process, animate_by_cycle, animate_line_state


def local_css(file_name: str):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Archivo de estilos no encontrado: {file_name}")

# Cargar estilo opcional
local_css(".streamlit/style.css")


def main():
    st.sidebar.header("🔧 Modo de simulación")
    mode = st.sidebar.radio("¿Qué quieres simular?", ["Calendarización", "Sincronización"])

    # ----------------- CALENDARIZACIÓN ----------------- #
    if mode == "Calendarización":
        algos = st.sidebar.multiselect(
            "Algoritmos",
            ["FIFO", "SJF (no-preempt)", "SRT", "Round Robin", "Priority"],
            default=["FIFO"]
        )
        quantum = st.sidebar.slider("Quantum (RR)", 1, 20, 2) if "Round Robin" in algos else None
        preempt = st.sidebar.checkbox("Priority preemptiva", True) if "Priority" in algos else None
        proc_file = st.sidebar.file_uploader("Procesos (.txt)", type="txt")
        run_btn = st.sidebar.button("▶️ Ejecutar Calendarización")

        if proc_file and run_btn:
            # 1) Parsing defensivo
            try:
                df_proc = parse_processes_txt(proc_file)
            except Exception as e:
                st.error(f"Error al parsear procesos: {e}")
                return

            # 2) Validar columnas y valores
            for col in ['pid', 'arrival', 'burst', 'priority']:
                if col not in df_proc.columns:
                    st.error(f"Falta columna '{col}' en procesos")
                    return

            if (df_proc.arrival < 0).any():
                st.error("Arrival debe ser ≥ 0")
                return
            if (df_proc.burst <= 0).any():
                st.error("Burst debe ser > 0")
                return
            if df_proc.pid.duplicated().any():
                st.error("PID duplicados encontrados en procesos")
                return

            # 3) Instanciar Process
            procs = [
                Process(pid=r.pid, arrival=int(r.arrival), burst=int(r.burst), priority=int(r.priority))
                for r in df_proc.itertuples()
            ]

            # 4) Ejecutar cada scheduler y guardar sus timelines en un DataFrame
            timelines = {}
            for algo in algos:
                try:
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
                except Exception as e:
                    st.error(f"Error en {algo}: {e}")
                    return

                # Convertimos la lista de tuplas (pid,start,end) en DataFrame
                df_tl = pd.DataFrame(tl, columns=["Process", "start", "end"])
                timelines[algo] = df_tl

            # 5) Validar que timelines no esté vacío
            if not timelines:
                st.warning("No hay timelines generados")
                return

            try:
                global_cmax = max(df["end"].max() for df in timelines.values())
            except Exception:
                st.error("Error calculando ciclo máximo")
                return

            # 6) Configurar colores fijos por PID
            pids = [p.pid for p in procs]
            color_map = {
                pid: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                for i, pid in enumerate(pids)
            }

            # 7) Animación apilada + Subheaders + Logs
            for algo in algos:
                st.subheader(algo)
                placeholder_chart = st.empty()    # aquí dibujamos la gráfica
                placeholder_log   = st.empty()    # aquí escribimos el log de texto

                df_tl = timelines[algo]
                cmax  = int(df_tl["end"].max())

                # 7.A) Si es NO‐PREEMPTIVO, dibujar bloque completo
                if algo in ["FIFO", "SJF (no-preempt)"] or (algo == "Priority" and not preempt):
                    # Llamamos animate_by_process → dibuja barra completa por proceso
                    tuplas = [(row.Process, row.start, row.end) for row in df_tl.itertuples()]
                    animate_by_process(
                        tl=tuplas,
                        cmax=global_cmax,
                        placeholder=placeholder_chart
                    )

                    # Construir log de eventos en texto para algoritmos no preemptivos:
                    # Para cada tupla (pid, start, end), agregamos dos líneas:
                    #   “t=<start>: <pid> start”
                    #   “t=<end>  : <pid> end”
                    lines = []
                    for (pid, start, end) in tuplas:
                        lines.append(f"t={start}: `{pid}` start")
                        lines.append(f"t={end} : `{pid}` end")
                    # Mostramos el log completo con placeholder_log
                    placeholder_log.markdown("```\n" + "\n".join(lines) + "\n```")

                # 7.B) Si es PREEMPTIVO, animar ciclo a ciclo
                else:
                    # Primero, construimos un log de “eventos por ciclo”:
                    # vamos a recopilar cada “start/end” para hacer log en orden cronológico.
                    all_events = []
                    for row in df_tl.itertuples():
                        pid, start, end = row.Process, row.start, row.end
                        all_events.append((start, pid, "start"))
                        all_events.append((end,   pid, "end"))
                    # Ordenamos por tiempo de evento
                    all_events.sort(key=lambda x: (x[0], x[2]))  # si empatan time, "end" después de "start"

                    # Pero para no recargar la animación con texto continuo en cada ciclo,
                    # vamos a mostrar el log completo \emph{al final} (o durante si prefieres).
                    # Aquí prefiero mostrarlo después de terminar la animación principal.

                    # Animación gráfica “ciclo a ciclo”
                    for t in range(0, cmax + 1):
                        vis = (
                            df_tl
                            .assign(
                                ds = df_tl["start"].clip(upper=t),
                                de = df_tl["end"].clip(upper=t)
                            )
                            .query("ds < de")
                        )
                        fig = px.bar(
                            vis,
                            x="de", base="ds",
                            y="Process",
                            orientation="h",
                            color="Process",
                            color_discrete_map=color_map
                        )
                        fig.update_yaxes(autorange="reversed")
                        fig.update_xaxes(range=[0, global_cmax], title="Ciclo")
                        fig.update_layout(
                            showlegend=True,
                            height=150,
                            margin=dict(l=40, r=20, t=10, b=10)
                        )

                        placeholder_chart.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"{algo}_cycle_{t}"
                        )
                        time.sleep(0.1)

                    # Una vez terminada la animación, imprimimos el log completo
                    lines = []
                    for (instant, pid, action) in all_events:
                        # action == "start" o "end"
                        if action == "start":
                            lines.append(f"t={instant}: `{pid}` start")
                        else:
                            lines.append(f"t={instant}: `{pid}` end")
                    placeholder_log.markdown("```\n" + "\n".join(lines) + "\n```")

                # Separador entre algoritmos
                st.markdown("---")

            # 8) Métricas comparativas (igual que antes)
            arrival_map = {p.pid: p.arrival for p in procs}
            metrics = []
            for algo, df_tl in timelines.items():
                waits = [r.start - arrival_map[r.Process] for r in df_tl.itertuples()]
                ta    = [r.end   - arrival_map[r.Process] for r in df_tl.itertuples()]
                metrics.append({
                    'Algoritmo'       : algo,
                    'Espera Media'    : sum(waits) / len(waits),
                    'Turnaround Medio': sum(ta)    / len(ta)
                })
            df_metrics = pd.DataFrame(metrics).set_index('Algoritmo')

            st.subheader("📊 Métricas de Scheduling")
            st.dataframe(df_metrics)

            fig_m = px.bar(
                df_metrics.reset_index(),
                x='Algoritmo',
                y=['Espera Media', 'Turnaround Medio'],
                barmode='group',
                labels={'value': 'Ciclos', 'variable': 'Métrica'},
                title='Comparativa de Métricas'
            )
            fig_m.update_layout(margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig_m, use_container_width=True)

            st.success("✅ Calendarización completada y métricas mostradas")

    # ----------------- SINCRONIZACIÓN ----------------- #
    else:
        sync_modes = st.sidebar.multiselect(
            "Modos de sincronización",
            ["Mutex", "Semaphore"],
            default=["Mutex", "Semaphore"]
        )
        f_proc   = st.sidebar.file_uploader("Procesos (.txt)", type="txt")
        f_res    = st.sidebar.file_uploader("Recursos (.txt)", type="txt")
        f_act    = st.sidebar.file_uploader("Acciones (.txt)", type="txt")
        run_sync = st.sidebar.button("▶️ Ejecutar Sincronización")

        if run_sync:
            # Validación de archivos
            if not (f_proc and f_res and f_act):
                st.error("Debes subir los 3 archivos: procesos, recursos y acciones")
                return

            # Parsing defensivo
            try:
                df_p = parse_processes_txt(f_proc)
                df_r = parse_resources_txt(f_res)
                df_a = parse_actions_txt(f_act)
            except Exception as e:
                st.error(f"Error al parsear archivos: {e}")
                return

            # Validaciones básicas
            if df_p.pid.duplicated().any():
                st.error("PID duplicados en procesos")
                return
            if (df_p.arrival < 0).any():
                st.error("Arrival debe ser ≥0")
                return
            if df_a['action'].isin([None]).any():
                st.error("Acciones inválidas en archivo de acciones")
                return

            sem_init = {r: c for r, c in zip(df_r.resource, df_r.initial_count)}
            if "Semaphore" in sync_modes and any(v < 0 for v in sem_init.values()):
                st.error("Initial count de semáforo debe ser ≥0")
                return

            # Simulación y construcción de df_state
            sync_results = {}
            for mode in sync_modes:
                try:
                    df_tl = simulate_sync(df_a, mode, sem_init if mode == "Semaphore" else None)
                except Exception as e:
                    st.error(f"Error en simulate_sync({mode}): {e}")
                    return

                if df_tl.empty:
                    st.warning(f"No hay eventos para {mode}")
                    continue

                times     = sorted(df_tl['time'].unique())
                resources = sorted(df_tl['resource'].unique())
                state = {r: (1 if mode == "Mutex" else sem_init[r]) for r in resources}
                records = []
                for t in times:
                    for ev in df_tl[df_tl.time == t].itertuples():
                        if mode == "Mutex":
                            state[ev.resource] = 0 if ev.status == "acquired" else 1
                        else:
                            state[ev.resource] += -1 if ev.status == "acquired" else +1
                    rec = {'time': t}
                    rec.update(state)
                    records.append(rec)
                sync_results[mode] = {'df_tl': df_tl, 'df_state': pd.DataFrame(records), 'resources': resources}

            if not sync_results:
                st.warning("No se simularon modos de sincronización")
                return

            # Placeholders full-width uno encima de otro
            placeholders = {}
            event_ph     = {}
            for mode in sync_results:
                st.subheader(mode)
                placeholders[mode] = st.empty()
                event_ph[mode]     = st.empty()

            # Expander con historial completo en inglés
            history_ph = st.expander("📜 Sync Event History")
            history = []

            delay     = 0.5
            global_max = max(res['df_state']['time'].max() for res in sync_results.values())

            # Animación sincronizada
            for t in range(global_max + 1):
                for mode, res in sync_results.items():
                    df_state = res['df_state']
                    df_tl    = res['df_tl']
                    resources = res['resources']

                    # 1) Línea de estado
                    df_plot = df_state[df_state['time'] <= t]
                    fig = px.line(
                        df_plot,
                        x='time', y=resources,
                        labels={'time': 'Cycle', 'value': 'State', 'variable': 'Resource'}
                    )
                    fig.update_layout(height=200, margin=dict(l=40, r=20, t=20, b=20))
                    placeholders[mode].plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"{mode}_line_{t}"
                    )

                    # 2) Mostrar eventos de este ciclo y acumular historial
                    events = df_tl[df_tl['time'] == t]
                    if not events.empty:
                        lines = []
                        for ev in events.itertuples():
                            lines.append(f"t={t}: `{ev.pid}` **{ev.status}** `{ev.resource}`")
                        event_ph[mode].markdown("".join(lines))
                        history.extend(lines)
                    else:
                        event_ph[mode].empty()

                # 3) Actualizar historial completo
                history_ph.markdown("".join(history))
                time.sleep(delay)

            metrics = []
            for mode, res in sync_results.items():
                df_tl    = res['df_tl']
                resources = res['resources']

                # Reconstruir colas y emparejamientos
                queues      = {r: deque() for r in resources}
                max_queue   = {r: 0        for r in resources}
                pending_wait= {}
                pending_hold= {}
                wait_times  = []
                hold_times  = []

                for _, row in df_tl.sort_values('time').iterrows():
                    t, pid, status, rsrc = row['time'], row['pid'], row['status'], row['resource']
                    key = (pid, rsrc)

                    if status == 'waiting':
                        queues[rsrc].append(pid)
                        max_queue[rsrc] = max(max_queue[rsrc], len(queues[rsrc]))
                        pending_wait[key] = t

                    elif status == 'acquired':
                        # espera = acquire_time - request_time
                        if key in pending_wait:
                            wait_times.append(t - pending_wait.pop(key))
                        # arrancamos hold timer
                        pending_hold[key] = t
                        # si estaba en cola, lo sacamos
                        if pid in queues[rsrc]:
                            queues[rsrc].remove(pid)

                    else:  # released
                        # hold time = release_time - acquire_time
                        if key in pending_hold:
                            hold_times.append(t - pending_hold.pop(key))

                avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
                avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0

                for r in resources:
                    metrics.append({
                        'Modo'         : mode,
                        'Recurso'      : r,
                        'Espera Media' : avg_wait,
                        'Uso Medio'    : avg_hold,
                        'Pico Cola'    : max_queue[r]
                    })

            df_ms = pd.DataFrame(metrics).set_index(['Modo','Recurso'])

            st.subheader("📊 Métricas de Sincronización")
            st.dataframe(df_ms)

            fig_sync = px.bar(
                df_ms.reset_index(),
                x='Recurso',
                y=['Espera Media','Uso Medio','Pico Cola'],
                color='Modo',
                barmode='group',
                labels={'value':'Ciclos','variable':'Métrica'},
                title='Comparativa de Métricas de Sincronización'
            )
            fig_sync.update_layout(margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig_sync, use_container_width=True)
            
            st.success(f"✅ Sincronización completada para: {', '.join(sync_results.keys())}")


if __name__ == "__main__":
    main()