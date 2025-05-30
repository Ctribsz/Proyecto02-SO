import time
import pandas as pd
import plotly.express as px
import streamlit as st

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

def animate_line_state(
    df_state: pd.DataFrame,
    df_events: pd.DataFrame,
    resources: list[str],
    sync_type: str,
    delay: float = 0.6
):
    """
    Animación de línea + texto de eventos.
    - df_state: DataFrame con columnas ['time'] + resources.
    - df_events: DataFrame original con ['time','pid','status','resource'].
    - resources: lista de nombres de recursos.
    - sync_type: "Mutex" o "Semaphore".
    - delay: segundos entre ciclos.
    """
    line_ph  = st.empty()
    event_ph = st.empty()
    max_cycle = int(df_state['time'].max())

    # Ordenamos los eventos por tiempo
    evs = df_events.sort_values('time').reset_index(drop=True)

    for t in df_state['time']:
        # 1) Dibujar la línea hasta el ciclo t
        df_plot = df_state[df_state['time'] <= t]
        fig = px.line(
            df_plot,
            x='time',
            y=resources,
            labels={'time':'Ciclo','value':'Estado','variable':'Recurso'},
            title=f"Estado de Recursos ({sync_type}) — Ciclo {t}/{max_cycle}"
        )
        fig.update_layout(
            height=350,
            margin=dict(l=60, r=20, t=40, b=20),
            legend=dict(orientation='h', y=1.02, x=1)
        )
        line_ph.plotly_chart(fig, use_container_width=True, key=f"{sync_type}_line_{t}")

        # 2) Mostrar la(s) acción(es) ocurrida(s) en este ciclo
        acts = evs[evs['time'] == t]
        if not acts.empty:
            # Puede haber múltiples eventos en el mismo t
            textos = []
            for _, ev in acts.iterrows():
                icon = '🟢' if ev.status=='acquired' else '🔴'
                verbo = 'adquirió' if ev.status=='acquired' else 'liberó'
                textos.append(f"{icon} `{ev.pid}` **{verbo}** `{ev.resource}`")
            # los unimos en un solo markdown
            event_ph.markdown("**Eventos:**  \n" + "  \n".join(textos))
        else:
            event_ph.empty()

        time.sleep(delay)

    st.success("✅ Animación de líneas y acciones completada")

def run_animation(tl, cmax, algo, placeholder):
    if algo in ["FIFO", "SJF (no-preempt)"]:
        animate_by_process(tl, cmax, placeholder)
    else:
        animate_by_cycle(tl, cmax, placeholder)