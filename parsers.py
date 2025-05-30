import pandas as pd

def parse_processes_txt(file) -> pd.DataFrame:
    """
    Lee procesos de un .txt con líneas: PID,arrival,burst[,priority]
    Retorna DataFrame con columnas ['pid','arrival','burst','priority']
    """
    df = pd.read_csv(
        file,
        sep=",",
        header=None,
        names=["pid","arrival","burst","priority"],
        dtype={"pid": str, "arrival": int, "burst": int, "priority": float}
    )
    df['priority'] = df['priority'].fillna(0).astype(int)
    return df


def parse_resources_txt(file) -> pd.DataFrame:
    """
    Lee recursos de un .txt con líneas: name,initial_count
    Retorna DataFrame con columnas ['resource','initial_count']
    """
    return pd.read_csv(file, sep=",", header=None, names=["resource","initial_count"] )


def parse_actions_txt(file) -> pd.DataFrame:
    """
    Lee acciones de un .txt con líneas: PID,action,resource,time
    Retorna DataFrame con columnas ['pid','action','resource','time']
    """
    return pd.read_csv(file, sep=",", header=None, names=["pid","action","resource","time"])  