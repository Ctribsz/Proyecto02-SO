import pandas as pd

def parse_processes_txt(file) -> pd.DataFrame:
    """
    Lee procesos de un .txt con líneas: PID,BT,AT,Priority
    Retorna DataFrame con columnas ['pid','burst','arrival','priority']
    """
    df = pd.read_csv(
        file,
        sep=",",
        header=None,
        names=["pid","burst","arrival","priority"],
        dtype={"pid": str, "burst": int, "arrival": int, "priority": int}
    )
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