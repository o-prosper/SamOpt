

# baseline.py
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SIRSetup:
    alpha: float
    beta: float
    N0: int
    I0: int
    R0: int
    C0: int
    Cobs0: int
    init_cond: np.ndarray
    mod_pars: np.ndarray
    tspan: tuple
    N: int
    dt: float
    T: np.ndarray
    tmin: float
    tmax: float
    maxiter: int            
    trans_matrix: np.ndarray

def _time_grid(tfinal: float, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive.")
    n_steps = int(np.floor((tfinal / dt) + 1e-12))
    T = dt * np.arange(n_steps + 1, dtype=float)
    if abs(T[-1] - tfinal) > 1e-12:
        T = np.append(T, float(tfinal))
    return T

def make_setup(
    beta: float = 0.002,
    alpha: float = 0.476,
    N0: int = 763,
    I0: int = 25,
    R0: int = 0,
    C0: int = 0,
    Cobs0: int = 0,
    tfinal: float = 20.0,
    dt: float = 1.0,
    maxiter: int = 10000,
):
    S0 = N0 - I0 - R0
    init_cond = np.array([S0, I0, R0, C0, Cobs0], dtype=float)
    mod_pars = np.array([alpha, beta], dtype=float)
    tspan = (0.0, float(tfinal))
    T = _time_grid(tfinal=float(tfinal), dt=float(dt))

    # Transition matrix for [S, I, R, C, Cobs]
    trans_matrix = np.array([
        [-1, +1, 0, +1, 0],   # infection
        [ 0, -1, +1, 0,  0],  # recovery
    ], dtype=float)

    return SIRSetup(
        alpha=alpha,
        beta=beta,
        N0=N0,
        I0=I0,
        R0=R0,
        C0=C0,
        Cobs0=Cobs0,
        init_cond=init_cond,
        mod_pars=mod_pars,
        tspan=tspan,
        N=S0 + I0 + R0,
        dt=float(dt),
        T=T,
        tmin=float(tspan[0]),
        tmax=float(T[-1]),
        maxiter=int(maxiter),     
        trans_matrix=trans_matrix,

    )
