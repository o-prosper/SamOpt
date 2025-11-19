# data_generator.py

def CTMC(purse):
    import numpy as np
    from copy import copy

    # Unpack setup
    T            = purse.T              # reporting times, increasing
    tmin         = purse.tmin
    tmax         = purse.tmax
    maxiter      = purse.maxiter
    mod_pars     = purse.mod_pars
    init_cond    = purse.init_cond
    trans_matrix = purse.trans_matrix

    alpha, beta = mod_pars[0], mod_pars[1]

    # Current CTMC state and time
    xt = np.array(init_cond, dtype=float)
    t  = float(tmin)

    # Event history (jump times & post-jump states)
    event_times  = [t]
    event_states = [xt.copy()]

    # Values at reporting times
    X_report = []

    ii = 0  # event counter
    extinct = False

    # Loop over reporting times
    # T[0] is usually tmin, so record it immediately
    for j, t_report in enumerate(T):
        if j == 0:
            # state at first reporting time
            X_report.append(xt.copy())
            continue

        # Simulate until we CROSS this reporting time or hit extinction / maxiter
        xt_prev = xt.copy()
        while (t < t_report) and (ii < maxiter):
            ii += 1

            S, I = xt[0], xt[1]
            rates = np.array([beta * S * I, alpha * I], dtype=float)
            sum_rates = rates.sum()

            if sum_rates <= 0:
                # No more events possible: process stays constant afterwards
                extinct = True
                break

            # Gillespie step
            tau = np.random.exponential(1.0 / sum_rates)
            k   = np.random.uniform(0.0, 1.0)

            # choose which event
            cdf = np.cumsum(rates / sum_rates)
            which = 0 if k <= cdf[0] else 1

            # store previous state, then apply jump
            xt_prev = xt.copy()
            t += tau
            xt += trans_matrix[which]

            # record event time & new state
            event_times.append(t)
            event_states.append(xt.copy())

        if extinct or ii >= maxiter:
            # From now on, state stays at current xt
            X_report.append(xt.copy())
            # Fill any remaining reporting times with the same state
            for _ in range(j + 1, len(T)):
                X_report.append(xt.copy())
            break

        # Here we know t >= t_report:
        # the last event happened at time t > t_report,
        # so the state AT t_report is xt_prev (just before crossing)
        X_report.append(xt_prev.copy())

    # Safety: ensure lengths match exactly
    X_report = X_report[:len(T)]
    X_report = np.asarray(X_report, dtype=float)
    event_times  = np.asarray(event_times, dtype=float)
    event_states = np.asarray(event_states, dtype=float)

    # xt is current state at last simulated event time
    return X_report, xt, event_times, event_states
