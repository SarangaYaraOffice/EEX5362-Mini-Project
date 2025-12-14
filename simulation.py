import math, random
import pandas as pd

def _pick_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    for c in df.columns:
        cl = c.lower()
        for name in candidates:
            if name.lower() in cl:
                return c
    return None

def _to_dt(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.Series([pd.NaT] * len(s))

def _safe_float_series(s):
    try:
        x = pd.to_numeric(s, errors="coerce")
        return x
    except Exception:
        return pd.Series([math.nan] * len(s))

def load_inputs(xlsx_path: str):
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print("error: cannot read excel:", e)
        return None

    time_candidates = [
        "RequestTime","Request_Time","OrderTime","Order_Time","PickupTime","Pickup_Time",
        "StartTime","Start_Time","DispatchTime","Dispatch_Time","CreatedAt","Created_At",
        "Datetime","DateTime","Timestamp","Time","Date"
    ]
    service_candidates = [
        "Delivery_Time_Min","DeliveryTimeMin","DeliveryTime","Travel_Time_Min","TravelTimeMin",
        "Time_Min","TimeMin","Duration_Min","DurationMin","Delivery_Time","Travel_Time","Minutes"
    ]
    server_candidates = ["Vehicle_ID","VehicleID","Vehicle","Driver_ID","DriverID"]

    tcol = _pick_col(df, time_candidates)
    scol = _pick_col(df, service_candidates)
    vcol = _pick_col(df, server_candidates)

    if scol is None:
        print("warn: service-time column not found. Add a column like Delivery_Time_Min (minutes).")
        return None

    service_min = _safe_float_series(df[scol]).dropna()
    service_min = service_min[service_min > 0]
    if service_min.empty:
        print("warn: service-time column found but no valid positive values.")
        return None

    servers = None
    if vcol is not None:
        try:
            servers = int(df[vcol].nunique())
        except Exception:
            servers = None

    arrivals = None
    if tcol is not None:
        dt = _to_dt(df[tcol])
        dt = dt.dropna().sort_values()
        if len(dt) >= 3:
            diffs = dt.diff().dt.total_seconds().dropna() / 60.0
            diffs = diffs[(diffs > 0) & (diffs < diffs.quantile(0.99))]
            if len(diffs) >= 3:
                mean_iat = float(diffs.mean())
                arrivals = {"mean_interarrival_min": mean_iat, "lambda_per_min": 1.0 / mean_iat}

    if arrivals is None:
        print("warn: arrival timestamps not usable. Using fallback λ from count per day assumption.")
        n = len(df)
        days_guess = 21.0
        lam = n / (days_guess * 24.0 * 60.0)
        arrivals = {"mean_interarrival_min": 1.0 / lam if lam > 0 else 10.0, "lambda_per_min": lam}

    return {
        "df": df,
        "service_min": service_min.tolist(),
        "lambda_per_min": float(arrivals["lambda_per_min"]),
        "mean_interarrival_min": float(arrivals["mean_interarrival_min"]),
        "servers_guess": servers if servers and servers > 0 else 3
    }

def simulate_mgc(xlsx_path: str, c: int | None = None, horizon_min: float = 8 * 60, seed: int = 7, service_speed: float = 1.0):
    try:
        import simpy
    except Exception as e:
        print("error: simpy not installed in your environment:", e)
        return None

    inp = load_inputs(xlsx_path)
    if inp is None:
        return None

    random.seed(seed)
    lam = inp["lambda_per_min"]
    mean_iat = inp["mean_interarrival_min"]
    service_samples = inp["service_min"]
    servers = int(c if c and c > 0 else inp["servers_guess"])

    waits = []
    systems = []
    busy_time = [0.0] * servers
    served = 0
    arrived = 0

    def sample_service():
        x = random.choice(service_samples)
        x = float(x) / float(service_speed if service_speed > 0 else 1.0)
        return max(0.01, x)

    def arrival_process(env, res):
        nonlocal arrived
        while True:
            try:
                iat = random.expovariate(lam) if lam > 0 else mean_iat
                yield env.timeout(max(0.01, float(iat)))
                arrived += 1
                env.process(job(env, res, arrived))
            except Exception as e:
                print("warn: arrival loop issue:", e)
                yield env.timeout(1)

    def job(env, res, jid):
        nonlocal served
        t0 = env.now
        with res.request() as req:
            yield req
            t1 = env.now
            wq = t1 - t0
            st = sample_service()
            waits.append(wq)
            yield env.timeout(st)
            systems.append((env.now - t0))
            served += 1

    def utilization_tracker(env, res):
        while True:
            try:
                in_use = res.count
                dt = 1.0
                for i in range(min(in_use, servers)):
                    busy_time[i] += dt
                yield env.timeout(dt)
            except Exception as e:
                print("warn: util tracker issue:", e)
                yield env.timeout(1)

    try:
        env = simpy.Environment()
        res = simpy.Resource(env, capacity=servers)
        env.process(arrival_process(env, res))
        env.process(utilization_tracker(env, res))
        env.run(until=float(horizon_min))
    except Exception as e:
        print("error: simulation failed:", e)
        return None

    avg_wq = sum(waits) / len(waits) if waits else 0.0
    avg_w = sum(systems) / len(systems) if systems else 0.0
    throughput = served / horizon_min if horizon_min > 0 else 0.0
    rho = (lam / (servers * (1.0 / (sum(service_samples) / len(service_samples)))) ) if servers > 0 else 0.0
    util = (sum(busy_time) / (servers * horizon_min)) if servers > 0 and horizon_min > 0 else 0.0

    return {
        "c": servers,
        "lambda_per_min": lam,
        "mean_interarrival_min": mean_iat,
        "avg_service_min_empirical": sum(service_samples) / len(service_samples),
        "avg_wait_queue_min_Wq": avg_wq,
        "avg_time_in_system_min_W": avg_w,
        "throughput_per_min": throughput,
        "utilization_sim": util,
        "rho_approx": rho,
        "arrived": arrived,
        "served": served,
        "horizon_min": horizon_min
    }

def run_scenarios(xlsx_path: str):
    base = simulate_mgc(xlsx_path, c=None, horizon_min=8*60, seed=7, service_speed=1.0)
    if not base:
        return
    c_base = base["c"]
    s1 = base
    s2 = simulate_mgc(xlsx_path, c=c_base + 1, horizon_min=8*60, seed=7, service_speed=1.0)
    s3 = simulate_mgc(xlsx_path, c=c_base, horizon_min=8*60, seed=7, service_speed=1.15)

    def fmt(x):
        try:
            return f"{x:.3f}"
        except Exception:
            return str(x)

    rows = [s for s in [s1, s2, s3] if s]
    print("DES (M/G/c) Simulation Results")
    print("-"*90)
    print(f"{'Scenario':<28} {'c':>3} {'Wq(min)':>10} {'W(min)':>10} {'X(/min)':>10} {'Util':>8} {'λ(/min)':>10}")
    print("-"*90)
    for i, r in enumerate(rows):
        name = ["Baseline", "Add 1 vehicle", "15% faster service"][i]
        print(f"{name:<28} {r['c']:>3} {fmt(r['avg_wait_queue_min_Wq']):>10} {fmt(r['avg_time_in_system_min_W']):>10} {fmt(r['throughput_per_min']):>10} {fmt(r['utilization_sim']):>8} {fmt(r['lambda_per_min']):>10}")
    print("-"*90)

# Example:
# run_scenarios("/mnt/data/sri_lanka_delivery_dataset-221428573 (1) (3).xlsx")
