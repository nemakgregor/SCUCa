import numpy as np
import pandas as pd

from tqdm import tqdm
import gzip
import json

def prepare_statistics(initial_files, hist_files):
    statistics = []

    for init_f, hist_f in tqdm(zip(initial_files, hist_files), total=len(hist_files)):
        with gzip.open(init_f, "rt", encoding="utf-8") as f:
            data = json.load(f)
            
        with np.load(hist_f, allow_pickle=True) as npzfile:
            hist_data = {k: npzfile[k] for k in npzfile.files}
            
        hours = data['Parameters']['Time horizon (h)']
        gens = data["Generators"]
    
        for gen_id, gen in gens.items():
            p_curve = gen["Production cost curve (MW)"]
            c_curve = gen["Production cost curve ($)"]
            
            pmin = p_curve[0]
            pmax = p_curve[-1]
            
            # TODO: investigate features for power-cost curve
            avg_cost_per_mw = (c_curve[-1] - c_curve[0]) / (pmax - pmin + 1e-6)
            
            startup_costs = gen["Startup costs ($)"]
            startup_delays = gen["Startup delays (h)"]
            
            delay_min = min(startup_delays)
            delay_max = max(startup_delays)

            # TODO: investigate features for startup costs-delays curve
            cost_per_hour = []
            for i in range(len(startup_costs)):
                cost_per_hour.append(startup_costs[i] / (startup_delays[i] + 1e-6))
            avg_cost_per_hour = np.mean(cost_per_hour)
            
            ramp_up = gen["Ramp up limit (MW)"]
            ramp_down = gen["Ramp down limit (MW)"]
            
            startup_limit = gen["Startup limit (MW)"]
            shutdown_limit = gen["Shutdown limit (MW)"]
            
            min_uptime = gen["Minimum uptime (h)"]
            min_downtime = gen["Minimum downtime (h)"]       

            initial_status = gen["Initial status (h)"]
            initial_power = gen["Initial power (MW)"]
        
            is_on = hist_data[f"Is_on/{gen_id}"]        # target
            
            reserve = data["Reserves"]["r1"]["Amount (MW)"]
    
            load_per_bus = []
            for v in data["Buses"].values():
                load = v["Load (MW)"]
                if isinstance(load, list):
                    load_per_bus.append(load)
                else:
                    load_per_bus.append([load] * hours)
                    
            # TODO: investigate features for buses
            load_per_bus = np.array(load_per_bus)
            load_total = np.sum(load_per_bus, axis=0)
            
            for t in range(hours):
                statistics.append({
                    "gen_id": int(gen_id[1:]),
                    "hour": t,
                    "pmin": pmin,
                    "pmax": pmax,
                    "avg_cost_per_mw": avg_cost_per_mw,
                    "avg_startup_cost_per_hour": avg_cost_per_hour,
                    "startup_delay_min": delay_min,
                    "startup_delay_max": delay_max,
                    "ramp_up": ramp_up,
                    "ramp_down": ramp_down,
                    "startup_limit": startup_limit,
                    "shutdown_limit": shutdown_limit,
                    "min_uptime": min_uptime,
                    "min_downtime": min_downtime,
                    "initial_status": initial_status,
                    "initial_power": initial_power,
                    "load_total": load_total[t],
                    "reserve": reserve[t],
                    "is_on": is_on[t]
                    })
                
    df = pd.DataFrame(statistics)
    return df


def prepare_xy(df):
    exclude = ["is_on"]
    features = [col for col in df.columns if col not in exclude]
    X = df[features]
    y = df["is_on"]
    return X, y


def prepare_gru_dataset(initial_files, hist_files, y_labels):
    seq_features = []
    target = []

    idx = 0 

    for init_f, hist_f in tqdm(zip(initial_files, hist_files), total=len(hist_files)):
        with gzip.open(init_f, "rt", encoding="utf-8") as f:
            data = json.load(f)

        with np.load(hist_f, allow_pickle=True) as npzfile:
            hist_data = {k: npzfile[k] for k in npzfile.files}

        hours = data['Parameters']['Time horizon (h)']
        gens = data["Generators"]
        
        reserve = data["Reserves"]["r1"]["Amount (MW)"]
        
        load_per_bus = []
        for v in data["Buses"].values():
            load = v["Load (MW)"]
            if isinstance(load, list):
                load_per_bus.append(load)
            else:
                load_per_bus.append([load] * hours)
                
        load_per_bus = np.array(load_per_bus)
        load_total = np.sum(load_per_bus, axis=0)

        for gen_id, gen in gens.items():
            y_seq_labels = y_labels[idx:idx + hours]
            
            is_on_seq = []
            for t in range(hours):
                if t == 0:
                    prev_status = int(gen["Initial status (h)"] > 0)
                else:
                    prev_status = y_seq_labels[t - 1]
                is_on_seq.append(prev_status)
            idx += hours

            features = np.stack([load_total, reserve, is_on_seq], axis=-1)

            power = hist_data[f"Thermal production (MW)/{gen_id}"]

            seq_features.append(features)
            target.append(power)

    return np.array(seq_features), np.array(target)
