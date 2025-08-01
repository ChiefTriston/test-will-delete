import matplotlib.pyplot as plt, os

def generate_segment_plot_map(full_scores, segment_start, segment_end,
                              clip_id, tier1_trans, drift_reason, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    times = [pt['time'] for pt in full_scores]
    comps = [pt['vader_compound'] for pt in full_scores]
    plt.figure(figsize=(12,4))
    plt.plot(times, comps)
    plt.axvspan(segment_start, segment_end, color='red', alpha=0.3)
    plt.title(f"{clip_id} | {tier1_trans} | {drift_reason}")
    plt.savefig(os.path.join(save_dir,f"{clip_id}.png"))
    plt.close()

def save_drift_plot(drifts, drift_events, out_png):
    import numpy as np
    t = [d['t'] for d in drifts]
    plt.figure(figsize=(4,2))
    for key in ['Δpitch','Δenergy','Δspeech_rate','Δpause','Δcompound']:
        plt.plot(t,[d[key] for d in drifts],label=key)
    for ev in drift_events:
        plt.axvline(ev,color='r',linestyle='--')
    plt.legend(fontsize='xx-small')
    plt.savefig(out_png)
    plt.close()
