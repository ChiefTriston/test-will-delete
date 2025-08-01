# modules/utils/plot_utils.py

import os
import plotly.graph_objects as go


def generate_segment_plot_map(full_scores, segment_start, segment_end, clip_id, tier1_transition, drift_reason, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    times = [pt['time'] for pt in full_scores]
    comps = [pt['vader_compound'] for pt in full_scores]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=comps, mode='lines', name='Compound Score'))
    fig.add_shape(
        type='rect',
        x0=segment_start,
        x1=segment_end,
        y0=min(comps, default=0),
        y1=max(comps, default=0),
        fillcolor='red',
        opacity=0.3,
        line_width=0
    )
    fig.update_layout(
        title=f"{clip_id} | {tier1_transition} | {drift_reason}",
        xaxis_title="Time (s)",
        yaxis_title="Compound Score",
        showlegend=False
    )

    html_path = os.path.join(save_dir, f"{clip_id}.html")
    fig.write_html(html_path)
    return fig


def save_drift_plot(drifts, drift_events, out_png):
    save_dir = os.path.dirname(out_png)
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(out_png))[0]
    html_path = os.path.join(save_dir, f"{base}.html")

    fig = go.Figure()
    for key in ['Δpitch', 'Δenergy', 'Δspeech_rate', 'Δpause', 'Δcompound']:
        vals = [d.get(key, 0) for d in drifts]
        fig.add_trace(go.Scatter(x=[d['t'] for d in drifts], y=vals, mode='lines', name=key))
    for ev in drift_events:
        fig.add_vline(x=ev, line=dict(color='red', dash='dash'), annotation_text='Drift', annotation_position='top')
    fig.update_layout(
        title="Drift Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        legend_title="Metrics"
    )

    fig.write_html(html_path)
    return fig


def plotly_trends(time, f0_z, energy_z):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=f0_z, mode='lines', name='F0 Z'))
    fig.add_trace(go.Scatter(x=time, y=energy_z, mode='lines', name='Energy Z'))
    fig.update_layout(
        title="Prosody Trends",
        xaxis_title="Time (s)",
        yaxis_title="Z-Normalized Value",
        legend_title="Features"
    )
    return fig
