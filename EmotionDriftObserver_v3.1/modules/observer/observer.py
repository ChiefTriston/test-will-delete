import streamlit as st
import json
import os
import portalocker
import numpy as np
import plotly.graph_objects as go
from modules.utils.plot_utils import plotly_trends, save_drift_plot

all_emotions = [
    'Anger', 'Anxiety', 'Contempt', 'Despair', 'Disgust', 'Fear', 'Frustration',
    'Guilt', 'Irritation', 'Jealousy', 'Loneliness', 'Negative Surprise',
    'Sadness', 'Boredom', 'Calm', 'Concentration', 'Flat narration', 'Hesitant',
    'Matter-of-fact Informational tone', 'Neutral', 'Tired', 'Amusement',
    'Enthusiasm', 'Gratitude', 'Happiness', 'Hope', 'Inspiration', 'Love',
    'Pleasant', 'Relief', 'Surprise'
]

def update_emotion_rules(feedback, rules_path):
    with open(rules_path, 'r+') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        rules = json.load(f)
        corrections = rules.setdefault('corrections', [])
        corrections.append(feedback)
        f.seek(0)
        json.dump(rules, f, indent=2)
        f.truncate()
        portalocker.unlock(f)

def run(context):
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']

    # Ensure learned_rules.json exists
    rules_path = os.path.join(output_dir, 'learned_rules.json')
    if not os.path.exists(rules_path):
        with open(rules_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump({}, f)
            portalocker.unlock(f)

    # Sidebar job summary
    st.sidebar.header("Job Summary")
    total_slices = 0
    flagged_anomalies = 0
    for spk in speaker_ids:
        spk_dir = os.path.join(output_dir, 'emotion_tags', spk)
        try:
            with open(os.path.join(spk_dir, 'transcript.json'), 'r') as tf:
                tdata = json.load(tf)
            total_slices += len(tdata.get('slices', []))
        except FileNotFoundError:
            pass
        try:
            with open(os.path.join(spk_dir, 'drift_vector.json'), 'r') as df:
                ddata = json.load(df)
            flagged_anomalies += len(ddata.get('anomalies', []))
        except FileNotFoundError:
            pass
    arc_path = os.path.join(output_dir, 'arc_classification.json')
    try:
        with open(arc_path, 'r') as af:
            arc = json.load(af)
        completed_arcs = arc.get('named_arc', 'N/A')
    except FileNotFoundError:
        completed_arcs = 'N/A'

    st.sidebar.write(f"Total Slices: {total_slices}")
    st.sidebar.write(f"Flagged Anomalies: {flagged_anomalies}")
    st.sidebar.write(f"Completed Arcs: {completed_arcs}")

    st.title("Observer: Manual Review Dashboard")

    tab1, tab2, tab3 = st.tabs(["Global Overview", "Per-Speaker Review", "Beats Overview"])

    with tab1:
        st.header("Global Prosody Overview")
        for spk in speaker_ids:
            spk_dir = os.path.join(output_dir, 'emotion_tags', spk)
            try:
                with open(os.path.join(spk_dir, 'prosody_trend.json'), 'r') as pf:
                    prosody = json.load(pf)
                time = np.array(prosody['frame_series']['time'])
                f0_z = np.array(prosody['frame_series']['f0_z'])
                energy_z = np.array(prosody['frame_series']['energy_z'])
                st.subheader(f"{spk} Prosody")
                st.plotly_chart(plotly_trends(time, f0_z, energy_z), use_container_width=True)
            except FileNotFoundError:
                continue

    with tab2:
        st.header("Per-Speaker Review")
        speaker_tab = st.selectbox("Select Speaker", speaker_ids)
        spk_dir = os.path.join(output_dir, 'emotion_tags', speaker_tab)

        try:
            with open(os.path.join(spk_dir, 'tier2_tags.json'), 'r') as tf:
                tags = json.load(tf)
            with open(os.path.join(spk_dir, 'transcript.json'), 'r') as tf:
                transcript = json.load(tf)
            with open(os.path.join(spk_dir, 'drift_vector.json'), 'r') as df:
                drift = json.load(df)
            with open(os.path.join(spk_dir, 'prosody_trend.json'), 'r') as pf:
                prosody = json.load(pf)
        except FileNotFoundError:
            tags = []
            transcript = {'slices': []}
            drift = {'deltas': [], 'anomalies': []}
            prosody = {'frame_series': {'time': [], 'f0_z': [], 'energy_z': []}}

        st.subheader("Prosody Trends")
        time = np.array(prosody['frame_series']['time'])
        f0_z = np.array(prosody['frame_series']['f0_z'])
        energy_z = np.array(prosody['frame_series']['energy_z'])
        st.plotly_chart(plotly_trends(time, f0_z, energy_z), use_container_width=True)

        st.subheader("Drift Vector")
        deltas = drift.get('deltas', [])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(deltas))), y=deltas, mode='lines', name='Delta'))
        fig.update_layout(
            title="Drift Vector",
            xaxis_title="Slice Index",
            yaxis_title="Delta"
        )
        st.plotly_chart(fig, use_container_width=True)

        num_slices = len(tags)
        page_size = 10
        total_pages = (num_slices - 1) // page_size + 1
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, num_slices)

        corrections = []
        for idx in range(start_idx, end_idx):
            tag = tags[idx]
            slice_text = transcript['slices'][idx].get('text', '')
            st.write(f"Slice {idx}: {slice_text[:100]}...")
            corrected = st.selectbox(
                f"Label (original: {tag.get('label')})", all_emotions,
                index=all_emotions.index(tag.get('label')) if tag.get('label') in all_emotions else 0,
                key=f"select_{idx}"
            )
            notes = st.text_input("Notes", key=f"notes_{idx}")
            severity = st.slider("Severity", 1, 5, 1, key=f"severity_{idx}")
            if corrected != tag.get('label') or notes or severity > 1:
                corrections.append({
                    'slice': idx,
                    'correction': corrected,
                    'original': tag.get('label'),
                    'rule_id': tag.get('rule_id'),
                    'notes': notes,
                    'severity': severity
                })

        st.subheader("Suggested Rules")
        suggested_rules = [
            f"Adjust weight for {c['original']}→{c['correction']}" for c in corrections
        ]
        for rule in suggested_rules:
            if st.checkbox(rule):
                st.write(f"Selected to apply: {rule}")

        if st.button(f"Commit Feedback for {speaker_tab}"):
            for corr in corrections:
                update_emotion_rules(corr, rules_path)
            st.success("Feedback committed!")

    with tab3:
        st.header("Beats Overview")
        plot_map_path = os.path.join(output_dir, 'plot_map.json')
        try:
            with open(plot_map_path, 'r') as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                beats = json.load(f)
                portalocker.unlock(f)
        except FileNotFoundError:
            st.error("plot_map.json not found")
            beats = []

        if beats:
            # Timeline plot of beats
            fig = go.Figure()
            for beat in beats:
                fig.add_vrect(
                    x0=beat['start'],
                    x1=beat['end'],
                    fillcolor='blue' if beat['arc_confidence'] >= 0.7 else 'red',
                    opacity=0.3,
                    annotation_text=f"{beat['beat_id']}: {beat['dominant_emotion']} ({beat['arc_confidence']:.2f})",
                    annotation_position="top left"
                )
            fig.update_layout(
                title="Narrative Beats Timeline",
                xaxis_title="Time (s)",
                yaxis_title="Beats",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display beat details
            for beat in beats:
                st.subheader(f"Beat {beat['beat_id']}: {beat['title']}")
                st.write(f"Time: {beat['start']:.2f}s - {beat['end']:.2f}s")
                st.write(f"Summary: {beat['summary']}")
                st.write(f"Speakers: {', '.join(beat['speakers'])}")
                st.write(f"Dominant Emotion: {beat['dominant_emotion']}")
                st.write(f"Arc Segment: {beat['arc_segment']} (Confidence: {beat['arc_confidence']:.2f})")
                st.write(f"Named Arc: {beat['named_arc']}")
                st.write(f"Average Prosody Score: {beat['avg_prosody_score']:.2f}")
                
                # Speaker insights
                st.write("Speaker Insights:")
                for spk, insights in beat['speaker_insights'].items():
                    st.write(f"- {spk}: Dominant Tag={insights['dominant_tag']}, "
                             f"Confidence={insights['avg_confidence']:.2f}, "
                             f"Entropy={insights['entropy']:.2f}, "
                             f"Drift={insights['avg_drift']:.2f}, "
                             f"Slope={insights['drift_slope']:.2f}")

                # Display pre-generated plot
                if beat.get('plot_path') and os.path.exists(beat['plot_path']):
                    with open(beat['plot_path'], 'r') as f:
                        st.components.v1.html(f.read(), height=400)
                else:
                    st.warning(f"No plot available for beat {beat['beat_id']}")

        else:
            st.warning("No beats available")

    return {'learned_rules': rules_path}
