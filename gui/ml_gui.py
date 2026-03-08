import streamlit as st
import os
import pandas as pd
import numpy as np
import time
import shutil
import altair as alt

from Core.config import FEATURE_COLS
from Core.ml_preprocessor import create_simulated_training_set, get_feature_matrix
from Core.ml_core import EvolutionCore

MODEL_DIR = "models"
DATASET_DIR = "datasets"


def display_ml_training_page(params=None):
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)

    st.subheader("🧠 ML Training: Walk-Forward Evolution")

    # --- MODELL-VERWALTUNG ---
    with st.expander("📂 Modell-Verwaltung", expanded=False):
        all_models = [f.replace(".pkl", "") for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
        if all_models:
            c_src, c_dest, c_do = st.columns([2, 2, 1])
            src_model = c_src.selectbox("Quell-Modell:", all_models)
            dest_name = c_dest.text_input("Ziel-Name:", placeholder="z.B. gold_v2")
            if c_do.button("🚀 Klonen", width='stretch'):
                if dest_name:
                    shutil.copy(os.path.join(MODEL_DIR, f"{src_model}.pkl"),
                                os.path.join(MODEL_DIR, f"{dest_name}.pkl"))
                    st.rerun()

    st.divider()

    # --- KONFIGURATION ---
    col1, col2 = st.columns(2)
    safety_cap_pips = col1.slider("Maximaler Safety SL (Pips)", 50, 1000, 300)
    punishment_pips = col2.slider("Einstiegs-Punishment (Pips)", 0, 100, 25)

    csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    if not csv_files:
        st.warning("Keine Datensätze gefunden.")
        return

    selected_csv = st.selectbox("Asset für Training:", csv_files)
    new_model_name = st.text_input("Modellname:", "silver_v9_walkforward")

    training_active = st.checkbox("🚀 Evolution starten")

    if training_active:
        with st.status("Daten werden vorbereitet...") as status:
            train_df, test_df = create_simulated_training_set(selected_csv, safety_cap_pips / 100000)
            if train_df is None: return
            _, features = get_feature_matrix(train_df)
            status.update(label="Daten bereit!", state="complete")

        core = EvolutionCore(new_model_name, len(features))

        # --- STATISCHE PLATZHALTER ---
        header_spot = st.empty()
        log_spot = st.empty()
        st.divider()
        chart_area = st.empty()

        history_data = []
        event_logs = []
        stagnation_counter = 0
        current_max_fitness = -np.inf
        cycle = 0

        while training_active:
            cycle += 1
            curr, is_better = core.run_cycle(train_df, test_df, punishment_pips, cycle, stagnation_counter)

            fitness_val = curr['total_fit']
            record_label = None

            if is_better:
                if 'last_best_fit' in st.session_state and st.session_state.last_best_fit != 0:
                    diff_pct = ((fitness_val - st.session_state.last_best_fit) / abs(
                        st.session_state.last_best_fit)) * 100
                else:
                    diff_pct = 0.0

                st.session_state.last_best_fit = fitness_val
                record_label = f"+{diff_pct:.1f}%"
                stagnation_counter = 0

                # Feature Importance Puffer
                importances = curr['model'].feature_importances_
                feat_df_new = pd.DataFrame(
                    {'Feature': FEATURE_COLS, 'Wichtigkeit': importances, 'Typ': 'Aktueller Rekord'})
                if 'old_feat_df' not in st.session_state:
                    st.session_state.old_feat_df = feat_df_new.copy()
                    st.session_state.old_feat_df['Typ'] = 'Vorheriger Rekord'

                st.session_state.last_feat_chart_data = pd.concat([st.session_state.old_feat_df, feat_df_new])
                st.session_state.old_feat_df = feat_df_new.copy()
                st.session_state.old_feat_df['Typ'] = 'Vorheriger Rekord'
            else:
                stagnation_counter += 1

            if fitness_val >= current_max_fitness:
                current_max_fitness = fitness_val
                history_data.append({"Zyklus": cycle, "Fitness": fitness_val, "Event": record_label})

            # --- 1. HEADER ---
            with header_spot.container():
                st.markdown(f"#### 🧬 Evolution aktiv... (Stagnation: {stagnation_counter})")
                c1, c2, c3 = st.columns(3)
                c1.metric("Beste Fitness", f"{core.best_fit:,.0f}", delta=record_label if is_better else None)
                c2.metric("Winrate (Test)", f"{curr['wr']:.1f}%")
                c3.metric("Trades (Train)", f"L:{curr['lw']} S:{curr['sw']}")

            # --- 2. LOG-BUCH (MIT STATUS & TRADES) ---
            if is_better or cycle % 20 == 0:
                dna_entry = {
                    "Cycle": cycle,
                    "Status": "⭐ REKORD" if is_better else "🔄 Update",
                    "Total-Fit": f"{curr['total_fit']:,.0f}",
                    "Test-Fit": f"{curr['test_fit']:,.0f}",
                    "WR%": f"{curr['wr']:.1f}%",
                    "PF": f"{curr.get('profit_factor', 0):.2f}",
                    "L": f"{curr['long_total_trades']}T/{curr['long_winners']}W",
                    "S": f"{curr['short_total_trades']}T/{curr['short_winners']}W"
                }
                for f_idx, f_name in enumerate(FEATURE_COLS):
                    dna_entry[f_name] = round(curr['weights'][f_idx], 2)

                event_logs.insert(0, dna_entry)
                if len(event_logs) > 12: event_logs.pop()
                log_spot.dataframe(pd.DataFrame(event_logs), width='stretch', hide_index=True)

            # --- 3. DIAGRAMME ---
            if cycle % 5 == 0:
                with chart_area.container():
                    col_l, col_r = st.columns(2)
                    if history_data:
                        df_p = pd.DataFrame(history_data)
                        line = alt.Chart(df_p).mark_line(color='white', strokeWidth=2).encode(
                            x=alt.X('Zyklus:Q', title='Zyklus'),
                            y=alt.Y('Fitness:Q', title='Highscore Fitness', scale=alt.Scale(zero=False))
                        )
                        points = alt.Chart(df_p.dropna(subset=['Event'])).mark_point(color='#00ff00', size=70,
                                                                                     filled=True).encode(
                            x='Zyklus:Q', y='Fitness:Q'
                        )
                        text = points.mark_text(align='left', dx=5, dy=-10, color='#00ff00', fontSize=10).encode(
                            text='Event:N')
                        col_l.altair_chart(line + points + text, width='stretch')

                    if 'last_feat_chart_data' in st.session_state:
                        f_chart = alt.Chart(st.session_state.last_feat_chart_data).mark_bar(opacity=0.6).encode(
                            x=alt.X('Feature:N', sort='-y'),
                            y=alt.Y('Wichtigkeit:Q', stack=None),
                            color=alt.Color('Typ:N', scale=alt.Scale(domain=['Vorheriger Rekord', 'Aktueller Rekord'],
                                                                     range=['#ff4b4b', '#00ff00']))
                        ).properties(height=280)
                        col_r.altair_chart(f_chart, width='stretch')

            time.sleep(0.01)
    else:
        st.info("Wähle ein Asset und starte die Evolution.")
