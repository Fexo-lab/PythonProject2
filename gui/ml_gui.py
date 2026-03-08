import streamlit as st
import os
import pandas as pd
import numpy as np
import time
import shutil
import altair as alt

from Core.config import FEATURE_COLS, MODELS_DIR, DATASETS_DIR
from Core.ml.preprocessor import create_simulated_training_set, get_feature_matrix, analyze_feature_importance, get_top_features
from Core.ml.core import EvolutionCore

MODEL_DIR = MODELS_DIR
DATASET_DIR = DATASETS_DIR


def display_ml_training_page(params=None):
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)

    st.subheader("🧠 ML Training: Walk-Forward Evolution")

    # --- MODEL MANAGEMENT ---
    with st.expander("📂 Model Management", expanded=False):
        all_models = [f.replace(".pkl", "") for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
        if all_models:
            c_src, c_dest, c_do = st.columns([2, 2, 1])
            src_model = c_src.selectbox("Source Model:", all_models)
            dest_name = c_dest.text_input("Target Name:", placeholder="e.g. gold_v2")
            if c_do.button("🚀 Clone", width='stretch'):
                if dest_name:
                    shutil.copy(os.path.join(MODEL_DIR, f"{src_model}.pkl"),
                                os.path.join(MODEL_DIR, f"{dest_name}.pkl"))
                    st.rerun()

    st.divider()

    # --- CONFIGURATION ---
    col1, col2 = st.columns(2)
    safety_cap_pips = col1.slider(
        "Maximum Safety SL (Pips)", 
        50, 1000, 300,
        help="Stop-Loss distance. Larger = longer trades, riskier. Smaller = faster exit, less loss"
    )
    punishment_pips = col2.slider(
        "Entry Punishment (Pips)", 
        0, 100, 25,
        help="⚠️ DEPRECATED: No longer used (new fitness formula uses Win Rate + Profit Factor)"
    )

    csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    if not csv_files:
        st.warning("No datasets found.")
        return

    selected_csv = st.selectbox("Asset for Training:", csv_files)
    new_model_name = st.text_input("Model Name:", "silver_v9_walkforward")

    training_active = st.checkbox("🚀 Start Evolution")

    if training_active:
        with st.status("Preparing data...") as status:
            train_df, test_df = create_simulated_training_set(selected_csv, safety_cap_pips / 100000)
            if train_df is None: return
            _, features = get_feature_matrix(train_df)
            
            # Feature Importance Analysis
            X_train = train_df[features]
            y_train = train_df['target_label']
            X_test = test_df[features]
            y_test = test_df['target_label']
            
            feature_imp_dict, sorted_features = analyze_feature_importance(X_train, X_test, y_train, y_test)
            top_features_list = get_top_features(feature_imp_dict, top_n=6)
            
            status.update(label=f"Data ready! Using top {len(top_features_list)} features.", state="complete")

        # Show Feature Importance
        with st.expander("📊 Feature Importance Analysis", expanded=False):
            st.write("**Top Features for Trading Signals:**")
            feat_df = pd.DataFrame([
                {"Feature": f, "Importance": round(score, 4)} 
                for f, score in sorted_features
            ])
            st.dataframe(feat_df, use_container_width=True)

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

                # Feature Importance Buffer
                importances = curr['model'].feature_importances_
                feat_df_new = pd.DataFrame(
                    {'Feature': FEATURE_COLS, 'Importance': importances, 'Type': 'Current Record'})
                if 'old_feat_df' not in st.session_state:
                    st.session_state.old_feat_df = feat_df_new.copy()
                    st.session_state.old_feat_df['Type'] = 'Previous Record'

                st.session_state.last_feat_chart_data = pd.concat([st.session_state.old_feat_df, feat_df_new])
                st.session_state.old_feat_df = feat_df_new.copy()
                st.session_state.old_feat_df['Type'] = 'Previous Record'
            else:
                stagnation_counter += 1

            if fitness_val >= current_max_fitness:
                current_max_fitness = fitness_val
                history_data.append({"Cycle": cycle, "Fitness": fitness_val, "Event": record_label})

            # --- 1. HEADER ---
            with header_spot.container():
                st.markdown(f"#### 🧬 Evolution Running... (Stagnation: {stagnation_counter})")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Best Fitness", f"{core.best_fit:,.0f}", delta=record_label if is_better else None)
                c2.metric("Win Rate (Test)", f"{curr['wr']:.1f}%")
                c3.metric("Trades (Train)", f"L:{curr['lw']} S:{curr['sw']}")
                c4.metric("Quality Score", f"{curr.get('quality_score', 0):.1%}", 
                         help="Combination of Overfitting, Significance & Realism (0-100%)")
                
                # === QUALITY WARNINGS ===
                if curr.get('is_overfitting', False):
                    st.warning(f"⚠️ Overfitting Detected (Ratio: {curr.get('overfitting_ratio', 0):.2f}). Training too much better than test!")
                if not curr.get('is_significant', False):
                    st.info(f"ℹ️ Too Few Trades ({curr.get('test_long_total_trades', 0) + curr.get('test_short_total_trades', 0)}) for statistical significance (Min: 50)")
                if not curr.get('is_realistic', False):
                    st.warning(f"⚠️ Win Rate {curr['wr']:.1f}% might be unrealistic. Expected: 30-70%")

            # --- 2. EVENT LOG (WITH STATUS & TRADES) ---
            if is_better or cycle % 20 == 0:
                dna_entry = {
                    "Cycle": cycle,
                    "Status": "⭐ RECORD" if is_better else "🔄 Update",
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

            # --- 3. CHARTS ---
            if cycle % 5 == 0:
                with chart_area.container():
                    col_l, col_r = st.columns(2)
                    if history_data:
                        df_p = pd.DataFrame(history_data)
                        line = alt.Chart(df_p).mark_line(color='white', strokeWidth=2).encode(
                            x=alt.X('Cycle:Q', title='Cycle'),
                            y=alt.Y('Fitness:Q', title='Highscore Fitness', scale=alt.Scale(zero=False))
                        )
                        points = alt.Chart(df_p.dropna(subset=['Event'])).mark_point(color='#00ff00', size=70,
                                                                                     filled=True).encode(
                            x='Cycle:Q', y='Fitness:Q'
                        )
                        text = points.mark_text(align='left', dx=5, dy=-10, color='#00ff00', fontSize=10).encode(
                            text='Event:N')
                        col_l.altair_chart(line + points + text, width='stretch')

                    if 'last_feat_chart_data' in st.session_state:
                        f_chart = alt.Chart(st.session_state.last_feat_chart_data).mark_bar(opacity=0.6).encode(
                            x=alt.X('Feature:N', sort='-y'),
                            y=alt.Y('Importance:Q', stack=None),
                            color=alt.Color('Type:N', scale=alt.Scale(domain=['Previous Record', 'Current Record'],
                                                                     range=['#ff4b4b', '#00ff00']))
                        ).properties(height=280)
                        col_r.altair_chart(f_chart, width='stretch')

            time.sleep(0.01)
    else:
        st.info("Choose an asset and start the evolution.")
