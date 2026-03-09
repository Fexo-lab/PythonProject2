import streamlit as st
import os
import pandas as pd
import numpy as np
import time
import shutil
import altair as alt

from Core.config import FEATURE_COLS, MODELS_DIR, DATASETS_DIR
from Core.ml.preprocessor import create_simulated_training_set, get_feature_matrix
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
    col1, col2, col3 = st.columns(3)
    
    # Stop Loss Mode Selection
    sl_mode = col1.radio(
        "Stop Loss Mode",
        ["ATR-Adaptive", "Fixed %"],
        help="ATR-Adaptive: Adjusts to market volatility (Recommended)\nFixed %: Static percentage (Simpler)"
    )
    use_atr_sl = sl_mode == "ATR-Adaptive"
    
    if use_atr_sl:
        atr_multiplier = col2.slider(
            "ATR Multiplier",
            0.5, 3.0, 1.5, 0.1,
            help="Stop Loss = Entry ± (ATR × Multiplier)\n1.5 = moderate, 2.0 = wider, 1.0 = tighter"
        )
        safety_cap_pips = col3.slider(
            "Fallback SL (Pips)",
            50, 500, 200,
            help="Used if ATR unavailable"
        )
    else:
        atr_multiplier = 1.5  # Default, unused
        safety_cap_pips = col2.slider(
            "Fixed Stop Loss (%)",
            0.1, 2.0, 0.3, 0.1,
            help="Stop-Loss as % of price. 0.3% = tight, 1.0% = moderate, 2.0% = wide"
        )
        safety_cap_pips = safety_cap_pips / 100  # Convert % to decimal
        col3.empty()  # Placeholder for alignment
    
    punishment_pips = st.slider(
        "Entry Punishment (Pips)",
        0, 100, 25,
        help="⚠️ DEPRECATED: Entry cost (spread/slippage)"
    )

    csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    if not csv_files:
        st.warning("No datasets found.")
        return

    selected_csv = st.selectbox("Asset for Training:", csv_files)
    new_model_name = st.text_input("Model Name:", "silver_v9_walkforward")
    
    # Training Duration Limits
    col_t1, col_t2 = st.columns(2)
    max_cycles = col_t1.slider(
        "Max Cycles",
        100, 100000, 50000,
        help="Stop training after this many cycles (safety limit for long runs). 10h ≈ 50k cycles"
    )
    
    # Stop Mode selection
    stop_mode = col_t2.radio(
        "Stop Mode",
        ["Smart Checkpoint (1000 cycles)", "Quality Target", "Never (Until Max)"],
        help="Smart Checkpoint: Pause/Resume every 1000 cycles (Best for 10h+ runs)\nQuality Target: Stop when reaching target\nNever: Run until max cycles (no early stopping)"
    )
    
    # Target quality (only shown if Quality Target mode)
    target_quality = 0.75
    if stop_mode == "Quality Target":
        target_quality = col_t2.slider(
            "Target Quality Score",
            0.0, 1.0, 0.75, 0.05,
            help="Stop early when reaching this quality score (0-100%)",
            key="target_quality_slider"
        )

    training_active = st.checkbox("🚀 Start Evolution")

    if training_active:
        with st.status("Preparing data...") as status:
            # Handle pips vs percentage conversion
            if use_atr_sl:
                # ATR mode: convert pips to percentage for fallback
                max_sl_pct = safety_cap_pips / 100000  # Pips to %
                sl_config = f"ATR (×{atr_multiplier}), Fallback {safety_cap_pips}p"
            else:
                # Fixed % mode
                max_sl_pct = safety_cap_pips  # Already in decimal
                sl_config = f"Fixed {max_sl_pct*100:.1f}%"
            
            train_df, test_df = create_simulated_training_set(
                selected_csv,
                max_sl_pct,
                punishment_pips=punishment_pips,
                use_atr_sl=use_atr_sl,
                atr_multiplier=atr_multiplier
            )
            if train_df is None: return
            _, features = get_feature_matrix(train_df)
            status.update(label=f"Data ready! (SL: {sl_config})", state="complete")

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
        checkpoint_count = 0
        checkpoint_interval = 1000  # Every 1000 cycles
        
        # --- TIMER TRACKING ---
        training_start_time = time.time()
        cycle_times = []  # Track last 100 cycle times
        last_cycle_time = training_start_time
        
        # Timer display container
        timer_spot = st.empty()

        while training_active:
            cycle += 1
            current_time = time.time()
            
            # Track cycle time (only after first cycle)
            if cycle > 1:
                cycle_time = current_time - last_cycle_time
                cycle_times.append(cycle_time)
                if len(cycle_times) > 100:
                    cycle_times.pop(0)  # Keep only last 100 cycles
            
            last_cycle_time = current_time
            
            # Safety checks for long training runs
            if cycle > max_cycles:
                st.warning(f"⏹️ Reached max cycles ({max_cycles}). Training stopped.")
                break
            
            curr, is_better = core.run_cycle(train_df, test_df, punishment_pips, cycle, stagnation_counter)
            
            # Smart Checkpoint Logic (instead of early stopping)
            quality = curr.get('quality_score', 0)
            
            if stop_mode == "Smart Checkpoint (1000 cycles)" and cycle % checkpoint_interval == 0:
                # Smart checkpoint: pause and resume
                checkpoint_count += 1
                st.info(f"💾 Checkpoint #{checkpoint_count} reached at cycle {cycle}. Pausing for 5 seconds...")
                time.sleep(5)
                st.info(f"▶️ Resuming training from cycle {cycle}...")
                time.sleep(0.5)
            
            # Early stopping if quality target reached (only in Quality Target mode)
            elif stop_mode == "Quality Target" and quality >= target_quality and cycle > 50:
                st.success(f"✅ Target quality {target_quality:.0%} reached at cycle {cycle}. Training complete!")
                break
            
            # Never mode: just keep training until max_cycles

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

            # --- TIMER METRICS ---
            if cycle % 5 == 0 and cycle_times:
                avg_cycle_time = np.mean(cycle_times)
                elapsed_time = current_time - training_start_time
                
                # Calculate time estimates
                elapsed_h = int(elapsed_time // 3600)
                elapsed_m = int((elapsed_time % 3600) // 60)
                elapsed_s = int(elapsed_time % 60)
                
                # Last 100 cycles time
                last_100_time = sum(cycle_times)
                last_100_h = int(last_100_time // 3600)
                last_100_m = int((last_100_time % 3600) // 60)
                last_100_s = int(last_100_time % 60)
                
                # Estimate remaining time
                cycles_remaining = max_cycles - cycle
                est_remaining_time = cycles_remaining * avg_cycle_time
                est_h = int(est_remaining_time // 3600)
                est_m = int((est_remaining_time % 3600) // 60)
                est_s = int(est_remaining_time % 60)
                
                with timer_spot.container():
                    st.markdown("### ⏱️ Training Timer")
                    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                    
                    col_t1.metric(
                        "Elapsed",
                        f"{elapsed_h}h {elapsed_m}m {elapsed_s}s",
                        help="Total time since training started"
                    )
                    col_t2.metric(
                        "Last 100 Cycles",
                        f"{last_100_h}h {last_100_m}m {last_100_s}s",
                        delta=f"{avg_cycle_time:.2f}s/cycle",
                        help="Time for last 100 cycles + avg per cycle"
                    )
                    col_t3.metric(
                        "Est. Remaining",
                        f"{est_h}h {est_m}m {est_s}s",
                        help=f"Estimated time to reach {max_cycles} cycles"
                    )
                    col_t4.metric(
                        "Progress",
                        f"{cycle}/{max_cycles}",
                        delta=f"{(cycle/max_cycles)*100:.1f}%",
                        help="Cycles completed"
                    )

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
                # Calculate raw profit from pips (independent of ML)
                long_pips = curr.get('long_total_pips', 0)
                short_pips = curr.get('short_total_pips', 0)
                total_pips = long_pips + short_pips
                
                dna_entry = {
                    "Cycle": cycle,
                    "Status": "⭐ RECORD" if is_better else "🔄 Update",
                    "Total-Fit": f"{curr['total_fit']:,.0f}",
                    "Test-Fit": f"{curr['test_fit']:,.0f}",
                    "WR%": f"{curr['wr']:.1f}%",
                    "PF": f"{curr.get('profit_factor', 0):.2f}",
                    "Profit(Pips)": f"{total_pips:+.0f}",  # Raw profit from test trades
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
