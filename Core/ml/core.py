import os, joblib, json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from Core.config import MODELS_DIR


class EvolutionCore:
    def __init__(self, model_name, features_count):
        self.model_name, self.features_count = model_name, features_count
        self.model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        self.dna_path = os.path.join(MODELS_DIR, f"{model_name}_dna.json")
        self.best_fit = -1e9
        self.population = self._initialize_population()

    def _initialize_population(self):
        # DNA besteht aus: 9 Weights + 9 Biases + Hyperparameter (max_depth, n_estimators)
        pop = []
        for _ in range(15):
            dna = {
                'weights': np.random.uniform(0.1, 3.0, self.features_count),      # Relative Multiplikatoren!
                'biases': np.random.uniform(-1.0, 1.0, self.features_count),     # Additive Offsets
                'max_depth': np.random.randint(4, 16),
                'n_estimators': np.random.randint(10, 100)
            }
            pop.append(dna)
        
        # Beste bisherige DNA laden
        if os.path.exists(self.dna_path):
            try:
                with open(self.dna_path, 'r') as f:
                    best_dna = json.load(f)
                    if 'weights' in best_dna:
                        pop[0] = {
                            'weights': np.array(best_dna.get('weights', pop[0]['weights'])),
                            'biases': np.array(best_dna.get('biases', pop[0]['biases'])),
                            'max_depth': best_dna.get('max_depth', pop[0]['max_depth']),
                            'n_estimators': best_dna.get('n_estimators', pop[0]['n_estimators'])
                        }
                        self.best_fit = best_dna.get('fitness', -1e9)
            except:
                pass
        return pop

    def run_cycle(self, train_df, test_df, punishment_pips, cycle, stagnation_counter):
        from Core.config import FEATURE_COLS
        X_train, y_train, pips_train = train_df[FEATURE_COLS], train_df['target_label'], train_df['pip_result'].values
        X_test, y_test, pips_test = test_df[FEATURE_COLS], test_df['target_label'], test_df['pip_result'].values

        fitness_scores = []

        for dna in self.population:
            # Wende Weights und Biases auf die Daten an
            weights = dna['weights']
            biases = dna['biases']
            max_depth = dna['max_depth']
            n_estimators = dna['n_estimators']
            
            X_train_scaled = (X_train.values * weights) + biases
            X_test_scaled = (X_test.values * weights) + biases

            # Modell mit evolvierten Hyperparametern
            model = RandomForestClassifier(
                n_estimators=int(n_estimators), 
                max_depth=int(max_depth), 
                n_jobs=-1, 
                random_state=None
            )
            model.fit(X_train_scaled, y_train)

            # === TRAIN-SET EVALUIERUNG ===
            probs = model.predict_proba(X_train_scaled)
            if probs.shape[1] > 1:
                preds_train = model.classes_[np.argmax(probs, axis=1)]
            else:
                preds_train = np.zeros(len(X_train), dtype=int)
            
            # Separieren: Long (1) und Short (2) Trades
            long_idx = np.where(preds_train == 1)[0]
            short_idx = np.where(preds_train == 2)[0]
            
            # Long Statistiken
            long_pips = pips_train[long_idx] if len(long_idx) > 0 else np.array([])
            long_winners = np.sum(long_pips > 0) if len(long_pips) > 0 else 0
            long_total_pips = np.sum(long_pips) if len(long_pips) > 0 else 0
            
            # Short Statistiken
            short_pips = pips_train[short_idx] if len(short_idx) > 0 else np.array([])
            short_winners = np.sum(short_pips > 0) if len(short_pips) > 0 else 0
            short_total_pips = np.sum(short_pips) if len(short_pips) > 0 else 0
            
            total_trades = len(long_idx) + len(short_idx)
            total_pips_train = long_total_pips + short_total_pips
            
            # === NEUE FITNESS LOGIK für TRAIN (ähnlich wie TEST) ===
            if total_trades > 0:
                # Win Rate im Training
                total_winners = long_winners + short_winners
                train_wr = (total_winners / total_trades) * 100
                
                # Profit Factor im Training
                all_train_pips = np.concatenate([long_pips, short_pips]) if len(long_pips) > 0 or len(short_pips) > 0 else np.array([])
                train_winners = all_train_pips[all_train_pips > 0]
                train_losers = all_train_pips[all_train_pips < 0]
                
                if len(train_winners) > 0 and len(train_losers) > 0:
                    train_avg_win = np.mean(train_winners)
                    train_avg_loss = abs(np.mean(train_losers))
                    train_profit_factor = train_avg_win / train_avg_loss if train_avg_loss > 0 else 1.0
                else:
                    train_profit_factor = 1.0 if len(train_winners) > 0 else 0.0
                
                train_fit = total_pips_train * 0.3  # Actual Pips zu 30%
                train_fit += max(0, (train_wr - 50) * 100)  # Win Rate Bonus
                train_fit += max(0, (train_profit_factor - 1.0) * 500)  # Profit Factor Bonus
                
                if total_trades >= 100:
                    train_fit += 1000
                elif total_trades >= 50:
                    train_fit += 500
            else:
                train_fit = -10000  # Keine Trades = sehr schlecht

            # === TEST-SET EVALUIERUNG ===
            t_probs = model.predict_proba(X_test_scaled)
            if t_probs.shape[1] > 1:
                t_preds = model.classes_[np.argmax(t_probs, axis=1)]
            else:
                t_preds = np.zeros(len(X_test), dtype=int)
            
            # Separieren: Long und Short im Test-Set
            t_long_idx = np.where(t_preds == 1)[0]
            t_short_idx = np.where(t_preds == 2)[0]
            
            # Long Statistiken im Test-Set
            t_long_pips = pips_test[t_long_idx] if len(t_long_idx) > 0 else np.array([])
            t_long_winners = np.sum(t_long_pips > 0) if len(t_long_pips) > 0 else 0
            t_long_total_pips = np.sum(t_long_pips) if len(t_long_pips) > 0 else 0
            
            # Short Statistiken im Test-Set
            t_short_pips = pips_test[t_short_idx] if len(t_short_idx) > 0 else np.array([])
            t_short_winners = np.sum(t_short_pips > 0) if len(t_short_pips) > 0 else 0
            t_short_total_pips = np.sum(t_short_pips) if len(t_short_pips) > 0 else 0
            
            t_total_trades = len(t_long_idx) + len(t_short_idx)
            t_total_pips = t_long_total_pips + t_short_total_pips
            
            # === NEUE FITNESS LOGIK: Profit Factor & Win Rate fokussiert ===
            if t_total_trades > 0:
                # Win Rate
                t_total_winners = t_long_winners + t_short_winners
                win_rate = (t_total_winners / t_total_trades) * 100
                
                # Profit Factor Berechnung
                all_pips = np.concatenate([t_long_pips, t_short_pips]) if len(t_long_pips) > 0 or len(t_short_pips) > 0 else np.array([])
                winners = all_pips[all_pips > 0]
                losers = all_pips[all_pips < 0]
                
                if len(winners) > 0 and len(losers) > 0:
                    avg_win = np.mean(winners)
                    avg_loss = abs(np.mean(losers))
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0
                else:
                    profit_factor = 1.0 if len(winners) > 0 else 0.0
                
                # NEUE FITNESS FORMEL (statt einfacher Pips Bestrafung):
                # 1. Basis: Actual Pips (nur 0.3x gewichtet, da zu volatil)
                # 2. +Bonus: Win Rate über 50% (jedes % extra = +100 points)
                # 3. +Bonus: Profit Factor über 1.0 (jede 0.1x = +500 points)
                # 4. +Basis: Trade Count (nur 20 trades minimum für Signifikanz)
                
                test_fit = t_total_pips * 0.3  # Actual Pips zählen, aber nur zu 30%
                
                # Win Rate Bonus: 50% = 0 points, 60% = 1000 points, 70% = 2000 points
                win_rate_bonus = max(0, (win_rate - 50) * 100)
                test_fit += win_rate_bonus
                
                # Profit Factor Bonus: 1.0 = 0 points, 1.5 = 250 points, 2.0 = 500 points
                profit_factor_bonus = max(0, (profit_factor - 1.0) * 500)
                test_fit += profit_factor_bonus
                
                # Trade Count: Mehr Trades = höheres Vertrauen (aber nicht unbegrenzt)
                if t_total_trades >= 100:
                    test_fit += 1000  # Bonus für statistische Signifikanz
                elif t_total_trades >= 50:
                    test_fit += 500
            else:
                test_fit = -10000
                win_rate = 0.0
                profit_factor = 0.0

            # === GESAMT FITNESS: 70% Test, 30% Train ===
            total_fit = (0.7 * test_fit) + (0.3 * train_fit)
            
            # === QUALITY METRICS ===
            from Core.config import OVERFIT_THRESHOLD, MIN_TRADES_FOR_SIGNIFICANCE, REALISTIC_WR_MIN, REALISTIC_WR_MAX, REALISTIC_PF_MIN
            
            # Overfitting Detection
            overfitting_ratio = abs(train_fit - test_fit) / (abs(test_fit) + 1)
            is_overfitting = overfitting_ratio > OVERFIT_THRESHOLD
            
            # Statistical Significance
            is_significant = t_total_trades >= MIN_TRADES_FOR_SIGNIFICANCE
            
            # Realistic Check
            is_realistic = (REALISTIC_WR_MIN <= win_rate <= REALISTIC_WR_MAX) and (profit_factor >= REALISTIC_PF_MIN)
            
            # Feature Importance (größte Weights zuerst)
            weight_importance = np.abs(weights / np.sum(np.abs(weights)) * 100)
            
            # Quality Score (0-1)
            quality_score = (1.0 if not is_overfitting else 0.5) * \
                           (1.0 if is_significant else 0.7) * \
                           (1.0 if is_realistic else 0.5)

            fitness_scores.append({
                'total_fit': total_fit, 
                'test_fit': test_fit,
                'train_fit': train_fit,
                'fit': train_fit,  # Für Kompatibilität
                'weights': weights, 
                'biases': biases,
                'max_depth': max_depth, 
                'n_estimators': n_estimators, 
                'model': model,
                # RÜCKWÄRTS-KOMPATIBLE Felder (für GUI)
                'lw': len(long_idx),      # Long Trade Count
                'sw': len(short_idx),     # Short Trade Count
                'wr': win_rate,           # Win Rate %
                # NEUE DETAILLIERTE Felder
                'long_winners': long_winners,
                'long_total_trades': len(long_idx),
                'long_total_pips': long_total_pips,
                'short_winners': short_winners,
                'short_total_trades': len(short_idx),
                'short_total_pips': short_total_pips,
                'test_long_winners': t_long_winners,
                'test_long_total_trades': len(t_long_idx),
                'test_long_total_pips': t_long_total_pips,
                'test_short_winners': t_short_winners,
                'test_short_total_trades': len(t_short_idx),
                'test_short_total_pips': t_short_total_pips,
                'profit_factor': profit_factor,
                # QUALITY METRICS
                'overfitting_ratio': overfitting_ratio,
                'is_overfitting': is_overfitting,
                'is_significant': is_significant,
                'is_realistic': is_realistic,
                'quality_score': quality_score,
                'feature_importance': weight_importance.tolist()
            })

        # Sortieren nach Fitness
        fitness_scores.sort(key=lambda x: x['total_fit'], reverse=True)
        curr = fitness_scores[0]

        is_better = curr['total_fit'] > self.best_fit
        if is_better:
            self.best_fit = curr['total_fit']
            joblib.dump(curr['model'], self.model_path)
            with open(self.dna_path, 'w') as f:
                json.dump({
                    'weights': curr['weights'].tolist(),
                    'biases': curr['biases'].tolist(),
                    'max_depth': int(curr['max_depth']),
                    'n_estimators': int(curr['n_estimators']),
                    'fitness': self.best_fit,
                    'test_fit': curr['test_fit'],
                    'train_fit': curr['train_fit']
                }, f)

        # Evolution
        self.population = self._evolve_custom(fitness_scores)

        return curr, is_better

    def _evolve_custom(self, scores):
        # 1. Die 2 besten Eltern übernehmen
        parent1_dna = scores[0]
        parent2_dna = scores[1]

        new_population = []

        # 1. Beide Eltern direkt in die neue Generation (Elitism)
        new_population.append({
            'weights': parent1_dna['weights'].copy(),
            'biases': parent1_dna['biases'].copy(),
            'max_depth': parent1_dna['max_depth'],
            'n_estimators': parent1_dna['n_estimators']
        })
        new_population.append({
            'weights': parent2_dna['weights'].copy(),
            'biases': parent2_dna['biases'].copy(),
            'max_depth': parent2_dna['max_depth'],
            'n_estimators': parent2_dna['n_estimators']
        })

        # 2. Intelligentes Crossover (Hybrid aus beiden Eltern)
        crossover_mask = np.random.rand(len(parent1_dna['weights'])) > 0.5
        child_hybrid = {
            'weights': np.where(crossover_mask, parent1_dna['weights'], parent2_dna['weights']).copy(),
            'biases': np.where(crossover_mask, parent1_dna['biases'], parent2_dna['biases']).copy(),
            'max_depth': parent1_dna['max_depth'] if np.random.rand() > 0.5 else parent2_dna['max_depth'],
            'n_estimators': parent1_dna['n_estimators'] if np.random.rand() > 0.5 else parent2_dna['n_estimators']
        }
        new_population.append(child_hybrid)

        # 3. 10 aggressive mutierte Varianten der besten Eltern
        for i in range(10):
            parent_dna = parent1_dna if i % 2 == 0 else parent2_dna
            
            # Mutation stärke variiert (20-40% statt 15-30%)
            mutation_strength = 0.2 + (i * 0.03)
            
            mutant = {
                'weights': parent_dna['weights'].copy(),
                'biases': parent_dna['biases'].copy(),
                'max_depth': parent_dna['max_depth'],
                'n_estimators': parent_dna['n_estimators']
            }
            
            # Weights mutieren (jetzt nur 0.1-3.0 Bereich)
            weight_mask = np.random.rand(len(mutant['weights'])) < 0.5  # 50% der Weights
            mutant['weights'][weight_mask] *= (1 + np.random.normal(0, mutation_strength, np.sum(weight_mask)))
            mutant['weights'] = np.clip(mutant['weights'], 0.1, 3.0)
            
            # Biases mutieren (-1 bis 1)
            bias_mask = np.random.rand(len(mutant['biases'])) < 0.5
            mutant['biases'][bias_mask] += np.random.normal(0, mutation_strength * 0.5, np.sum(bias_mask))
            mutant['biases'] = np.clip(mutant['biases'], -1.0, 1.0)
            
            # Hyperparameter mutieren
            mutant['max_depth'] = int(np.clip(mutant['max_depth'] + np.random.randint(-2, 3), 4, 16))
            mutant['n_estimators'] = int(np.clip(mutant['n_estimators'] + np.random.randint(-10, 11), 10, 100))
            
            new_population.append(mutant)

        # 4. Nur 2 komplett neue Probanden für Diversität
        for _ in range(2):
            random_dna = {
                'weights': np.random.uniform(0.1, 3.0, len(parent1_dna['weights'])),
                'biases': np.random.uniform(-1.0, 1.0, len(parent1_dna['weights'])),
                'max_depth': np.random.randint(4, 16),
                'n_estimators': np.random.randint(10, 100)
            }
            new_population.append(random_dna)

        return new_population
