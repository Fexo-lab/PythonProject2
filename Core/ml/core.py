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
            
            # Fitness für TRAIN: Pips - Strafe
            if total_trades > 0:
                train_fit = total_pips_train - (total_trades * punishment_pips)
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
            
            # Fitness für TEST: Out-of-Sample Performance ist wichtiger!
            if t_total_trades > 0:
                test_fit = t_total_pips - (t_total_trades * punishment_pips)
                # Win Rate: (Gewinnende Trades / Gesamte Trades) * 100
                t_total_winners = t_long_winners + t_short_winners
                win_rate = (t_total_winners / t_total_trades) * 100
                
                # Bonus/Malus: Profit Factor (Gewinn-Verlust-Verhältnis)
                all_pips = np.concatenate([t_long_pips, t_short_pips]) if len(t_long_pips) > 0 or len(t_short_pips) > 0 else np.array([])
                winners = all_pips[all_pips > 0]
                losers = all_pips[all_pips < 0]
                
                if len(winners) > 0 and len(losers) > 0:
                    avg_win = np.mean(winners)
                    avg_loss = abs(np.mean(losers))
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0
                    # Bonus für guten Profit Factor
                    test_fit = test_fit + (profit_factor * 50)
                else:
                    profit_factor = 1.0 if len(winners) > 0 else 0.0
            else:
                test_fit = -10000
                win_rate = 0.0
                profit_factor = 0.0

            # === GESAMT FITNESS: 70% Test, 30% Train ===
            total_fit = (0.7 * test_fit) + (0.3 * train_fit)

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
                'profit_factor': profit_factor
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
