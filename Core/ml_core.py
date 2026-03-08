import os, joblib, json, numpy as np
from sklearn.ensemble import RandomForestClassifier


class EvolutionCore:
    def __init__(self, model_name, features_count):
        self.model_name, self.features_count = model_name, features_count
        self.model_path, self.dna_path = f"models/{model_name}.pkl", f"models/{model_name}_dna.json"
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

            # Vorhersage via predict_proba
            probs = model.predict_proba(X_train_scaled)
            if probs.shape[1] > 1:
                preds_train = model.classes_[np.argmax(probs, axis=1)]
            else:
                preds_train = np.zeros(len(X_train), dtype=int)
            tr_idx = np.where(preds_train > 0)[0]

            # Fitness-Berechnung
            if len(tr_idx) >= 1:
                train_fit = np.sum(pips_train[tr_idx]) - (len(tr_idx) * punishment_pips)
                t_probs = model.predict_proba(X_test_scaled)
                if t_probs.shape[1] > 1:
                    t_preds = model.classes_[np.argmax(t_probs, axis=1)]
                else:
                    t_preds = np.zeros(len(X_test), dtype=int)
                t_idx = np.where(t_preds > 0)[0]
                test_fit = np.sum(pips_test[t_idx]) if len(t_idx) > 0 else 0
                wr = (np.sum((t_preds[t_idx] == y_test.values[t_idx])) / len(t_idx) * 100) if len(t_idx) > 0 else 0
                total_fit = train_fit + test_fit
            else:
                probs = model.predict_proba(X_train_scaled)
                conf = np.max(probs[:, 1:]) if probs.shape[1] > 1 else 0
                total_fit = -10000 + (conf * 100)
                train_fit, wr = total_fit, 0

            fitness_scores.append({
                'total_fit': total_fit, 'fit': train_fit, 'weights': weights, 'biases': biases,
                'max_depth': max_depth, 'n_estimators': n_estimators, 'model': model,
                'lw': np.sum(preds_train == 1), 'sw': np.sum(preds_train == 2), 'wr': wr
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
                    'fitness': self.best_fit
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
