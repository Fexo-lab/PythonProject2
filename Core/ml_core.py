import os, joblib, json, numpy as np
from sklearn.ensemble import RandomForestClassifier


class EvolutionCore:
    def __init__(self, model_name, features_count):
        self.model_name, self.features_count = model_name, features_count
        self.model_path, self.dna_path = f"models/{model_name}.pkl", f"models/{model_name}_dna.json"
        self.best_fit = -1e9
        self.population = self._initialize_population()

    def _initialize_population(self):
        # Initialisierung mit 15 Random-Probanden
        pop = [np.random.uniform(0.1, 30.0, self.features_count) for _ in range(15)]
        if os.path.exists(self.dna_path):
            try:
                with open(self.dna_path, 'r') as f:
                    d = json.load(f)
                    pop[0] = np.array(d['weights'])
                    self.best_fit = d.get('fitness', -1e9)
            except:
                pass
        return pop

    def run_cycle(self, train_df, test_df, punishment_pips, cycle, stagnation_counter):
        from Core.config import FEATURE_COLS
        X_train, y_train, pips_train = train_df[FEATURE_COLS], train_df['target_label'], train_df['pip_result'].values
        X_test, y_test, pips_test = test_df[FEATURE_COLS], test_df['target_label'], test_df['pip_result'].values

        fitness_scores = []

        for weights in self.population:
            # Schnelles Modell für die Evolution
            model = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42)
            model.fit(X_train.values * weights, y_train)

            # Vorhersage via predict_proba für mehr Kontrolle
            probs = model.predict_proba(X_train.values * weights)
            if probs.shape[1] > 1:
                # Nutze model.classes_ um korrekte Labels zu bekommen (nicht nur Index)
                preds_train = model.classes_[np.argmax(probs, axis=1)]
            else:
                preds_train = np.zeros(len(X_train), dtype=int)
            tr_idx = np.where(preds_train > 0)[0]

            # Fitness-Berechnung
            if len(tr_idx) >= 1:
                train_fit = np.sum(pips_train[tr_idx]) - (len(tr_idx) * punishment_pips)
                t_probs = model.predict_proba(X_test.values * weights)
                if t_probs.shape[1] > 1:
                    t_preds = model.classes_[np.argmax(t_probs, axis=1)]
                else:
                    t_preds = np.zeros(len(X_test), dtype=int)
                t_idx = np.where(t_preds > 0)[0]
                test_fit = np.sum(pips_test[t_idx]) if len(t_idx) > 0 else 0
                wr = (np.sum((t_preds[t_idx] == y_test.values[t_idx])) / len(t_idx) * 100) if len(t_idx) > 0 else 0
                total_fit = train_fit + test_fit
            else:
                # Wenn keine Trades, nutzen wir die Wahrscheinlichkeit als "Geruch" für die Evolution
                probs = model.predict_proba(X_train.values * weights)
                conf = np.max(probs[:, 1:]) if probs.shape[1] > 1 else 0
                total_fit = -10000 + (conf * 100)
                train_fit, wr = total_fit, 0

            fitness_scores.append({
                'total_fit': total_fit, 'fit': train_fit, 'weights': weights, 'model': model,
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
                json.dump({'weights': curr['weights'].tolist(), 'fitness': self.best_fit}, f)

        # Evolution nach deiner Logik: 2 Eltern, 1 AVG-Kind, 12 Randoms
        self.population = self._evolve_custom(fitness_scores)

        return curr, is_better

    def _evolve_custom(self, scores):
        # 1. Die 2 besten Eltern übernehmen
        parent1 = scores[0]['weights'].copy()
        parent2 = scores[1]['weights'].copy()

        # 2. Das Kind als Durchschnitt (AVG) bilden
        child_avg = (parent1 + parent2) / 2.0

        new_population = [parent1, parent2, child_avg]

        # 3. 12 neue Probanden mit Random Werten auffüllen
        while len(new_population) < 15:
            random_proband = np.random.uniform(0.1, 40.0, self.features_count)
            new_population.append(random_proband)

        return new_population
