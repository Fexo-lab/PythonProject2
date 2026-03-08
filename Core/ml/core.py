import os, joblib, json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from Core.config import MODELS_DIR
from Core.ml.evaluator import Evaluator
from Core.ml.fitness import FitnessCalculator
from Core.ml.evolution import Evolver


class EvolutionCore:
    def __init__(self, model_name, features_count):
        self.model_name, self.features_count = model_name, features_count
        self.model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        self.dna_path = os.path.join(MODELS_DIR, f"{model_name}_dna.json")
        self.best_fit = -1e9
        self.population = self._initialize_population()

    def _initialize_population(self):
        # DNA consists of: 9 Weights + 9 Biases + Hyperparameter (max_depth, n_estimators)
        pop = []
        for _ in range(15):
            dna = {
                'weights': np.random.uniform(0.1, 3.0, self.features_count),      # Relative Multipliers!
                'biases': np.random.uniform(-1.0, 1.0, self.features_count),     # Additive Offsets
                'max_depth': np.random.randint(4, 16),
                'n_estimators': np.random.randint(10, 100)
            }
            pop.append(dna)
        
        # Load best previous DNA
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

        fitness_scores = Evaluator.evaluate_population(self.population, X_train, y_train, pips_train, X_test, y_test, pips_test, FitnessCalculator)
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

        self.population = Evolver.evolve_custom(fitness_scores)
        return curr, is_better

    def _evaluate_population(self, population, X_train, y_train, pips_train, X_test, y_test, pips_test):
        fitness_scores = []
        for dna in population:
            score = self._evaluate_single_dna(dna, X_train, y_train, pips_train, X_test, y_test, pips_test)
            fitness_scores.append(score)
        return fitness_scores

    def _evaluate_single_dna(self, dna, X_train, y_train, pips_train, X_test, y_test, pips_test):
            # Apply weights and biases to the data
            weights = dna['weights']
            biases = dna['biases']
            max_depth = dna['max_depth']
            n_estimators = dna['n_estimators']
            
            X_train_scaled = (X_train.values * weights) + biases
            X_test_scaled = (X_test.values * weights) + biases

            # Model with evolved hyperparameters + REGULARIZATION
            # Limit max_depth to prevent overfitting on small datasets
            safe_max_depth = min(int(max_depth), 10)
            
            model = RandomForestClassifier(
                n_estimators=max(int(n_estimators), 20),
                max_depth=safe_max_depth,
                min_samples_leaf=5,           # Prevent single-sample leaves
                min_samples_split=10,         # Prevent excessive splitting
                max_features=0.7,             # Use only 70% of features per split
                subsample=0.9,                # Use 90% of samples (row subsampling)
                n_jobs=-1,
                random_state=None
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate on both train and test sets
            train_fit, train_stats = self._evaluate_train_set(model, X_train_scaled, y_train, pips_train)
            test_fit, test_stats, win_rate, profit_factor = self._evaluate_test_set(model, X_test_scaled, y_test, pips_test)

            # Calculate total fitness with OVERFITTING PENALTY
            total_fit = (0.7 * test_fit) + (0.3 * train_fit)
            
            # Calculate quality metrics
            quality_score, overfitting_ratio, is_overfitting, is_significant, is_realistic, weight_importance = self._calculate_quality_metrics(
                train_fit, test_fit, test_stats['total_trades'], win_rate, profit_factor, weights
            )

            # APPLY STRONG OVERFITTING PENALTY: Reduce fitness if overfitting detected
            if is_overfitting:
                overfit_penalty = 1.0 - (min(overfitting_ratio, 1.0) * 0.5)  # Max 50% penalty
                total_fit *= overfit_penalty

            return {
                'total_fit': total_fit, 
                'test_fit': test_fit,
                'train_fit': train_fit,
                'fit': train_fit,
                'weights': weights, 
                'biases': biases,
                'max_depth': max_depth, 
                'n_estimators': n_estimators, 
                'model': model,
                'lw': train_stats['long_total_trades'],
                'sw': train_stats['short_total_trades'],
                'wr': win_rate,
                'long_winners': train_stats['long_winners'],
                'long_total_trades': train_stats['long_total_trades'],
                'long_total_pips': train_stats['long_total_pips'],
                'short_winners': train_stats['short_winners'],
                'short_total_trades': train_stats['short_total_trades'],
                'short_total_pips': train_stats['short_total_pips'],
                'test_long_winners': test_stats['long_winners'],
                'test_long_total_trades': test_stats['long_total_trades'],
                'test_long_total_pips': test_stats['long_total_pips'],
                'test_short_winners': test_stats['short_winners'],
                'test_short_total_trades': test_stats['short_total_trades'],
                'test_short_total_pips': test_stats['short_total_pips'],
                'profit_factor': profit_factor,
                'overfitting_ratio': overfitting_ratio,
                'is_overfitting': is_overfitting,
                'is_significant': is_significant,
                'is_realistic': is_realistic,
                'quality_score': quality_score,
                'feature_importance': weight_importance.tolist()
            }

    def _evaluate_train_set(self, model, X_train_scaled, y_train, pips_train):
            probs = model.predict_proba(X_train_scaled)
            if probs.shape[1] > 1:
                preds_train = model.classes_[np.argmax(probs, axis=1)]
            else:
                preds_train = np.zeros(len(X_train_scaled), dtype=int)
            
            # Separate: Long (1) and Short (2) Trades
            long_idx = np.where(preds_train == 1)[0]
            short_idx = np.where(preds_train == 2)[0]
            
            # Long Statistics
            long_pips = pips_train[long_idx] if len(long_idx) > 0 else np.array([])
            long_winners = np.sum(long_pips > 0) if len(long_pips) > 0 else 0
            long_total_pips = np.sum(long_pips) if len(long_pips) > 0 else 0
            
            # Short Statistics
            short_pips = pips_train[short_idx] if len(short_idx) > 0 else np.array([])
            short_winners = np.sum(short_pips > 0) if len(short_pips) > 0 else 0
            short_total_pips = np.sum(short_pips) if len(short_pips) > 0 else 0
            
            total_trades = len(long_idx) + len(short_idx)
            total_pips_train = long_total_pips + short_total_pips
            
            train_fit = self._calculate_trade_fitness(
                total_trades, long_winners, len(long_idx), short_winners, len(short_idx), 
                long_pips, short_pips, total_pips_train
            )
            
            train_stats = {
                'long_winners': long_winners,
                'long_total_trades': len(long_idx),
                'long_total_pips': long_total_pips,
                'short_winners': short_winners,
                'short_total_trades': len(short_idx),
                'short_total_pips': short_total_pips,
                'total_trades': total_trades
            }
            
            return train_fit, train_stats

    def _evaluate_test_set(self, model, X_test_scaled, y_test, pips_test):
            t_probs = model.predict_proba(X_test_scaled)
            if t_probs.shape[1] > 1:
                t_preds = model.classes_[np.argmax(t_probs, axis=1)]
            else:
                t_preds = np.zeros(len(X_test_scaled), dtype=int)
            
            # Separate: Long and Short in test set
            t_long_idx = np.where(t_preds == 1)[0]
            t_short_idx = np.where(t_preds == 2)[0]
            
            # Long Statistics
            t_long_pips = pips_test[t_long_idx] if len(t_long_idx) > 0 else np.array([])
            t_long_winners = np.sum(t_long_pips > 0) if len(t_long_pips) > 0 else 0
            t_long_total_pips = np.sum(t_long_pips) if len(t_long_pips) > 0 else 0
            
            # Short Statistics
            t_short_pips = pips_test[t_short_idx] if len(t_short_idx) > 0 else np.array([])
            t_short_winners = np.sum(t_short_pips > 0) if len(t_short_pips) > 0 else 0
            t_short_total_pips = np.sum(t_short_pips) if len(t_short_pips) > 0 else 0
            
            t_total_trades = len(t_long_idx) + len(t_short_idx)
            t_total_pips = t_long_total_pips + t_short_total_pips
            
            win_rate, profit_factor = self._calculate_performance_metrics(t_long_pips, t_short_pips)
            
            test_fit = self._calculate_test_fitness(
                t_total_trades, t_total_pips, win_rate, profit_factor
            )
            
            test_stats = {
                'long_winners': t_long_winners,
                'long_total_trades': len(t_long_idx),
                'long_total_pips': t_long_total_pips,
                'short_winners': t_short_winners,
                'short_total_trades': len(t_short_idx),
                'short_total_pips': t_short_total_pips,
                'total_trades': t_total_trades
            }
            
            return test_fit, test_stats, win_rate, profit_factor

    def _calculate_performance_metrics(self, long_pips, short_pips):
            all_pips = np.concatenate([long_pips, short_pips]) if len(long_pips) > 0 or len(short_pips) > 0 else np.array([])
            
            if len(all_pips) == 0:
                return 0.0, 0.0
            
            winners = all_pips[all_pips > 0]
            losers = all_pips[all_pips < 0]
            
            # Win Rate calculation
            if len(winners) > 0 or len(losers) > 0:
                win_rate = (len(winners) / len(all_pips)) * 100
            else:
                win_rate = 0.0
            
            # Profit Factor calculation
            if len(winners) > 0 and len(losers) > 0:
                avg_win = np.mean(winners)
                avg_loss = abs(np.mean(losers))
                profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0
            else:
                profit_factor = 1.0 if len(winners) > 0 else 0.0
            
            return win_rate, profit_factor

    def _calculate_trade_fitness(self, total_trades, long_winners, long_total_trades, short_winners, short_total_trades, long_pips, short_pips, total_pips):
            if total_trades > 0:
                train_wr = ((long_winners + short_winners) / total_trades) * 100
                all_pips = np.concatenate([long_pips, short_pips]) if len(long_pips) > 0 or len(short_pips) > 0 else np.array([])
                
                if len(all_pips) > 0:
                    train_winners = all_pips[all_pips > 0]
                    train_losers = all_pips[all_pips < 0]
                    
                    if len(train_winners) > 0 and len(train_losers) > 0:
                        train_avg_win = np.mean(train_winners)
                        train_avg_loss = abs(np.mean(train_losers))
                        train_profit_factor = train_avg_win / train_avg_loss if train_avg_loss > 0 else 1.0
                    else:
                        train_profit_factor = 1.0 if len(train_winners) > 0 else 0.0
                else:
                    train_profit_factor = 0.0
                
                fit = total_pips * 0.3
                fit += max(0, (train_wr - 50) * 100)
                fit += max(0, (train_profit_factor - 1.0) * 500)
                
                if total_trades >= 100:
                    fit += 1000
                elif total_trades >= 50:
                    fit += 500
                
                return fit
            else:
                return -10000

    def _calculate_test_fitness(self, t_total_trades, t_total_pips, win_rate, profit_factor):
            if t_total_trades > 0:
                test_fit = t_total_pips * 0.3
                win_rate_bonus = max(0, (win_rate - 50) * 100)
                test_fit += win_rate_bonus
                
                profit_factor_bonus = max(0, (profit_factor - 1.0) * 500)
                test_fit += profit_factor_bonus
                
                if t_total_trades >= 100:
                    test_fit += 1000
                elif t_total_trades >= 50:
                    test_fit += 500
                
                return test_fit
            else:
                return -10000

    def _calculate_quality_metrics(self, train_fit, test_fit, total_trades, win_rate, profit_factor, weights):
            from Core.config import OVERFIT_THRESHOLD, MIN_TRADES_FOR_SIGNIFICANCE, REALISTIC_WR_MIN, REALISTIC_WR_MAX, REALISTIC_PF_MIN
            
            # Overfitting Detection
            overfitting_ratio = abs(train_fit - test_fit) / (abs(test_fit) + 1)
            is_overfitting = overfitting_ratio > OVERFIT_THRESHOLD
            
            # Statistical Significance
            is_significant = total_trades >= MIN_TRADES_FOR_SIGNIFICANCE
            
            # Realistic Check
            is_realistic = (REALISTIC_WR_MIN <= win_rate <= REALISTIC_WR_MAX) and (profit_factor >= REALISTIC_PF_MIN)
            
            # Feature Importance
            weight_importance = np.abs(weights / np.sum(np.abs(weights)) * 100)
            
            # Quality Score (0-1)
            quality_score = (1.0 if not is_overfitting else 0.5) * \
                           (1.0 if is_significant else 0.7) * \
                           (1.0 if is_realistic else 0.5)
            
            return quality_score, overfitting_ratio, is_overfitting, is_significant, is_realistic, weight_importance

    def _evolve_custom(self, scores):
        parent1_dna = scores[0]
        parent2_dna = scores[1]
        new_population = []

        # Elitism: Keep both best parents
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

        # Intelligent Crossover
        child_hybrid = self._create_crossover_child(parent1_dna, parent2_dna)
        new_population.append(child_hybrid)

        # Mutated variants
        mutated_children = self._create_mutated_children(parent1_dna, parent2_dna, 10)
        new_population.extend(mutated_children)

        # New random candidates for diversity
        random_children = self._create_random_children(parent1_dna, 2)
        new_population.extend(random_children)

        return new_population

    def _create_crossover_child(self, parent1_dna, parent2_dna):
        crossover_mask = np.random.rand(len(parent1_dna['weights'])) > 0.5
        return {
            'weights': np.where(crossover_mask, parent1_dna['weights'], parent2_dna['weights']).copy(),
            'biases': np.where(crossover_mask, parent1_dna['biases'], parent2_dna['biases']).copy(),
            'max_depth': parent1_dna['max_depth'] if np.random.rand() > 0.5 else parent2_dna['max_depth'],
            'n_estimators': parent1_dna['n_estimators'] if np.random.rand() > 0.5 else parent2_dna['n_estimators']
        }

    def _create_mutated_children(self, parent1_dna, parent2_dna, count):
        mutated_children = []
        for i in range(count):
            parent_dna = parent1_dna if i % 2 == 0 else parent2_dna
            mutation_strength = 0.2 + (i * 0.03)
            
            mutant = {
                'weights': parent_dna['weights'].copy(),
                'biases': parent_dna['biases'].copy(),
                'max_depth': parent_dna['max_depth'],
                'n_estimators': parent_dna['n_estimators']
            }
            
            # Mutate weights
            weight_mask = np.random.rand(len(mutant['weights'])) < 0.5
            mutant['weights'][weight_mask] *= (1 + np.random.normal(0, mutation_strength, np.sum(weight_mask)))
            mutant['weights'] = np.clip(mutant['weights'], 0.1, 3.0)
            
            # Mutate biases
            bias_mask = np.random.rand(len(mutant['biases'])) < 0.5
            mutant['biases'][bias_mask] += np.random.normal(0, mutation_strength * 0.5, np.sum(bias_mask))
            mutant['biases'] = np.clip(mutant['biases'], -1.0, 1.0)
            
            # Mutate hyperparameters
            mutant['max_depth'] = int(np.clip(mutant['max_depth'] + np.random.randint(-2, 3), 4, 16))
            mutant['n_estimators'] = int(np.clip(mutant['n_estimators'] + np.random.randint(-10, 11), 10, 100))
            
            mutated_children.append(mutant)
        
        return mutated_children

    def _create_random_children(self, parent_dna, count):
        random_children = []
        for _ in range(count):
            random_dna = {
                'weights': np.random.uniform(0.1, 3.0, len(parent_dna['weights'])),
                'biases': np.random.uniform(-1.0, 1.0, len(parent_dna['weights'])),
                'max_depth': np.random.randint(4, 16),
                'n_estimators': np.random.randint(10, 100)
            }
            random_children.append(random_dna)
        
        return random_children
