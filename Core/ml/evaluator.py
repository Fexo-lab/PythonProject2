import numpy as np
from sklearn.ensemble import RandomForestClassifier


class Evaluator:
    """Handles model evaluation for DNA candidates."""
    
    @staticmethod
    def evaluate_population(population, X_train, y_train, pips_train, X_test, y_test, pips_test, fitness_calculator):
        """Evaluate all DNA candidates in population."""
        fitness_scores = []
        for dna in population:
            score = Evaluator.evaluate_single_dna(dna, X_train, y_train, pips_train, X_test, y_test, pips_test, fitness_calculator)
            fitness_scores.append(score)
        return fitness_scores

    @staticmethod
    def evaluate_single_dna(dna, X_train, y_train, pips_train, X_test, y_test, pips_test, fitness_calculator):
        """Evaluate a single DNA candidate."""
        weights = dna['weights']
        biases = dna['biases']
        max_depth = dna['max_depth']
        n_estimators = dna['n_estimators']
        
        X_train_scaled = (X_train.values * weights) + biases
        X_test_scaled = (X_test.values * weights) + biases

        # Train model with BALANCED REGULARIZATION
        safe_max_depth = min(int(max_depth), 8)  # Increased from 6
        
        model = RandomForestClassifier(
            n_estimators=max(int(n_estimators), 20),
            max_depth=safe_max_depth,                # Max depth: 8
            min_samples_leaf=10,                     # Was 15 - slightly less strict
            min_samples_split=20,                    # Was 30 - slightly less strict
            max_features=0.6,                        # Use 60% of features per split (was 0.5)
            max_samples=0.85,                        # Use 85% of samples (was 0.8)
            class_weight='balanced',                 # Weight minority classes (SELL/SHORT)
            n_jobs=-1,
            random_state=None
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate on both sets
        train_fit, train_stats = Evaluator.evaluate_train_set(model, X_train_scaled, y_train, pips_train, fitness_calculator)
        test_fit, test_stats, win_rate, profit_factor = Evaluator.evaluate_test_set(model, X_test_scaled, y_test, pips_test, fitness_calculator)

        # Total fitness with OVERFITTING PENALTY
        total_fit = (0.7 * test_fit) + (0.3 * train_fit)
        
        # Quality metrics
        quality_score, overfitting_ratio, is_overfitting, is_significant, is_realistic, weight_importance = fitness_calculator.calculate_quality_metrics(
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

    @staticmethod
    def evaluate_train_set(model, X_train_scaled, y_train, pips_train, fitness_calculator):
        """Evaluate model on training set."""
        probs = model.predict_proba(X_train_scaled)
        if probs.shape[1] > 1:
            preds_train = model.classes_[np.argmax(probs, axis=1)]
        else:
            preds_train = np.zeros(len(X_train_scaled), dtype=int)
        
        long_idx = np.where(preds_train == 1)[0]
        short_idx = np.where(preds_train == 2)[0]
        
        long_pips = pips_train[long_idx] if len(long_idx) > 0 else np.array([])
        long_winners = np.sum(long_pips > 0) if len(long_pips) > 0 else 0
        long_total_pips = np.sum(long_pips) if len(long_pips) > 0 else 0
        
        short_pips = pips_train[short_idx] if len(short_idx) > 0 else np.array([])
        short_winners = np.sum(short_pips > 0) if len(short_pips) > 0 else 0
        short_total_pips = np.sum(short_pips) if len(short_pips) > 0 else 0
        
        total_trades = len(long_idx) + len(short_idx)
        total_pips_train = long_total_pips + short_total_pips
        
        train_fit = fitness_calculator.calculate_trade_fitness(
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

    @staticmethod
    def evaluate_test_set(model, X_test_scaled, y_test, pips_test, fitness_calculator):
        """Evaluate model on test set."""
        t_probs = model.predict_proba(X_test_scaled)
        if t_probs.shape[1] > 1:
            t_preds = model.classes_[np.argmax(t_probs, axis=1)]
        else:
            t_preds = np.zeros(len(X_test_scaled), dtype=int)
        
        t_long_idx = np.where(t_preds == 1)[0]
        t_short_idx = np.where(t_preds == 2)[0]
        
        t_long_pips = pips_test[t_long_idx] if len(t_long_idx) > 0 else np.array([])
        t_long_winners = np.sum(t_long_pips > 0) if len(t_long_pips) > 0 else 0
        t_long_total_pips = np.sum(t_long_pips) if len(t_long_pips) > 0 else 0
        
        t_short_pips = pips_test[t_short_idx] if len(t_short_idx) > 0 else np.array([])
        t_short_winners = np.sum(t_short_pips > 0) if len(t_short_pips) > 0 else 0
        t_short_total_pips = np.sum(t_short_pips) if len(t_short_pips) > 0 else 0
        
        t_total_trades = len(t_long_idx) + len(t_short_idx)
        t_total_pips = t_long_total_pips + t_short_total_pips
        
        win_rate, profit_factor = fitness_calculator.calculate_performance_metrics(t_long_pips, t_short_pips)
        
        test_fit = fitness_calculator.calculate_test_fitness(
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
