import numpy as np


class FitnessCalculator:
    """Handles fitness calculations and quality metrics."""
    
    @staticmethod
    def calculate_performance_metrics(long_pips, short_pips):
        """Calculate win rate and profit factor."""
        all_pips = np.concatenate([long_pips, short_pips]) if len(long_pips) > 0 or len(short_pips) > 0 else np.array([])
        
        if len(all_pips) == 0:
            return 0.0, 0.0
        
        winners = all_pips[all_pips > 0]
        losers = all_pips[all_pips < 0]
        
        # Win Rate
        if len(winners) > 0 or len(losers) > 0:
            win_rate = (len(winners) / len(all_pips)) * 100
        else:
            win_rate = 0.0
        
        # Profit Factor
        if len(winners) > 0 and len(losers) > 0:
            avg_win = np.mean(winners)
            avg_loss = abs(np.mean(losers))
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0
        else:
            profit_factor = 1.0 if len(winners) > 0 else 0.0
        
        return win_rate, profit_factor

    @staticmethod
    def calculate_trade_fitness(total_trades, long_winners, long_total_trades, short_winners, short_total_trades, long_pips, short_pips, total_pips):
        """Calculate fitness for training set."""
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

    @staticmethod
    def calculate_test_fitness(t_total_trades, t_total_pips, win_rate, profit_factor):
        """Calculate fitness for test set."""
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

    @staticmethod
    def calculate_quality_metrics(train_fit, test_fit, total_trades, win_rate, profit_factor, weights):
        """Calculate quality metrics and filters."""
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
