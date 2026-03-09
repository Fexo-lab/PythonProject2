import numpy as np


class Evolver:
    """Handles genetic algorithm evolution and mutation."""
    
    # Static variable to track stagnation
    stagnation_counter = 0
    last_best_fitness = -1e9
    
    @staticmethod
    def evolve_custom(scores, stagnation_count=0):
        """Evolve population from best candidates with adaptive mutation."""
        parent1_dna = scores[0]
        parent2_dna = scores[1] if len(scores) > 1 else scores[0]
        new_population = []

        # Elitism: Keep top 3 best parents (increased from 2)
        for i in range(min(3, len(scores))):
            new_population.append({
                'weights': scores[i]['weights'].copy(),
                'biases': scores[i]['biases'].copy(),
                'max_depth': scores[i]['max_depth'],
                'n_estimators': scores[i]['n_estimators']
            })

        # Intelligent Crossover (3 children instead of 1)
        for _ in range(3):
            child_hybrid = Evolver.create_crossover_child(parent1_dna, parent2_dna)
            new_population.append(child_hybrid)

        # Adaptive Mutation: Increase mutation when stagnating
        # stagnation_count > 1000 means 1000+ cycles without improvement
        mutation_intensity = 1.0  # Default
        if stagnation_count > 1000:
            mutation_intensity = 2.0  # 2x stronger mutation if stagnating
        elif stagnation_count > 500:
            mutation_intensity = 1.5  # 1.5x stronger if minor stagnation
        
        # Mutated variants (20 instead of 10 for more exploration)
        mutated_children = Evolver.create_mutated_children(
            parent1_dna, parent2_dna, 20, mutation_intensity
        )
        new_population.extend(mutated_children)

        # New random candidates for DIVERSITY (10 instead of 2 - major improvement!)
        random_children = Evolver.create_random_children(parent1_dna, 10)
        new_population.extend(random_children)

        return new_population

    @staticmethod
    def create_crossover_child(parent1_dna, parent2_dna):
        """Create offspring through crossover breeding."""
        crossover_mask = np.random.rand(len(parent1_dna['weights'])) > 0.5
        return {
            'weights': np.where(crossover_mask, parent1_dna['weights'], parent2_dna['weights']).copy(),
            'biases': np.where(crossover_mask, parent1_dna['biases'], parent2_dna['biases']).copy(),
            'max_depth': parent1_dna['max_depth'] if np.random.rand() > 0.5 else parent2_dna['max_depth'],
            'n_estimators': parent1_dna['n_estimators'] if np.random.rand() > 0.5 else parent2_dna['n_estimators']
        }

    @staticmethod
    def create_mutated_children(parent1_dna, parent2_dna, count, mutation_intensity=1.0):
        """Create mutated variants of parents with adaptive intensity."""
        mutated_children = []
        for i in range(count):
            parent_dna = parent1_dna if i % 2 == 0 else parent2_dna
            
            # Stronger base mutation: 0.5-1.5 instead of 0.2-0.5
            base_mutation = 0.5 + (i * 0.05)
            mutation_strength = base_mutation * mutation_intensity
            
            mutant = {
                'weights': parent_dna['weights'].copy(),
                'biases': parent_dna['biases'].copy(),
                'max_depth': parent_dna['max_depth'],
                'n_estimators': parent_dna['n_estimators']
            }
            
            # Mutate weights: 70% instead of 50% (more aggressive)
            weight_mask = np.random.rand(len(mutant['weights'])) < 0.7
            mutant['weights'][weight_mask] *= (1 + np.random.normal(0, mutation_strength, np.sum(weight_mask)))
            mutant['weights'] = np.clip(mutant['weights'], 0.1, 3.0)
            
            # Mutate biases: 70% instead of 50%
            bias_mask = np.random.rand(len(mutant['biases'])) < 0.7
            mutant['biases'][bias_mask] += np.random.normal(0, mutation_strength * 0.5, np.sum(bias_mask))
            mutant['biases'] = np.clip(mutant['biases'], -1.0, 1.0)
            
            # Mutate hyperparameters with more aggressive changes
            depth_change = np.random.randint(-3, 4)  # -3 to +3 instead of -2 to +2
            mutant['max_depth'] = int(np.clip(mutant['max_depth'] + depth_change, 4, 16))
            
            est_change = np.random.randint(-20, 21)  # -20 to +20 instead of -10 to +10
            mutant['n_estimators'] = int(np.clip(mutant['n_estimators'] + est_change, 10, 100))
            
            mutated_children.append(mutant)
        
        return mutated_children

    @staticmethod
    def create_random_children(parent_dna, count):
        """Create completely random candidates for diversity."""
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
