import numpy as np


class Evolver:
    """Handles genetic algorithm evolution and mutation."""
    
    @staticmethod
    def evolve_custom(scores):
        """Evolve population from best candidates."""
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
        child_hybrid = Evolver.create_crossover_child(parent1_dna, parent2_dna)
        new_population.append(child_hybrid)

        # Mutated variants
        mutated_children = Evolver.create_mutated_children(parent1_dna, parent2_dna, 10)
        new_population.extend(mutated_children)

        # New random candidates for diversity
        random_children = Evolver.create_random_children(parent1_dna, 2)
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
    def create_mutated_children(parent1_dna, parent2_dna, count):
        """Create mutated variants of parents."""
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
