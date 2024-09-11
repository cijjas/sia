from typing import Dict, Any

class Selector:
    def __init__(self, data: Dict[str, Any]):
        self.method = data.get('method', 'elite')
        self.weight = data.get('weight', 1.0)
        if self.method == 'boltzmann':
            self.k = data.get('k', 1.0)
            self.t_0 = data.get('t_0', 1.0)
            self.t_C = data.get('t_C', 10.0)
        elif self.method == 'deterministic_tournament':
            self.tournament_size = data.get('tournament_size', 2)
        elif self.method == 'probabilistic_tournament':
            self.threshold = data.get('threshold', 0.5)

    def __str__(self) -> str:
        return f"Method: {self.method}, Weight: {self.weight}, Params: {self.params}"

           

class Mutator:

    def __init__(self, data: Dict[str, Any]):
        self.method = data.get('method', 'multigen_uniform')
        self.amount = data.get('amount', 1)
        self.distribution = data.get('distribution', 'uniform')
        self.distribution_params = data.get('distribution_params', {})
        self.rate_method = data.get('rate_method', 'constant')
        self.initial_rate = data.get('initial_rate', 0.5)
        self.final_rate = data.get('final_rate', 0.1)
        self.decay_rate = data.get('decay_rate', 0.99)


    def __str__(self) -> str:
        return f"Method: {self.method}, Amount: {self.amount}, Rate: {self.rate}, Distribution: {self.distribution}, Distribution Params: {self.distribution_params}"

class Terminator:
    def __init__(self, data: Dict[str, Any]):
        self.max_generations = data.get('max_generations', 1000)
        self.structure_portion = data.get('structure', {}).get('portion', None)
        self.structure_generations = data.get('structure', {}).get('generations', None)
        self.content = data.get('content', None)
        self.desired_fitness = data.get('desired_fitness', None)

    def __str__(self) -> str:
        return f"Max Generations: {self.max_generations}, Structure Portion: {self.structure_portion}, Structure Generations: {self.structure_generations}, Content: {self.content}, Desired Fitness: {self.desired_fitness}"

class Hyperparameters:
    def __init__(self, data: Dict[str, Any]):
        self.population_size = data.get('population_size', 100)
        self.crossover_method = data.get('operators', {}).get('crossover', {}).get('method', 'uniform')
        self.crossover_rate = data.get('operators', {}).get('crossover', {}).get('rate', 0.5)
        
        self.mutation = Mutator(data.get('operators', {}).get('mutation', {}))
        self.selection_rate = data.get('selection', {}).get('selection_rate', 0.5)
        parents = data.get('selection', {}).get('parents', [])
        replacements = data.get('selection', {}).get('replacement', [])
        self.parents_selection_methods = [Selector(parent) for parent in parents]
        self.replacements_selection_methods = [Selector(replacement) for replacement in replacements]
        self.termination_criteria = Terminator(data.get('termination_criteria', {}))
        


