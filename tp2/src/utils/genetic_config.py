from typing import Dict, Any

class SelectionMethod:
    def __init__(self, data: Dict[str, Any]):
        self.method = data.get('method', 'elite')
        self.weight = data.get('weight', 1.0)
        if self.method == 'boltzmann':
            self.k = 1.0
            self.t_0 = 1.0
            self.t_C = 0.9
        elif self.method == 'deterministic_tournament':
            self.tournament_size = 2
        elif self.method == 'probabilistic_tournament':
            self.threshold = 0.5

    def __str__(self) -> str:
        return f"Method: {self.method}, Weight: {self.weight}, Params: {self.params}"

           



class TerminatorConfig:
    def __init__(self, data: Dict[str, Any]):
        self.max_generations = data.get('max_generations', 1000)
        self.structure_portion = data.get('structure', {}).get('portion', None)
        self.structure_generations = data.get('structure', {}).get('generations', None)
        self.content = data.get('content', None)
        self.desired_fitness = data.get('desired_fitness', None)

    def __str__(self) -> str:
        return f"Max Generations: {self.max_generations}, Structure Portion: {self.structure_portion}, Structure Generations: {self.structure_generations}, Content: {self.content}, Desired Fitness: {self.desired_fitness}"

class GAConfig:
    def __init__(self, data: Dict[str, Any]):
        self.population_size = data.get('population_size', 100)
        self.crossover_method = data.get('operators', {}).get('crossover', {}).get('method', 'uniform')
        self.mutation_method = data.get('operators', {}).get('mutation', {}).get('method', 'multigen_uniform')
        self.mutation_rate = data.get('operators', {}).get('mutation', {}).get('rate', 0.1)
        self.selection_rate = data.get('selection', {}).get('selection_rate', 0.5)
        parents = data.get('selection', {}).get('parents', [])
        replacements = data.get('selection', {}).get('replacement', [])
        self.parents_selection_methods = [SelectionMethod(parent) for parent in parents]
        self.replacements_selection_methods = [SelectionMethod(replacement) for replacement in replacements]
        self.termination_criteria = TerminatorConfig(data.get('termination_criteria', {}))
        


