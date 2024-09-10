import random

def normalizer(individual, total_points):
    current_sum = individual.genes.get_total_points()

    if current_sum == total_points:
        return

    scaling_factor = total_points / current_sum
    individual.genes.attributes = [int(attr * scaling_factor) for attr in individual.genes.attributes[:-1]] + [individual.genes.attributes[-1]]

    final_sum = individual.genes.get_total_points()
    residual = total_points - final_sum

    while abs(residual) > 0 :
        random_idx = random.randint(0, len(individual.genes.attributes) - 2)
        if residual > 0:
            adjustment = min(residual, 1)
            individual.genes[random_idx] += adjustment
        else:
            adjustment = max(residual, -1)
            individual.genes[random_idx] -= adjustment

        residual = total_points - individual.genes.get_total_points()
        if individual.genes[random_idx] < 0:
            residual += individual.genes[random_idx]  
            individual.genes[random_idx] = 0