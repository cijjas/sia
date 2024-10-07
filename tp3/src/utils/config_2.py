import json
from typing import NamedTuple, Optional, List
from utils.activation_function import ActivationFunction
from utils.optimizer import Optimizer

class LayerConfig(NamedTuple):
    num_neurons: int
    activation_function: Optional[ActivationFunction] = None

    def __str__(self) -> str:
        return f"num_neurons={self.num_neurons}, activation_function={self.activation_function})"
    
class Config(NamedTuple):
    # Problem configuration
    type: Optional[str] = None
    data: Optional[str] = None
    testing_data: Optional[str] = None
    output: Optional[str] = None

    # Network configuration
    topology: Optional[List[LayerConfig]] = None
    optimizer: Optional[Optimizer] = None

    # Training configuration
    seed: Optional[int] = None
    epochs: Optional[int] = None
    mini_batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    epsilon: Optional[float] = None

    @classmethod
    def from_json_string(cls, json_string: str) -> 'Config':
        problem_data = json_string.get('problem', {})
        network_data = json_string.get('network', {})
        training_data = json_string.get('training', {})


        # Parse problem section
        problem_type = problem_data.get('type', None)
        problem_data_path = problem_data.get('data', None)
        testing_data_path = problem_data.get('testing_data', None)
        output_path = problem_data.get('output', None)

        # Parse network section
        topology_data = network_data.get('topology', [])
        topology = []
        
        for layer in topology_data:
            num_neurons = layer.get('num_neurons')
            activation_function_data = layer.get('activation_function', None)
            if activation_function_data:
                activation_function = ActivationFunction(
                    method=activation_function_data.get('method'),
                    beta=activation_function_data.get('beta', None)
                )
            else:
                activation_function = None
            topology.append(
                LayerConfig(
                    num_neurons=num_neurons, 
                    activation_function=activation_function
                )
            )

        optimizer_data = network_data.get('optimizer', {})
        optimizer = Optimizer(
            method=optimizer_data.get('method', "gradient_descent"),
            mini_batch_size=training_data.get('mini_batch_size', None),
            eta=optimizer_data.get('learning_rate', None), 
            alpha=optimizer_data.get('alpha', None),
            beta_1=optimizer_data.get('beta_1', None),
            beta_2=optimizer_data.get('beta_2', None),
            epsilon=optimizer_data.get('epsilon', None)
        )

        # Parse training section
        seed = training_data.get('seed', None)
        epochs = training_data.get('epochs', None)
        mini_batch_size = training_data.get('mini_batch_size', None)
        learning_rate = optimizer_data.get('learning_rate', None)
        epsilon = training_data.get('epsilon', None)



        return Config(
            type=problem_type,
            data=problem_data_path,
            testing_data=testing_data_path,
            output=output_path,
            topology=topology,
            optimizer=optimizer,
            seed=seed,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            learning_rate=learning_rate,
            epsilon=epsilon
        )

    @classmethod
    def from_json_file(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as file:
            data = json.load(file)
        return cls.from_json(data)
    
    def __str__(self) -> str:
        return f"Config(type={self.type}, data={self.data}, testing_data={self.testing_data}, output={self.output}, topology={self.topology}, optimizer={self.optimizer}, seed={self.seed}, epochs={self.epochs}, mini_batch_size={self.mini_batch_size}, learning_rate={self.learning_rate}, epsilon={self.epsilon})"
