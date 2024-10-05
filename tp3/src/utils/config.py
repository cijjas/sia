from typing import NamedTuple, Optional
from utils.activation_function import ActivationFunction
from utils.optimizer import Optimizer
import json

class Config(NamedTuple):
    type: Optional[str] = None
    data: Optional[str] = None
    testing_data: Optional[str] = None
    output: Optional[str] = None
    topology: Optional[list] = None
    activation_function: Optional[ActivationFunction] = None
    optimizer: Optional[Optimizer] = None
    epochs: Optional[int] = None
    mini_batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    epsilon: Optional[float] = None
    seed: Optional[int] = None


    def get_json(self, data):
        return Config(
            type=data['problem'].get("type", None),  # Uses None if 'type' is missing
            data=data['problem'].get("data", None),
            testing_data=data['problem'].get("testing_data", None),
            output=data['problem'].get("output", None),
            topology=data['network'].get('topology', None),
            activation_function=ActivationFunction(
                method=data['network']['activation_function'].get('method', None),
                beta=data['network']['activation_function'].get('beta', None)
            ),
            optimizer=Optimizer(
                method=data['network']['optimizer'].get('method', "gradient_descent"),
                mini_batch_size=data['training'].get('mini_batch_size', None),
                eta=data['training'].get('learning_rate', None),
                alpha=data['network']['optimizer'].get('alpha',None),
                beta_1=data['network']['optimizer'].get('beta_1', None),
                beta_2=data['network']['optimizer'].get('beta_2', None),
                epsilon=data['network']['optimizer'].get('epsilon', None) # different epsilon than the one in the training section
            ),
            epochs=data['training'].get('epochs', None),
            mini_batch_size=data['training'].get('mini_batch_size', None),
            learning_rate=data['training'].get('learning_rate', None),
            epsilon=data['training'].get('epsilon', None),
            seed=data['training'].get('seed', None)
        )
        

    def read_config(self, config_path: str) -> 'Config':
        with open(config_path, 'r') as file:
            data = json.load(file)
        
        return self.get_json(data)
