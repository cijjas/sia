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
    n_splits: Optional[int] = None
    seed: Optional[int] = None


    def read_config(self, config_path: str) -> 'Config':
        with open(config_path, 'r') as file:
            data = json.load(file)
        
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
                method=data['network']['optimizer'].get('method', None),
                eta=data['training'].get('learning_rate', None), # nasty sori
                beta1=data['network']['optimizer'].get('beta1', None),
                beta2=data['network']['optimizer'].get('beta2', None),
                epsilon=data['network']['optimizer'].get('epsilon', None),
                m=data['network']['optimizer'].get('m', None),
                v=data['network']['optimizer'].get('v', None),
                t=data['network']['optimizer'].get('t', None)
            ),
            epochs=data['training'].get('epochs', None),
            mini_batch_size=data['training'].get('mini_batch_size', None),
            n_splits=data['training'].get('n_splits', None),
            learning_rate=data['training'].get('learning_rate', None),
            epsilon=data['training'].get('epsilon', None),
            seed=data['training'].get('seed', None)
        )
