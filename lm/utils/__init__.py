
from .config import ModelConfig, TrainConfig, DataConfig
from .options import OptionBridge
from .registry import opts as registry_values

model_config = ModelConfig(10,20)
train_config = TrainConfig()
data_config = DataConfig("./data")

from .argument_parser import create_argument_parser, get_arg_groups
from .dotmap import DotMap
