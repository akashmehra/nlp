
from abc import ABC, abstractmethod
from typing import List, Optional 
import torch
from .options import OptionBridge
from .registry import Registry

class BaseConfig(ABC):

    def __init__(self):
        self._params_list = None

    @Registry("base_config")
    def _named_params(self) -> List[str] : 
        self._params_list = {key:value for key, value in vars(self).items() if not key.startswith('_')} 
        return self._params_list

    def params(self):
        return {key: getattr(self,key)() for key in self._params_list}
        
 
class ModelConfig(BaseConfig):
    def __init__(self, input_size: Optional[int] = 20000, hidden_size : Optional[int] = 256, 
                 num_layers: Optional[int] = 2, 
                 dropout_p: Optional[float] = 0.4):
        super(ModelConfig, self).__init__()
        self.input_size = OptionBridge[int]("input_size", input_size, 
                                   desc="Size of the input "\
                                   +"layer, this usually is the size of the "\
                                   + "vocab in case of text modality.", 
                                   required=False)
        self.hidden_size = OptionBridge[int]("hidden_size", hidden_size, 
                                    desc="Size of the"\
                                    +" hidden layer", required=False)
        self.num_layers = OptionBridge[int]("num_layers", num_layers, desc="Number of "\
                              + " layers of encoder", required=False)
        self.dropout_p = OptionBridge[float]("dropout_p", dropout_p, 
                                    desc="dropout probability", required=False)
        super(ModelConfig, self)._named_params()
    
        
class TrainConfig(BaseConfig):
    def __init__(self, lr=1.0, gamma=0.9, momentum=0.9, clip_value=0.25, 
                 num_epochs=5, log_interval=10, device='cpu',
                 checkpoint_dir='./models'):
        super(TrainConfig, self).__init__()
        self.lr = OptionBridge[float]("lr", lr, desc="learning rate", required=False)
        self.gamma = OptionBridge[float]("gamma", gamma, desc="gamma for lr scheduler", 
                                required=False)
        self.momentum = OptionBridge[float]("momentum", momentum, desc="Momentum value",
                                   required=False)
        self.clip_value = OptionBridge[float]("clip_value", clip_value, 
                                     desc="clip value for grad", required=False)
        self.num_epochs = OptionBridge[int]("num_epochs", num_epochs, 
                                   desc="Number of epochs", required=False)
        self.log_interval = OptionBridge[int]("log_interval", log_interval, 
                                     desc="Interval between logs", 
                                     required=False)
        self.device = OptionBridge[str]("device", device, desc="device: cpu or cuda",
                                     required=False, choices=['cpu', 'cuda'])
        self.checkpoint_dir = OptionBridge[str]("checkpoint_dir", checkpoint_dir,
                                       desc="Directory for checkpointing")
        super(TrainConfig, self)._named_params()


class DataConfig(BaseConfig):
    def __init__(self, data_dir, train_batch_size=24, valid_batch_size=24,
                 num_workers=0, shuffle_data=True, drop_last=True):
        super(DataConfig, self).__init__()
        self.data_dir = OptionBridge[str]("data_dir", data_dir, 
                                 desc="directory where data is located", 
                                 required=True)
        self.train_batch_size = OptionBridge[int]("train_batch_size", train_batch_size,
                                         desc="Batch Size for training",
                                         default=24)
        self.valid_batch_size = OptionBridge[int]("valid_batch_size", valid_batch_size,
                                         desc="Batch Size for validation data",
                                         default=24)
        self.shuffle_data = OptionBridge[bool]("shuffle_data", shuffle_data, 
                                      desc="data is shuffled if true",
                                      default=True)
        self.drop_last = OptionBridge[bool]("drop_last", drop_last, 
                                    desc="left over samples that don't fit "\
                                    + "in a batch are dropped if true",
                                    default=True)
        self.num_workers = OptionBridge[int]("num_workers", num_workers,
                                             desc="Number of workers required "\
                                             +" to load the data",
                                             default= 0)
        super(DataConfig, self)._named_params()
