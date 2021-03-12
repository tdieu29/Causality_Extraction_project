from CausalityExtraction.model import *
import wandb

#wandb.login()

config = {
    'cuda_devices': 0,
    'cpu_core': 1,
    'seed': 2,
    'k_fold': 0,
    'lstm_size': 256,
    'dropout_rate': 0.5,
    'nb_head': 3,
    'size_per_head': 8,
    'learning_rate': 0.001,
    'clip_norm': 5.0,
    'num_epochs': 200,
    'batch_size': 16
}

#run = wandb.init(project='run', config = config_dict)
#config = wandb.config

def train_model(config=config):
  data_train = Data(config)
  extractor = CausalityExtractor(config)
  extractor.train(data_train)

train_model(config)