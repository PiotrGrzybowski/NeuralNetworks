from ast import literal_eval as make_tuple

from neurons.data_generator import LogicalFunctionsGenerator
from neurons.neuron import Neuron
from neurons.optimizer import Optimizer


def generate_data_set_from_config(config):
    config['samples'] = int(config['samples'])
    config['high_range'] = make_tuple(config['high_range'])
    config['low_range'] = make_tuple(config['low_range'])

    return LogicalFunctionsGenerator.generate_data_set(**config)


def generate_neuron_from_config(config):
    config['number_of_inputs'] = int(config['number_of_inputs'])
    config['weights_range'] = make_tuple(config['weights_range'])
    config['bias'] = float(config['bias'])

    return Neuron(**config)


def generate_optimizer_from_config(config):
    config['epochs'] = int(config['epochs'])
    config['learning_rate'] = float(config['learning_rate'])
    config['stop_error'] = float(config['stop_error'])

    return Optimizer(**config)
