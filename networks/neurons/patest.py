import matplotlib.pyplot as plt
import yaml

from neurons.config import generate_data_set_from_config, generate_neuron_from_config, generate_optimizer_from_config

with open("perceptron_and_bipolar.yml", 'r') as yml_file:
    config = yaml.load(yml_file)

training_set_config = config['data_set']['training_data']
validation_set_config = config['data_set']['validation_data']
neuron_config = config['neuron']
optimizer_config = config['optimizer']

training_data = generate_data_set_from_config(training_set_config)
validation_data = generate_data_set_from_config(validation_set_config)
neuron = generate_neuron_from_config(neuron_config)
optimizer = generate_optimizer_from_config(optimizer_config)
optimizer.train(neuron, training_data, validation_data, config['visualize'])


plt.xlim((-2.1, 2.1))
plt.ylim((-2.1, 2.1))

for sample in validation_data:
    result = neuron.predict(sample)
    if result == 1:
        plt.plot(sample[0], sample[1], 'or')
    else:
        plt.plot(sample[0], sample[1], 'ob')

plt.show()