import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import yaml
import numpy as np
from neurons.config import generate_data_set_from_config, generate_neuron_from_config, generate_optimizer_from_config

with open("configs/adaline_and.yml", 'r') as yml_file:
    config = yaml.load(yml_file)

training_set_config = config['data_set']['training_data']
validation_set_config = config['data_set']['validation_data']
neuron_config = config['neuron']
optimizer_config = config['optimizer']

#####Required number of epochs, depends on training examples#########
# training_samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50]
# tries = 10
# results_epochs = np.zeros((len(training_samples), tries))
# for k in range(len(training_samples)):
#     print(k)
#     for i in range(tries):
#         training_set_config['samples'] = training_samples[k]
#         training_data = generate_data_set_from_config(training_set_config)
#         validation_data = generate_data_set_from_config(validation_set_config)
#         neuron = generate_neuron_from_config(neuron_config)
#         optimizer = generate_optimizer_from_config(optimizer_config)
#         trains, validate, epochs = optimizer.train(neuron, training_data, validation_data, config['visualize'])
#
#         results_epochs[k][i] = epochs
#
# means = np.mean(results_epochs, axis=1)
# stds = np.std(results_epochs, axis=1)
# fig, ax0 = plt.subplots(nrows=1, sharex=True)
# ax0.errorbar(np.asarray(training_samples)*4, means, yerr=stds, fmt='-o', color='green')
# ax0.set_title('Wymagana liczba epok w zależności od liczby danych treningowych. \nPerceptron bipolar AND.')
#
#
# green_patch = mpatches.Patch(color='green', label='Liczba epok')
# plt.legend(handles=[green_patch])
# plt.xlabel('Liczba danych treningowych')
# plt.ylabel('Liczba epok')
# plt.show()
##########################################################################

#####Required number of epochs, depends on learning rate##################
# learning_rates = np.arange(0.01, 5, 0.5)
# tries = 20
# results_epochs = np.zeros((len(learning_rates), tries))
# for k in range(len(learning_rates)):
#     print(k)
#     for i in range(tries):
#         optimizer_config['learning_rate'] = learning_rates[k]
#         training_data = generate_data_set_from_config(training_set_config)
#         validation_data = generate_data_set_from_config(validation_set_config)
#         neuron = generate_neuron_from_config(neuron_config)
#         optimizer = generate_optimizer_from_config(optimizer_config)
#         trains, validate, epochs = optimizer.train(neuron, training_data, validation_data, config['visualize'])
#
#         results_epochs[k][i] = epochs
#
# means = np.mean(results_epochs, axis=1)
# stds = np.std(results_epochs, axis=1)
# fig, ax0 = plt.subplots(nrows=1, sharex=True)
# ax0.errorbar(np.asarray(learning_rates), means, yerr=stds, fmt='-o', color='green')
# ax0.set_title('Wymagana liczba epok w zależności od stałej uczenia. \nPerceptron unipolar AND.')
#
#
# green_patch = mpatches.Patch(color='green', label='Stała uczenia')
# plt.legend(handles=[green_patch])
# plt.xlabel('Stała uczenia')
# plt.ylabel('Liczba epok')
# plt.show()
##########################################################################

#####Required number of epochs, depends on weights##################
# weights_ranges = [(-i, i) for i in np.arange(0, 1.1, 0.1)]
# tries = 20
# results_epochs = np.zeros((len(weights_ranges), tries))
# for k in range(len(weights_ranges)):
#     print(k)
#     for i in range(tries):
#         neuron_config['weights_range'] = weights_ranges[k]
#         training_data = generate_data_set_from_config(training_set_config)
#         validation_data = generate_data_set_from_config(validation_set_config)
#         neuron = generate_neuron_from_config(neuron_config)
#         optimizer = generate_optimizer_from_config(optimizer_config)
#         trains, validate, epochs = optimizer.train(neuron, training_data, validation_data, config['visualize'])
#
#         results_epochs[k][i] = epochs
#
# means = np.mean(results_epochs, axis=1)
# stds = np.std(results_epochs, axis=1)
# fig, ax0 = plt.subplots(nrows=1, sharex=True)
# ax0.errorbar(np.asarray([ll[1] for ll in weights_ranges]), means, yerr=stds, fmt='-o', color='green')
# ax0.set_title('Wymagana liczba epok w zależności od initializowanych wag. \nPerceptron unipolar AND.')
#
#
# green_patch = mpatches.Patch(color='green', label='Liczba epok')
# plt.legend(handles=[green_patch])
# plt.xlabel('Zakres wag')
# plt.ylabel('Liczba epok')
# plt.show()
##########################################################################
# for k in range(30):
#     i = 0
#     for samples in training_samples:
#         training_set_config['samples'] = samples
#         training_data = generate_data_set_from_config(training_set_config)
#         validation_data = generate_data_set_from_config(validation_set_config)
#         neuron = generate_neuron_from_config(neuron_config)
#         optimizer = generate_optimizer_from_config(optimizer_config)
#
#         trains, validate, epochs = optimizer.train(neuron, training_data, validation_data, config['visualize'])
#         results[k, i] = epochs
#         i += 1
# colors = ['red', 'green', 'blue', 'black', 'yellow', 'purple']
# for n in range(len(training_samples)):
#     stds_epochs = np.std(results, axis=0)
#     means_epochs = np.mean(results, axis=0)
#     fig, ax0 = plt.subplots(nrows=1, sharex=True)
#     ax0.errorbar(training_samples, means_epochs, yerr=stds_epochs, fmt='-o', color=colors[n])
#




# # plt.show()
samples = 1
epochs = optimizer_config['epochs']
traines = np.zeros((samples, epochs))
validates = np.zeros((samples, epochs))

for i in range(samples):
    training_data = generate_data_set_from_config(training_set_config)
    validation_data = generate_data_set_from_config(validation_set_config)
    neuron = generate_neuron_from_config(neuron_config)
    optimizer = generate_optimizer_from_config(optimizer_config)

    trains, validate, epoch = optimizer.train(neuron, training_data, validation_data, config['visualize'])

    traines[i, :] = trains
    validates[i, :] = validate

print(neuron.weights)
print(neuron.bias)





# print(traines.shape)
# epochs = np.arange(0, epochs)
# stds_trains = np.std(traines, axis=0)
# means_trains = np.mean(traines, axis=0)
#
# stds_validates = np.std(validates, axis=0)
# means_validates = np.mean(validates, axis=0)
#
# print(stds_trains.shape)
# print(means_trains.shape)
# print(epochs.shape)
# fig, ax0 = plt.subplots(nrows=1, sharex=True)
# ax0.errorbar(epochs, means_trains, yerr=stds_trains, fmt='-o', color='green')
# ax0.errorbar(epochs, means_validates, yerr=stds_validates, fmt='-o', color='red')
# ax0.set_title('Error depends by epoch. Adaline neuron.')
#
#
# red_patch = mpatches.Patch(color='red', label='Training data error')
# green_patch = mpatches.Patch(color='green', label='Validate data error')
# plt.legend(handles=[red_patch, green_patch])
# plt.xlabel('epochs')
# plt.ylabel('erroradaline')
# plt.show()
#     plt.plot(np.arange(0, len(trains)), trains, color='green')
#     plt.plot(np.arange(0, len(trains)), validate, color='red')
#     plt.show()
#
plt.xlim((-4.1, 4.1))
plt.ylim((-4.1, 4.1))

for sample in validation_data:
    result = neuron.predict(sample)
    if result == 1:
        plt.plot(sample[0], sample[1], 'or')
    else:
        plt.plot(sample[0], sample[1], 'ob')

plt.show()