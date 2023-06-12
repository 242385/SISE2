import random
import sys


class MLP:

    def __init__(self, input_layer, hidden_layers, output_layer):
        self.num_inputs = len(input_layer.neurons)
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.num_outputs = len(output_layer.neurons)

        self.init_weights_from_inputs_to_hidden_layer_neurons(self.randomWeights(len(input_layer.neurons) * len(hidden_layers[0].neurons)))

        for i in range(1, len(hidden_layers)):
            self.init_weights_from_hidden_layer_neurons_to_next_hidden_layer_neurons(i, self.randomWeights(len(hidden_layers[i - 1].neurons) * len(hidden_layers[i].neurons)))

        self.init_weights_from_last_hidden_layer_neurons_to_output_layer_neurons(self.randomWeights(len(hidden_layers[-1].neurons) * len(output_layer.neurons)))

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        self._init_weights_between_layers(self.num_inputs, self.hidden_layers[0], hidden_layer_weights)

    def init_weights_from_hidden_layer_neurons_to_next_hidden_layer_neurons(self, i, hidden_layer_weights):
        self._init_weights_between_layers(len(self.hidden_layers[i - 1].neurons), self.hidden_layers[i], hidden_layer_weights)

    def init_weights_from_last_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        self._init_weights_between_layers(len(self.hidden_layers[-1].neurons), self.output_layer, output_layer_weights)

    def _init_weights_between_layers(self, num_neurons_in, layer_to, weights):
        weight_num = 0
        for neuron_to in layer_to.neurons:
            neuron_to.weights = []
            for _ in range(num_neurons_in):
                if weights is None:
                    neuron_to.weights.append(random.random())
                else:
                    if weight_num < len(weights):
                        neuron_to.weights.append(weights[weight_num])
                        weight_num += 1
                    else:
                        raise ValueError('The provided weights list is not long enough for the number of connections.')

    def forwardPropagation(self, inputs):
        current_inputs = inputs

        for hidden_layer in self.hidden_layers:
            current_inputs = hidden_layer.forwardPropagation(current_inputs)

        final_outputs = self.output_layer.forwardPropagation(current_inputs)
        return final_outputs

    # Uses online learning, ie updating the weights after each training case
    def backPropagation(self, training_inputs, training_outputs, learning_rate, momentum_coeff, consider_bias, consider_momentum):
        self.forwardPropagation(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # ∂E/∂zⱼ
            try:
                pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])
            except IndexError:
                print("An index error occured. Is output neurons number correct for this dataset?")
                sys.exit()

        # 2. Hidden neurons deltas
        pd_errors_wrt_hidden_neuron_total_net_input = []
        for l in range(len(self.hidden_layers) -1, -1, -1):
            pd_errors_wrt_hidden_neuron_total_net_input.insert(0, [0] * len(self.hidden_layers[l].neurons))
            for h in range(len(self.hidden_layers[l].neurons)):

                # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
                d_error_wrt_hidden_neuron_output = 0

                # If we're in the last hidden layer
                if l == len(self.hidden_layers) - 1:
                    for o in range(len(self.output_layer.neurons)):
                        d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
                else:  # if we're in any hidden layer except the last
                    for o in range(len(self.hidden_layers[l + 1].neurons)):
                        d_error_wrt_hidden_neuron_output += pd_errors_wrt_hidden_neuron_total_net_input[l + 1][o] * self.hidden_layers[l + 1].neurons[o].weights[h]

                # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                pd_errors_wrt_hidden_neuron_total_net_input[0][h] = d_error_wrt_hidden_neuron_output * self.hidden_layers[l].neurons[h].calculate_pd_total_net_input_wrt_input()

        # Momentum factors for output and hidden neurons
        momentum_output = [0] * len(self.output_layer.neurons)
        momentum_hidden = [[0] * len(layer.neurons) for layer in self.hidden_layers]

        # 3. For output layer
        for o in range(len(self.output_layer.neurons)):
            # Update weights
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # Calculate the gradient and the change of weight
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)
                weight_delta = learning_rate * pd_error_wrt_weight
                # Consider momentum
                if consider_momentum:
                    weight_delta += momentum_coeff * momentum_output[o]
                    momentum_output[o] = weight_delta
                self.output_layer.neurons[o].weights[w_ho] -= weight_delta

        # 4. For hidden layers
        for l in range(len(self.hidden_layers)):
            for h in range(len(self.hidden_layers[l].neurons)):
                for w_ih in range(len(self.hidden_layers[l].neurons[h].weights)):
                    # Calculate the gradient and the change of weight
                    pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[l][h] * self.hidden_layers[l].neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)
                    weight_delta = learning_rate * pd_error_wrt_weight
                    # Consider momentum
                    if consider_momentum:
                        weight_delta += momentum_coeff * momentum_hidden[l][h]
                        momentum_hidden[l][h] = weight_delta
                    self.hidden_layers[l].neurons[h].weights[w_ih] -= weight_delta

        # 5. Update output neuron biases
        if consider_bias:
            for o in range(len(self.output_layer.neurons)):
                # ∂Eⱼ/∂b = ∂E/∂zⱼ * ∂zⱼ/∂b, note that ∂zⱼ/∂b = 1
                pd_error_wrt_bias = pd_errors_wrt_output_neuron_total_net_input[o]

                # Δb = α * ∂Eⱼ/∂b
                self.output_layer.neurons[o].bias -= learning_rate * pd_error_wrt_bias

            # 6. Update hidden neuron biases
            for l in range(len(self.hidden_layers) - 1, -1, -1):
                for h in range(len(self.hidden_layers[l].neurons)):
                    # ∂Eⱼ/∂b = ∂E/∂zⱼ * ∂zⱼ/∂b, note that ∂zⱼ/∂b = 1
                    pd_error_wrt_bias = pd_errors_wrt_hidden_neuron_total_net_input[l][h]

                    # Δb = α * ∂Eⱼ/∂b
                    self.hidden_layers[l].neurons[h].bias -= learning_rate * pd_error_wrt_bias

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.forwardPropagation(training_inputs)
            for o in range(len(training_outputs)):
                try:
                    total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
                except IndexError:
                    print("An index error occured. Is output neurons number correct for this dataset?")
                    sys.exit()
        return total_error

    def randomWeights(self, count):
        weights = list()
        for i in range(0, count):
            r = random.Random()
            x = r.uniform(-0.5, 0.5)
            weights.append(x)
        return weights
