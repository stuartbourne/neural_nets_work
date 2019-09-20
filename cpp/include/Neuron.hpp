#ifndef SBNEURON
#define SBNEURON
#include <iostream>
#include <vector>
#include <math.h>
#include <optional>

/**author: Stuart Bourne
 * Date: September 14th, 2019
 * This code is free for use, it serves the purpose of introducing encapsulation into the topic of neural networks.
 * This class represents a neuron class with a sigmoid activation function. Future work includes introducing other
 * activation functions. This class should be used in conjunction with the NeuralNet class associated.
*/
namespace sb_nn{
    enum class ActivationFunction : char {
        SIGMOID = 's',
        SOFTMAX = 'm',
    };
    template <typename T>
    struct NeuronInput{
        double weight;
        std::optional<T> value = std::nullopt;
    };
    template <typename T>
    class Neuron{
        //typedef NeuronInput std::pair<I, std::optional<I>>;
        public:
            Neuron( double bias, ActivationFunction type) : 
                    bias_(bias),
                    activation_fn_(type) {};
            Neuron() : bias_(0), activation_fn_(ActivationFunction::SIGMOID) {};
            const bool add_neuron_input(NeuronInput<T> input);
            const bool set_neuron_values(std::vector<T> values);
            const bool activate();  //calculates the dot product of inputs/weights and passes through
            const double sigmoid_d1(const double);
            const double sigmoid(const double);
            
        private:
            double bias_;
            double activation_energy_;
            T output_energy_;
            std::vector<NeuronInput<T>> neuron_inputs_;      //Define the weights to the neuron
            ActivationFunction activation_fn_;
            template <typename O>
            friend class NeuralNetClassifier;
    };
}
#endif