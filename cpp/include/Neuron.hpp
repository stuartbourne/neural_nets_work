#ifndef SBNEURON
#define SBNEURON
#include <iostream>
#include <vector>
#include <math.h>
#include <optional>

namespace sb_nn{
    enum ActivationFunction : char {
        SIGMOID = 's',
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
            double bias_;
            double activation_energy_;
            std::vector<NeuronInput<T>> neuron_inputs_;      //Define the weights to the neuron
            const double sigmoid_d1(const double);
            const double sigmoid(const double);
            
        private:
            ActivationFunction activation_fn_;
    };
}
#endif