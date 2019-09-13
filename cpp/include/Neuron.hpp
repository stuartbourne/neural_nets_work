#ifndef SBNEURON
#define SBNEURON
#include <iostream>
#include <vector>
#include <math.h>
#include <optional>

class NeuralNet;

namespace sb_nn{
    enum ActivationFunction : char {
        SIGMOID = 's',
    };
    struct NeuronInput{
        double weight;
        std::optional<double> value = std::nullopt;
    };
    template <typename I, typename O>
    class Neuron{
        //typedef NeuronInput std::pair<I, std::optional<I>>;
        public:
            Neuron( double bias, double learning_rate, ActivationFunction type) : 
                    bias_(bias),
                    activation_fn_(type) {};
            Neuron() : bias_(0), activation_fn_(ActivationFunction::SIGMOID) {};
            const bool add_neuron_input(NeuronInput input);
            const bool set_neuron_values(std::vector<I> values);
            friend NeuralNet;

        protected:
            double bias_;
            double activation_energy_;
            std::vector<NeuronInput> neuron_inputs_;      //Define the weights to the neuron
            const bool activate();  //calculates the dot product of inputs/weights and passes through
            const double sigmoid_d1(const double);
            const double sigmoid(const double);
            
        private:
            ActivationFunction activation_fn_;
    };
}
#endif