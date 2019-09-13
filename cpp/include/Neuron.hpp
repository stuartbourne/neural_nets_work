#include <iostream>
#include <vector>
#include <math.h>
#include <optional>

namespace sb_nn{
    enum ActivationFunction : char {
        SIGMOID = 's',
    };

    template <typename I, typename O>
    class Neuron{
        struct NeuronInput{
            I weight;
            std::optional<I> value = std::nullopt;
        };
        public:
            Neuron( double bias, double learning_rate, ActivationFunction type) : 
                    bias_(bias),
                    activation_fn_(type) {};
            const bool add_neuron_input(NeuronInput input);
            const bool set_neuron_values(std::vector<I> values);
        
        protected:
            const bool activate();  //calculates the dot product of inputs/weights and passes through
            double bias_;
            std::vector<NeuronInput> neuron_inputs_;      //Define the weights to the neuron
            double activation_energy_;
            const double sigmoid_d1(const double);
            const double sigmoid(const double);
            
        private:
            ActivationFunction activation_fn_;
    };
}
