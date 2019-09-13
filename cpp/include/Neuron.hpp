#include <iostream>
#include <vector>
#include <math.h>
#include <optional>

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
            const bool add_neuron_input(NeuronInput input);
            const bool set_neuron_values(std::vector<I> values);
        
        protected:
            double bias_;
            const bool activate();  //calculates the dot product of inputs/weights and passes through
            std::vector<NeuronInput> neuron_inputs_;      //Define the weights to the neuron
            double activation_energy_;
            const double sigmoid_d1(const double);
            const double sigmoid(const double);
            
        private:
            ActivationFunction activation_fn_;
    };
}
