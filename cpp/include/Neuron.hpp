#include <iostream>
#include <vector>

namespace sb_nn{
    //typedef int inputtype;    //TODO make this class templated
    //typedef double outputtype;
    struct ActivationType{
        static const char SIGMOID = 's';
        char function = SIGMOID;    //TODO refactor this?
    };
    struct NeuronInput{
        double weight;
        int value;
    };

    class Neuron{
        public:
            Neuron( double bias, double learning_rate, ActivationType type) : 
                    bias_(bias), 
                    learning_rate_(learning_rate),
                    activation_fn_(type) {};
            const bool add_input(NeuronInput input);
            void train(std::vector<int> input_data, std::vector<double> output_data);   //This should take in an input/output data set.
        private:
            double bias_;
            double learning_rate_;
            ActivationType activation_fn_;
            std::vector<NeuronInput> neuron_inputs_;    //Define the actual inputs to the neuron
            //now for the training data?

    };
}
