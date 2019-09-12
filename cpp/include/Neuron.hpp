#include <iostream>
#include <vector>
#include <math.h>

namespace sb_nn{
    //typedef int inputtype;    //TODO make this class templated
    //typedef double outputtype;
    enum ActivationFunction : char {
        SIGMOID = 's',
    };
    template <typename I, typename O>
    class Neuron{
        public:
            Neuron( double bias, double learning_rate, ActivationFunction type, int num_epochs = 2000) : 
                    bias_(bias), 
                    learning_rate_(learning_rate),
                    activation_fn_(type),
                    num_epochs_(num_epochs) {};
            const bool add_input(I input);
            const bool train();   //This uses the previously gathered input and output data sets to train the weights/bias
            const bool add_training_data(   std::vector<std::vector<I>> input_training_data,
                                            std::vector<O> output_training_data);
            const double feed_forward(std::vector<I> weights, std::vector<I> values, I bias);
            
        private:
            double bias_;
            double learning_rate_;
            int num_epochs_;
            ActivationFunction activation_fn_;
            std::vector<I> input_weights_;    //Define the actual inputs to the neuron (feature dimension)
            std::vector<O> neuron_outputs_;   //Define outputs of the neuron. For the initial exapmle there will only be one.
            std::vector<std::vector<I>> training_set_in_;
            std::vector<O> training_set_out_;   //one dimensional output
            const double sigmoid(const double);
            const double sigmoid_d1(const double);
            const double backpropagate(O desired_output, double activation_out);
    };
}
