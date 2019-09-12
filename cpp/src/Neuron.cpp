#include <Neuron.hpp>

using namespace sb_nn;

const bool Neuron::add_input(NeuronInput input){
    neuron_inputs_.push_back(input);
    std::cout << "Adding an input" << std::endl;
    return true;
} 

void Neuron::train(std::vector<int> input_data, std::vector<double> output_data){
    std::cout << "Training data...." << std::endl;
} 