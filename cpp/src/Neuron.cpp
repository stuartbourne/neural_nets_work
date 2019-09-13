#include <Neuron.hpp>
#include <type_traits>

using namespace sb_nn;
 
template <typename I>
const double dot(std::vector<I> v1, std::vector<I> v2);

template <typename I, typename O>
const double Neuron<I, O>::sigmoid(const double x){
    return 1/(1 + exp(-x));
}
template <typename I, typename O>
const double Neuron<I, O>::sigmoid_d1(const double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

template <typename I, typename O>
const bool Neuron<I, O>::add_neuron_input(NeuronInput input){
    neuron_inputs_.push_back(input);
    return true;
}

template <typename I, typename O>
const bool Neuron<I, O>::set_neuron_values(std::vector<I> values){
    assert(values.size() == neuron_inputs_.size() && "Passed in values must have the same dimensions as the neuron inputs!");
    for (size_t i = 0; i < values.size(); i++){
        neuron_inputs_.at(i).value = values.at(i);
    }
    return true;
}

template <typename I, typename O>
const bool Neuron<I, O>::activate(){
    //for each input, we'll want to compute w1x1 + w2x2 + w3x3 + bias
    std::vector<double> input_weights;
    std::vector<double> input_values;
    for (auto &input : neuron_inputs_){
        if (input.value == std::nullopt){
            std::cerr << "Input value is null! Cannot activate neuron!" <<std::endl;
            return false;
        }
        input_values.push_back(*input.value);
        input_weights.push_back(input.weight);
    }
    double activation_in = dot<double>(input_weights, input_values) + bias_;
    //now compute activation energy TODO in future, use passed in activation type
    activation_energy_ = sigmoid(activation_in);
    return true;
}

//Non-member functions
template <typename I>
const double dot(std::vector<I> v1, std::vector<I> v2){
    //static_assert(std::is_integral<I>::value, "Passed in template parameter must be integral!");
    double sum = 0;
    for (size_t i = 0; i < v1.size(); ++i){
        sum += v1.at(i) * v2.at(i);
    }
    return sum;
}

//explicitly define template types that are able to be used
//This allows for implementaiton and definition to be separated and ensures
//type safety across class functions
// template class sb_nn::Neuron<int, int>;
// template class sb_nn::Neuron<int, double>;
template class sb_nn::Neuron<double, double>;