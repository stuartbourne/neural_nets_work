#include <Neuron.hpp>
#include <type_traits>

using namespace sb_nn;
 
template <typename T>
const double dot(std::vector<T> v1, std::vector<T> v2);

template <typename T>
const double Neuron<T>::sigmoid(const double x){
    return 1/(1 + exp(-x));
}
template <typename T>
const double Neuron<T>::sigmoid_d1(const double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

template <typename T>
const bool Neuron<T>::add_neuron_input(NeuronInput<T> input){
    neuron_inputs_.push_back(input);
    return true;
}

template <typename T>
const bool Neuron<T>::set_neuron_values(std::vector<T> values){
    assert(values.size() == neuron_inputs_.size() && "Passed in values must have the same dimensions as the neuron inputs!");
    for (size_t i = 0; i < values.size(); i++){
        neuron_inputs_.at(i).value = values.at(i);
    }
    return true;
}

template <typename T>
const bool Neuron<T>::activate(){
    //for each input, we'll want to compute w1x1 + w2x2 + w3x3 + bias
    std::vector<double> input_weights;
    std::vector<T> input_values;
    for (auto &input : neuron_inputs_){
        if (input.value == std::nullopt){
            std::cerr << "Input value is null! Cannot activate neuron!" <<std::endl;
            return false;
        }
        input_values.push_back(*input.value);
        input_weights.push_back(input.weight);
    }
    output_energy_ = dot<T>(input_weights, input_values);// + bias_;   //remove bias term for multilayered networks for simplicity
    //now compute activation energy TODO in future, use passed in activation type
    activation_energy_ = sigmoid(output_energy_);
    return true;
}

//Non-member functions
template <typename T>
const double dot(std::vector<T> v1, std::vector<T> v2){
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
template class sb_nn::Neuron<double>;