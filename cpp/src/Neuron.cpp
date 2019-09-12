#include <Neuron.hpp>
#include <type_traits>

using namespace sb_nn;

template <typename I>
const double dot(std::vector<I> v1, std::vector<I> v2);


template <typename I, typename O>
const bool Neuron<I, O>::add_input(I input_weight){
    std::cout << "Adding an input" << std::endl;
    input_weights_.push_back(input_weight);
    return true;
} 

template <typename I, typename O>
const bool Neuron<I, O>::add_training_data( std::vector<std::vector<I>> input_training_data,
                                            std::vector<std::vector<O>> output_training_data){
    //check to make sure the dimensionality matches the amount of input lines
    assert(input_training_data.size() == output_training_data.size() && "Dimensionality of input data must match dimensionality of neuron inputs!");
    for (auto &in : input_training_data){
        assert(in.size() == input_weights_.size());
    }
    training_set_in_ = input_training_data;
    training_set_out_ = output_training_data;
    return true;
}

template <typename I, typename O>
const bool Neuron<I, O>::train(){
    //should check dimensionality of input data to ensure it is the proper dimensions...
    std::cout << "Training data...." << std::endl;
    assert(training_set_in_.size() == training_set_out_.size() && "Must have same number of input and output training data sets!");
    if (training_set_in_.size() <= 0 || training_set_out_.size() <= 0){
        std::cout << "No training data supplied!" << std::endl;
        return false;
    }
    for (unsigned int i= 0; i < num_epochs_; ++i){
        for (size_t j = 0; j < training_set_in_.size(); ++j){
            //for each input, we'll want to compute w1x1 + w2x2 + w3x3 + bias
            double XW = dot<I>(input_weights_, training_set_in_.at(j)) + bias_;
            //now compute activation energy TODO in future, use passed in activation type
            double z = sigmoid(XW);
            //Now start the backpropagation by first calculating the error
            double error = z - training_set_out_.at(j).at(0);
            //Now lets evaluate the change in cost relative to weight and minimize that 
            //we will do this using the chain rule by calculating dcost/dprediction and multiplying with dprediction/dz
            //backpropagation step 2
            double dcost_dpred = error;
            double dpred_dz = sigmoid_d1(z); 
            double dcost_dz = dcost_dpred * dpred_dz;
            //Now we've got the desired rate of change, lets apply it to our weights and biases
            //now apply that calculated change to the weights and bias
            for (size_t p = 0; p < input_weights_.size(); ++p){
                input_weights_.at(p) -= learning_rate_ * training_set_in_.at(j).at(p) * (dcost_dz);
            }
            
            bias_ -= learning_rate_ * dcost_dz;
        }
        // std::cout << "bias2: " << bias_ << "\nweights: " << input_weights_.at(0) << "\n" << input_weights_.at(1) << "\n";

    }
    std::cout << "bias2: " << bias_ << "\nweights: " << input_weights_.at(0) << "\n" << input_weights_.at(1) << "\n";
    return true;
} 

template <typename I, typename O>
const double Neuron<I, O>::sigmoid(const double x){
    return 1/(1 + exp(-x));
}
template <typename I, typename O>
const double Neuron<I, O>::sigmoid_d1(const double x){
    return sigmoid(x) * (1 - sigmoid(x));
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
//This allows for implementaiton and definition to be separated and ensures type safety
template class sb_nn::Neuron<int, int>;
template class sb_nn::Neuron<double, double>;
template class sb_nn::Neuron<int, double>;