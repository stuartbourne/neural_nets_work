#include <Neuron.hpp>
/*
using namespace sb_nn;

template <typename I, typename O>
const bool NeuralNet<I, O>::add_hidden_neuron(Neuron<I, O> neuron){
    hidden_neurons_.push_back(neuron);
    return true;
}

template <typename I, typename O>
const bool NeuralNet<I, O>::add_training_data(  std::vector<std::vector<I>> input_training_data,
                                                std::vector<O> output_training_data){
    //check to make sure the dimensionality matches the amount of input lines
    assert(input_training_data.size() == output_training_data.size() && "Dimensionality of input data must match dimensionality of the output data!");
    for (auto &in : input_training_data){
        assert(in.size() == num_features_ && "Dimensionality of input data must match dimensionality of the neuron inputs!" ); )
    }
    training_set_in_ = input_training_data;
    training_set_out_ = output_training_data;
    return true;
}

template <typename I, typename O>
const double NeuralNet<I, O>::feed_forward(){
    //go through all neurons in the netowkr and activate them. This includes the output node
    
    //for each input, we'll want to compute w1x1 + w2x2 + w3x3 + bias
    //double activation_in = dot<I>(weights, values) + bias;
    //now compute activation energy TODO in future, use passed in activation type
    // double activation_out = sigmoid(activation_in);
    //TODO propagate feed forward for each neuron in network
    return -1;
}

template <typename I, typename O>
const double Neuron<I, O>::backpropagate(O desired_output, double activation_out){
    //Calculate the output error relative to the output of the activation functions
    double error = activation_out - desired_output;
    //Now lets evaluate the change in cost relative to weight and minimize that 
    //we will do this using the chain rule by calculating dcost/dprediction and multiplying with dprediction/dz
    //backpropagation step 2
    double dcost_dpred = error;
    double dpred_dz = sigmoid_d1(activation_out); 
    double dcost_dz = dcost_dpred * dpred_dz;
    return dcost_dz;
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
            //for now we will assume three layers (input, hidden, output) for simplicity
            double z = feed_forward(input_weights_, training_set_in_.at(j), bias_);
            //Now start the backpropagation by first calculating the error
            double dcost_dz = backpropagate(training_set_out_.at(j), z);
            //Now we've got the desired rate of change, lets apply it to our weights and biases
            //now apply that calculated change to the weights and bias
            for (size_t p = 0; p < input_weights_.size(); ++p){
                input_weights_.at(p) -= learning_rate_ * training_set_in_.at(j).at(p) * (dcost_dz);
            }
            
            bias_ -= learning_rate_ * dcost_dz;
        }
    }
    std::vector<I> testing_point{0, 1, 0};
    std::cout << feed_forward(input_weights_, testing_point, bias_);
    return true;
} 

//lets define the neural net class to only operate with specific inputs/outputs
// template class sb_nn::NeuralNet<int, int>;
// template class sb_nn::NeuralNet<int, double>;
// template class sb_nn::NeuralNet<double, int>;
template class sb_nn::NeuralNet<double, double>;*/