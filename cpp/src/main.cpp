#include <iostream>
#include <NeuralNet.hpp>
#include <Neuron.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <matplotlibcpp.h>
#include <stdlib.h>
#include <time.h>
namespace plt = matplotlibcpp;

int main(int argc, char** argv){

    using namespace sb_nn;
    // Neuron<double, double> n(2, 0.05, ActivationFunction::SIGMOID);
    // NeuronInput in{0};
    // n.add_neuron_input(in);
    //n.set_neuron_values({1});
    //n.activate();
    
    std::vector<std::vector<double>> training_data_in;
    std::vector<double> training_data_out;
    training_data_in.push_back({0, 1, 0});
    training_data_in.push_back({0, 0, 1});
    training_data_in.push_back({1, 1, 0});
    training_data_in.push_back({1, 0, 1});
    training_data_in.push_back({0, 1, 1});
    training_data_out.push_back(1);
    training_data_out.push_back(0);
    training_data_out.push_back(0);
    training_data_out.push_back(1);
    training_data_out.push_back(1);
    NeuralNet<double> nn(1, 0.05);
    nn.set_feature_num(3);  //three input features
    //nn.initialize_hidden_neurons(1);  
    nn.set_training_data(training_data_in, training_data_out);
    nn.train_network();
    // std::cout << "Activation energy: " << n.activation_energy_ << "\n";
    return 0;
}
