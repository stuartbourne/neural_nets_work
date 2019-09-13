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
    NeuralNet<double, double> nn(200, 0.05);
    nn.set_feature_num(2);  //two input features
    nn.set_hidden_neuron_num(2);
    nn.train();
    // std::cout << "Activation energy: " << n.activation_energy_ << "\n";
    return 0;
}
