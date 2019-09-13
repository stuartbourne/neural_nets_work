#include <iostream>
#include <Neuron.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <matplotlibcpp.h>
#include <stdlib.h>
#include <time.h>
namespace plt = matplotlibcpp;

int main(int argc, char** argv){

    using namespace sb_nn;/*
    Neuron<double, double> n(2, 0.05, ActivationFunction::SIGMOID);
    n.add_input(0.23);  //add the inputs and their weights
    n.add_input(0.88);
    n.add_input(0.42);

    std::vector<std::vector<double>> training_inputs;
    training_inputs.push_back({0, 1, 0});
    training_inputs.push_back({0, 0, 1});
    training_inputs.push_back({1, 0, 0});
    training_inputs.push_back({1, 1, 0});
    training_inputs.push_back({1, 1, 1});
    
    std::vector<double> training_outputs;
    training_outputs.push_back(1);
    training_outputs.push_back(0);
    training_outputs.push_back(0);
    training_outputs.push_back(1);
    training_outputs.push_back(1);
    n.add_training_data(training_inputs, training_outputs);
    n.train();*/
    return 0;
}
