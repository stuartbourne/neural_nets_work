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
    std::vector<std::vector<double>> training_data_in;
    training_data_in.push_back({0, 1});
    // training_data_in.push_back({0, 0});
    // training_data_in.push_back({1, 1});
    // training_data_in.push_back({1, 0});
    // training_data_in.push_back({0, 1});
    std::vector<double> training_data_out{1};//, 0, 0, 1, 1};
    NeuralNet<double> nn(1, 0.01);
    nn.set_feature_num(2);  //two input features
    nn.add_hidden_layer(4);  
    nn.set_training_data(training_data_in, training_data_out);
    nn.train_network();
    //std::vector<double> sample_input{1, 1, 1};
    //std::cout << nn.fire_network(sample_input) << std::endl;
    return 0;
}
