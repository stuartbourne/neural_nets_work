#include <iostream>
#include <NeuralNetClassifier.hpp>
#include <Neuron.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <matplotlibcpp.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
namespace plt = matplotlibcpp;

const char* PYTHON_DATAFILE="/Users/stuartbourne/Documents/personal_development/neural_nets/python/classification2_data.txt";

const bool read_python_data(std::vector<std::vector<double>> &training_data_in, std::vector<double> &training_data_out){
    //lets populate the training data set
    std::ifstream data_file;
    data_file.open(PYTHON_DATAFILE);
    if(data_file.is_open()){
        std::string line;
        while (std::getline(data_file, line)){
            char delim = ' ';
            double x = std::stod(line.substr(0, line.find(delim)));
            line.erase(0, line.find(delim) + 1);
            double y = std::stod(line.substr(0, line.find(delim)));
            line.erase(0, line.find(delim) + 1);
            double res = std::stod(line);
            training_data_in.push_back({x, y});
            training_data_out.push_back(res);
        }
        data_file.close();
        return true;
    } else {
        std::cout << "Unable to open python data file: " << PYTHON_DATAFILE << std::endl;
        return false;
    }
}

int main(int argc, char** argv){
    using namespace sb_nn;
    std::vector<std::vector<double>> training_data_in;
    std::vector<double> training_data_out;
    int feature_num;
    if (read_python_data(training_data_in, training_data_out)){
        std::cout << "Using default python file" << std::endl;
        feature_num = 2;
    } else {
        std::cout << "Populating basic data set" << std::endl;
        training_data_in.push_back({0, 1, 0});
        training_data_in.push_back({0, 0, 1});
        training_data_in.push_back({1, 0, 0});
        training_data_in.push_back({1, 1, 0});
        training_data_in.push_back({1, 1, 1});
        training_data_out = {1, 0, 0, 1, 1};
        feature_num = 3;
    }
    NeuralNetClassifier<double> nn(2000, 0.2);
    nn.set_feature_num(feature_num);  //two input features
    nn.add_hidden_layer(6);  
    nn.set_training_data(training_data_in, training_data_out); 
    nn.train_network();
    std::vector<double> sample_input{0, 1};
    std::cout << nn.fire_network(sample_input) << std::endl;
    sample_input = {1, -0.5};
    std::cout << nn.fire_network(sample_input) << std::endl;
    sample_input = {0, 3};
    std::cout << nn.fire_network(sample_input) << std::endl;
    sample_input = {1, -3};
    std::cout << nn.fire_network(sample_input) << std::endl;
    return 0;
}
