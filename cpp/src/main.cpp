#include <iostream>
#include <Neuron.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <matplotlibcpp.h>
#include <stdlib.h>
#include <time.h>
namespace plt = matplotlibcpp;

double sigmoid(const double x){
    return 1/(1 + exp(-x));
}

double sigmoid_der(const double x){
    return sigmoid(x) * (1 - sigmoid(x));
}

int main(int argc, char** argv){
    //First will directly emulate the initial python implementation, integrate eigen, matplotlibcpp, etc.
    //Then will restructure
    srand(42);
    Eigen::Vector3d weights;
    // weights(0) = rand()/double(RAND_MAX);
    // weights(1) = rand()/double(RAND_MAX);
    // weights(2) = rand()/double(RAND_MAX);
    weights(0) = 0.23;
    weights(1) = 0.88;
    weights(2) = 0.42;
    double bias = 2; //rand()/double(RAND_MAX);
    double learning_rate = 0.05;
    std::vector<Eigen::Vector3d> input_features;  //These will simply be supplied binary 3-vectors which represent smokers, obese, exercise booleans
    input_features.push_back(Eigen::Vector3d(0, 1, 0));
    input_features.push_back(Eigen::Vector3d(0, 0, 1));
    input_features.push_back(Eigen::Vector3d(1, 0, 0));
    input_features.push_back(Eigen::Vector3d(1, 1, 0));
    input_features.push_back(Eigen::Vector3d(1, 1, 1));
    std::vector<int> labels{1, 0, 0, 1, 1};
    std::cout << labels.size() << " " << input_features.size() << std::endl;
    assert(input_features.size() == labels.size());
    
    const int NUM_EPOCHS = 2000;
    for (unsigned int j = 0; j < NUM_EPOCHS; ++j){
        double error_sum = 0;
        for (size_t i = 0; i < input_features.size(); ++i){
            //for each input, we'll want to compute w1x1 + w2x2 + w3x3 + bias
            double XW = input_features.at(i).dot(weights) + bias;    
            //now pass that result to the sigmoid
            double z = sigmoid(XW);
            //now compare the result from the sigmoid with the expected result to generate the error
            //backpropagation step 1
            double error = z - labels.at(i);
            error_sum += error; 
            if (i % input_features.size() == 0)
                std::cout << "error: " << error_sum << std::endl;
            //Now lets evaluate the change in cost relative to weight and minimize that 
            //we will do this using the chain rule by calculating dcost/dprediction and multiplying with dprediction/dz
            //backpropagation step 2
            double dcost_dpred = error;
            double dpred_dz = sigmoid_der(z); 
            double dcost_dz = dcost_dpred * dpred_dz;
            //now apply that calculated change to the weights and bias
            weights -= learning_rate * input_features.at(i) * (dcost_dz);
            bias -= learning_rate * dcost_dz;
        }    
    }
    Eigen::Vector3d single_point(0, 1, 0);
    std::cout << sigmoid(weights.dot(single_point) + bias);

    using namespace sb_nn;
    Neuron n(0,0, ActivationType());
    std::vector<int> a{0};
    std::vector<double> b{1};
    n.train(a, b);
    return 0;
}
