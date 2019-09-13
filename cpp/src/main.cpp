#include <iostream>
#include <Neuron.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <matplotlibcpp.h>
#include <stdlib.h>
#include <time.h>
namespace plt = matplotlibcpp;

int main(int argc, char** argv){

    using namespace sb_nn;
    Neuron<double, double> n(2, 0.05, ActivationFunction::SIGMOID);
    NeuronInput in{0};
    n.add_neuron_input(in);
    n.activate();
    return 0;
}
