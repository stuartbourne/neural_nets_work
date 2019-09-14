#ifndef SBNEURALNET
#define SBNEURALNET
#include <Neuron.hpp>
#include <stdlib.h>

namespace sb_nn{
    template <typename I, typename O>
    class NeuralNet{
        public:
            NeuralNet(const int num_epochs, const double learning_rate) : 
                                            num_epochs_(num_epochs), 
                                            learning_rate_(learning_rate),
                                            num_features_(0) {};
            const bool set_feature_num(const int feat_num);
            const bool initialize_hidden_neurons(const int neuron_num);
            const bool train_network();   //This uses the previously gathered input and output data sets to train the weights/bias
            const bool set_training_data(   std::vector<std::vector<I>> input_training_data,
                                            std::vector<O> output_training_data);
            //const double run_network(std::vector<I> input_data);
        private:
            int num_features_;
            const int num_epochs_;
            const double learning_rate_;
            std::vector<std::vector<I>> training_set_in_;
            std::vector<O> training_set_out_;
            std::vector<Neuron<I,O>> hidden_neurons_;
            std::vector<I> network_inputs_;     //inputs will simply be a vector of the specified input type
            const bool feed_forward(std::vector<I> data_in);
            const bool backpropagate(O desired_output);
    };

    template <typename I, typename O>
    const bool NeuralNet<I, O>::initialize_hidden_neurons(const int neuron_num){
        if (num_features_ <= 0){
            std::cerr << "Cannot initialize hidden neurons! Please call set_feature_num first!" << std::endl;
            return false;
        }
        if (neuron_num < 0){
            std::cerr << "Neuron number must be greater than or equal to 0!" << std::endl;
            return false;
        }
        for (unsigned int i = 0; i < neuron_num; ++i){
            //initialize hidden neuron list with random weights and biases yet.
            //double rand_bias = (double) rand()/ (RAND_MAX);
            double rand_bias = 2;
            //create neuron with random bias values
            Neuron<I, O> hidden_neuron(rand_bias, learning_rate_, ActivationFunction::SIGMOID);
            for (unsigned int i = 0; i < num_features_; ++i){  
                //for each number of features in the input layer, add an input to the neuron
                //double rand_weight = (double) rand()/(RAND_MAX);
                double rand_weight = 2;
                hidden_neuron.add_neuron_input(NeuronInput{rand_weight});
            }
            //now add that initialized hidden neuron to the network
            hidden_neurons_.push_back(hidden_neuron);
        }
        return true;
    }

    template <typename I, typename O>
    const bool NeuralNet<I, O>::set_feature_num(const int feature_num){
        if (feature_num < 1){
            std::cerr << "Feature number must be greater than or equal to 1!" << std::endl;
            return false;
        }
        num_features_ = feature_num;
        return true;
    }

    template <typename I, typename O>
    const bool NeuralNet<I, O>::train_network(){
        /* for training_data_in, training_data_out in training_set:
                feedforward(training_data_in)
                backpropagate(training_data_out)
            
        */
        //should check dimensionality of input data to ensure it is the proper dimensions...
        std::cout << "Training data...." << std::endl;
        //assert(training_set_in_.size() == training_set_out_.size() && "Must have same number of input and output training data sets!");
        if (training_set_in_.size() <= 0 || training_set_out_.size() <= 0){
            std::cout << "No training data supplied!" << std::endl;
            return false;
        }
        assert(training_set_in_.size() == training_set_out_.size());
        for (unsigned int i= 0; i < num_epochs_; ++i){
            for (size_t j = 0; j < training_set_in_.size(); ++j){
                feed_forward(training_set_in_.at(j));
                backpropagate(training_set_out_.at(j));
                /*
                //for now we will assume three layers (input, hidden, output) for simplicity
                double z = feed_forward(input_weights_, training_set_in_.at(j), bias_);
                //Now start the backpropagation by first calculating the error
                double dcost_dz = backpropagate(training_set_out_.at(j), z);
                //Now we've got the desired rate of change, lets apply it to our weights and biases
                //now apply that calculated change to the weights and bias
                for (size_t p = 0; p < input_weights_.size(); ++p){
                    input_weights_.at(p) -= learning_rate_ * training_set_in_.at(j).at(p) * (dcost_dz);
                }
                
                bias_ -= learning_rate_ * dcost_dz;*/
            }
        }
        std::vector<I> testing_point{0, 1, 0};
        //std::cout << feed_forward(input_weights_, testing_point, bias_);
        return true;
    } 

    template <typename I, typename O>
    const bool NeuralNet<I, O>::set_training_data(  std::vector<std::vector<I>> input_training_data,
                                                    std::vector<O> output_training_data){
        //check to make sure the dimensionality matches the amount of input lines
        assert(input_training_data.size() == output_training_data.size() && "Dimensionality of input data must match dimensionality of the output data!");
        for (std::vector<I> in : input_training_data){
            if (in.size() != num_features_ ){
                std::cerr   << "Dimensionality of input data must match dimensionality of the neuron inputs!"  
                            << "\nin.size() = " << in.size()
                            << "\nnum_features_ = " << num_features_
                            << std::endl;
                return false;
            }
        }
        training_set_in_ = input_training_data;
        training_set_out_ = output_training_data;
        return true;
    }

    template <typename I, typename O>
    const bool NeuralNet<I, O>::feed_forward(std::vector<I> network_inputs){ 
        //TODO refactor to put this somewhere else
        assert( network_inputs.size() == num_features_ && 
                "Input training data must match number of network inputs!");
        std::vector<O> output_values;
        for (auto &hidden_neuron : hidden_neurons_){
            //Now we set the inputs for each hidden neuron
            hidden_neuron.set_neuron_values(network_inputs);
            //Now the hidden neurons have weights and values, should fire them.
            hidden_neuron.activate();
            //output_values.push_back(hidden_neuron.activation_energy_);
        }
        //Now we need to set the neurons in the output layer to have the inputs of the neurons in the hidden layer
        return true;
    }

    template <typename I, typename O>
    const bool NeuralNet<I, O>::backpropagate(O desired_output){
        //for now we'll start with one hidden neuron
        //Calculate the output error relative to the output of the activation functions
        if (hidden_neurons_.size() <= 0){
            std::cerr << "Cannot backpropagate an empty network!" << std::endl;
        }
        Neuron<I, O> output_neuron = hidden_neurons_.at(0);  //TODO make this more generic for multilayered networks
        O error = output_neuron.activation_energy_ - desired_output;
        O dcost_dpred = error;
        O dpred_dz = output_neuron.sigmoid_d1(output_neuron.activation_energy_);
        O dcost_dz = dcost_dpred * dpred_dz;
        std::cout << "dcost_dz: " << dcost_dz << std::endl;
        double weight_cost = 0;
        for (auto &input : output_neuron.neuron_inputs_){
            //lets change each weight with respect to their input multiplied by the cost gradient 
            input.weight -= learning_rate_ * *input.value * dcost_dz;
        }
        std::cout << "weight cost: " << weight_cost << std::endl;
        output_neuron.bias_ -= learning_rate_ * dcost_dz;
        
        std::cout << "Bias: " << output_neuron.bias_ << std::endl;
        //now we have the gradient, we must change the neuron inputs by -= lr * (value * gradient)
        /*
        for neuron in hidden neuron (since we know there is only one)
        
        //Now lets evaluate the change in cost relative to weight and minimize that 
        //we will do this using the chain rule by calculating dcost/dprediction and multiplying with dprediction/dz
        //backpropagation step 2
        double dcost_dpred = error;
        double dpred_dz = sigmoid_d1(activation_out); 
        double dcost_dz = dcost_dpred * dpred_dz;
        return dcost_dz;
        */
       return true;
    }
}
#endif