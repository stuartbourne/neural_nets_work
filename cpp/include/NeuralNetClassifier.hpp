#ifndef SBNNCLASSIFIER
#define SBNNCLASSIFIER
#include <Neuron.hpp>
#include <NeuronLayer.hpp>
#include <stdlib.h>

//TODO make NeuralNet a friend class of Neuron and make associated memebers protected/private
namespace sb_nn{
    enum class ClassifierType : int{
        LINEAR = 1,
        NONLINEAR = 2,
        MULTICLASS = 3
    };
    template <typename T>
    class NeuralNetClassifier{
        typedef std::vector<Neuron<T>> neuron_layer;
        public:
            NeuralNetClassifier<T>(int num_epochs, double learning_rate, ClassifierType network_type);
            const bool train_network();   //This uses the previously gathered input and output data sets to train the weights/bias
            const bool set_training_data(   std::vector<std::vector<T>> &input_training_data,
                                            std::vector<T> &output_training_data);
            const T fire_network(std::vector<T> input){
                feed_forward(input);
                if (num_outputs_ > 1){
                    //TODO here
                    return -1;
                } else if (num_outputs_ == 1)
                    return output_layer_.at(0).activation_energy_;
                else
                    return -1;
            }

        private:
            int num_features_;
            int num_outputs_;
            int num_hidden_neurons_;
            ClassifierType network_type_;
            int num_epochs_;
            double learning_rate_;
            std::vector<std::vector<T>> training_set_in_;
            std::vector<T> training_set_out_;
            neuron_layer hidden_layer_;
            neuron_layer output_layer_;
            std::vector<T> network_inputs_;     //inputs will simply be a vector of the specified input type
            const bool feed_forward(std::vector<T> data_in);
            const bool backpropagate(T desired_output);
            const bool adjust_network_weights(T expected_out);
            const bool createNetwork(   const int num_features, const int num_hidden_neurons, 
                                        const int num_outputs, const ActivationFunction fn);
            neuron_layer createLayer(const int num_inputs, const int num_neurons, const ActivationFunction output_fn);

    };

    template <class T>
    NeuralNetClassifier<T>::NeuralNetClassifier(int num_epochs, double learning_rate, ClassifierType network_type){
        num_epochs_ = num_epochs;
        learning_rate_ = learning_rate;
        network_type_ = network_type;
        ActivationFunction output_fn = ActivationFunction::SIGMOID;
        switch (network_type){
            case (ClassifierType::LINEAR): 
                std::cout << "linear, default inputs = 2, default outputs = 1" << std::endl;
                //num inputs is two (can be modified), num outputs is one, no hidden layers, all activation functions are sigmoid
                num_features_ = 2;
                num_outputs_ = 1;
                num_hidden_neurons_ = 0;
                break;
            case (ClassifierType::NONLINEAR) : 
                std::cout << "nonlinear, default inputs = 2, default outputs = 1, nodes in hidden layer = 4" << std::endl;
                //nonlinear, num inputs is still two (can be modified), num outputs is still one, hidden layers exist
                num_features_ = 2;
                num_outputs_ = 1;
                num_hidden_neurons_ = 4;
                break;
            case (ClassifierType::MULTICLASS) : 
                std::cout << "multi-class, default inputs 2, default outputs = 3, nodes in hidden layer = 4" << std::endl;
                //multi class, num inputs defaults to two (can be modified), num outputs is variable, hidden layers exist
                num_features_ = 2;
                num_outputs_ = 3;
                num_hidden_neurons_ = 4;
                output_fn = ActivationFunction::SOFTMAX;
                break;
            default:
                std::cout << "No network found for type specified" << std::endl;
                num_features_ = -1;
                num_outputs_ = -1;
                num_hidden_neurons_ = -1;
                break;
        }
        if (createNetwork(num_features_, num_hidden_neurons_, num_outputs_, output_fn))
            std::cout << "Network created successfully" << std::endl;
        else
            std::cerr << "Error creating network" << std::endl;
    }

    template <typename T>
    const bool NeuralNetClassifier<T>::createNetwork(   const int num_features, const int num_hidden_neurons, 
                                                        const int num_outputs, const ActivationFunction output_fn){
        assert(num_hidden_neurons >= 0 && "Num hidden neurons must be positive!");
        if (num_hidden_neurons == 0){
            //simple linear classifier, weights to outputs == num inputs, output function sigqmoid
            output_layer_ = createLayer(num_features, num_outputs, ActivationFunction::SIGMOID);
        } else {
            //there exists a hidden layer, lets first create the number of hidden neurons, propagate activation
            //down to output function
            hidden_layer_ = createLayer(num_features, num_hidden_neurons, ActivationFunction::SIGMOID);
            output_layer_ = createLayer(num_hidden_neurons, num_outputs, output_fn);
        }
        return true;
    }

    template <typename T>
    typename NeuralNetClassifier<T>::neuron_layer NeuralNetClassifier<T>::createLayer(const int num_inputs, const int num_neurons, const ActivationFunction fn){
        assert(num_inputs >= 2 && "Num inputs must be at least 2!");
        assert(num_neurons >= 1 && "Num neurons in layer must be at least 1!");
        neuron_layer layer;
        for (unsigned int i=  0; i < num_neurons; ++i){
            double rand_bias = (double) rand()/(RAND_MAX);
            Neuron<T> neuron(rand_bias, fn);
            for (unsigned int j = 0; j < num_inputs; ++j){
                double rand_weight = (double) rand()/(RAND_MAX);
                neuron.add_neuron_input(NeuronInput<T>{rand_weight});
            }
            layer.push_back(neuron);
        }
        return layer;
    }

    template <typename T>
    const bool NeuralNetClassifier<T>::set_training_data(  std::vector<std::vector<T>> &input_training_data,
                                                    std::vector<T> &output_training_data){
        //check to make sure the dimensionality matches the amount of input lines
        assert(input_training_data.size() == output_training_data.size() && "Dimensionality of input data must match dimensionality of the output data!");
        for (std::vector<T> in : input_training_data){
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

    template <typename T>
    const bool NeuralNetClassifier<T>::train_network(){
        //should check dimensionality of input data to ensure it is the proper dimensions...
        std::cout << "Training network...." << std::endl;
        //assert(training_set_in_.size() == training_set_out_.size() && "Must have same number of input and output training data sets!");
        if (training_set_in_.size() <= 0 || training_set_out_.size() <= 0){
            std::cout << "No training data supplied!" << std::endl;
            return false;
        }
        assert(training_set_in_.size() == training_set_out_.size());
        if (hidden_layer_.size() <= 0){
            //no hidden layer, add input equal to num features for the output neuron
            //double rand_weight = 2;
            
        }
        for (unsigned int i= 0; i < num_epochs_; ++i){
            for (size_t j = 0; j < training_set_in_.size(); ++j){
                feed_forward(training_set_in_.at(j));
                backpropagate(training_set_out_.at(j));
                //double error = 1/2 * ((output_neuron_.activation_energy_ - training_set_out_.at(j))*(output_neuron_.activation_energy_ - training_set_out_.at(j)));
            }
        }
        return true;
    } 

    template <typename T>
    const bool NeuralNetClassifier<T>::feed_forward(std::vector<T> network_inputs){ 
        //TODO refactor to put this somewhere else
        assert( network_inputs.size() == num_features_ && 
                "Input training data must match number of network inputs!");
        bool retval = false;
        std::vector<T> hidden_outputs;
        switch (network_type_){
            case(ClassifierType::LINEAR):
                //No hidden neurons, so set output neuron inputs to be the network inputs
                output_layer_.at(0).set_neuron_values(network_inputs);
                output_layer_.at(0).activate();
                retval = true;
                break;
            case(ClassifierType::NONLINEAR):
                //activate hidden neurons and propagate to output layer
                for (auto &hidden_neuron : hidden_layer_){
                    //Now we set the inputs for each hidden neuron
                    hidden_neuron.set_neuron_values(network_inputs);
                    //Now the hidden neurons have weights and values, should fire them.
                    hidden_neuron.activate();
                    hidden_outputs.push_back(hidden_neuron.activation_energy_);
                }
                //Since it is nonlinear, there will only be 1 output neuron, set the hidden outputs to be inputs
                output_layer_.at(0).set_neuron_values(hidden_outputs);
                output_layer_.at(0).activate();
                retval = true;
                break;
            case(ClassifierType::MULTICLASS):
                //activate hidden neurons and propagate to output layer
                for (auto &hidden_neuron : hidden_layer_){
                    //Now we set the inputs for each hidden neuron
                    hidden_neuron.set_neuron_values(network_inputs);
                    //Now the hidden neurons have weights and values, should fire them.
                    hidden_neuron.activate();
                    hidden_outputs.push_back(hidden_neuron.activation_energy_);
                }
                //now since we have more than one output neuron, we will need to add the hidden outputs as inputs to each one
                for (auto &output_neuron : output_layer_){
                    output_neuron.set_neuron_values(hidden_outputs);
                    //TODO: find best way to do this with softmax since softmax function takes in a vector of output neurons
                    output_neuron.activate();
                }
                retval = true;
                break;
            default:
                std::cerr << "Cannot feed forward unknown network type!" << std::endl;
                retval = false;
                break;
        }
        return retval;
    }

    template <typename T>
    const bool NeuralNetClassifier<T>::backpropagate(T desired_output){
        //for now we'll start with one hidden neuron
        //Calculate the output error relative to the output of the activation functions
        if (hidden_layer_.size() > 0){
            //TODO: backpropagate
            adjust_network_weights(desired_output);
        } else if (hidden_layer_.size() <= 0){   
            //TODO: Test this functionality, and also test to handle uninitialized hidden_layer_ values.    
            T error = output_layer_.at(0).activation_energy_ - desired_output;
            T dcost_dpred = error;
            T dpred_dz = output_layer_.at(0).sigmoid_d1(output_layer_.at(0).activation_energy_);
            T dcost_dz = dcost_dpred * dpred_dz;
            //Now we've got the desired rate of change, lets apply it to our weights and biases
            //now apply that calculated change to the weights and bias
            for (auto &input : output_layer_.at(0).neuron_inputs_){
                input.weight -= learning_rate_ * *input.value * dcost_dz;
            }
            output_layer_.at(0).bias_ -= learning_rate_ * dcost_dz;
        }

        return true;
    }

    template <typename T>
    const bool NeuralNetClassifier<T>::adjust_network_weights(T expected_out){
        assert(output_layer_.at(0).neuron_inputs_.size() == hidden_layer_.size());
        for (unsigned int i = 0; i < output_layer_.at(0).neuron_inputs_.size(); ++i){
            
            //first adjust output weights
            Neuron<T> hidden_neuron = hidden_layer_.at(i);
            T dzOutput_dweight = *output_layer_.at(0).neuron_inputs_.at(i).value;
            T dactivationOut_dzOutput = hidden_neuron.sigmoid_d1(output_layer_.at(0).output_energy_);    //double check this
            T dcost_dactivationOut = (output_layer_.at(0).activation_energy_ - expected_out);  
            T dcost_dweightOut = dcost_dactivationOut * dactivationOut_dzOutput * dzOutput_dweight;        //get gradient via chain rule
            //This has been mathematically checked
            //now lets adjust the input->hidden weights
            for (auto &hidden_input : hidden_neuron.neuron_inputs_){
                if (hidden_input.value == std::nullopt){
                    std::cerr << "Cannot change weight of hidden input with no value!\n";
                    return false;
                }
                T dcost_dactivationHid = dcost_dactivationOut * dactivationOut_dzOutput * hidden_input.weight;
                T dzHid_dweightHid = *hidden_input.value;
                T dactivationHid_dzHid = hidden_neuron.sigmoid_d1(hidden_neuron.output_energy_);
                T dcost_dweightHid = dactivationHid_dzHid * dzHid_dweightHid * dcost_dactivationHid;
                hidden_input.weight -= dcost_dweightHid * learning_rate_;
            }
            output_layer_.at(0).neuron_inputs_.at(i).weight -= dcost_dweightOut * learning_rate_;
            
        }
        return true;
    }
}
#endif