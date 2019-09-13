#include <Neuron.hpp>

namespace sb_nn{
    template <typename I, typename O>
    class NeuralNet{
        public:
            NeuralNet(const int num_epochs, const double learning_rate) : num_epochs_(num_epochs), learning_rate_(learning_rate) {};
            const bool set_feature_num(const int feat_num) { num_features_ = feat_num;};  
            const bool add_hidden_neuron(Neuron neuron);    //in the future add the ability to specify which layer
            const bool train();   //This uses the previously gathered input and output data sets to train the weights/bias
            const bool add_training_data(   std::vector<std::vector<I>> input_training_data,
                                            std::vector<O> output_training_data);
            const double feed_forward();
        private:
            const int num_features_;
            std::vector<Neuron<I, O>> hidden_neurons_;
            Neuron<I, O> output_neuron_;
            const int num_epochs_;
            const double learning_rate_;
            std::vector<std::vector<I>> training_set_in_;
            std::vector<O> training_set_out_;
            const double backpropagate(O desired_output, double activation_out);
    };
}