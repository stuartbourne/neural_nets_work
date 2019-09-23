#include <Neuron.hpp>

namespace sb_nn{
    template <typename T>
    class NeuronLayer{
        public:
            //will move activation into layer as opposed to the Neuron class
            NeuronLayer(const int num_features, const int num_neurons, const ActivationFunction output_fn);
            const bool activateLayer();
        private:
            std::vector<Neuron<T>> neurons_;

    };
}
