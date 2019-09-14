#include <Neuron.hpp>

namespace sb_nn{
    template <typename T>
    class NeuronLayer{
        private:
            std::vector<Neuron<T>> neurons_;
    };
}
