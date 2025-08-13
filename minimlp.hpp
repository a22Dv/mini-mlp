#pragma once

// #define MINIMLP_IMPLEMENTATION // DEVELOPMENT

#ifdef MINIMLP_IMPLEMENTATION
#endif

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlp {

using f32_t = float;
using u32_t = std::uint32_t;
using u8_t = std::uint8_t;

/**
    TODO: Finish Tensor API. Currently at the at() methods.
*/

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> class Tensor {
  private:
    std::vector<u32_t> _dimensions{};
    std::vector<u32_t> _strideUnits{};
    std::vector<T> _data{};

    template <typename... Indices, std::size_t... Dims> T &at(std::index_sequence<Dims...>, Indices... indices) {
        std::size_t idx{((indices * _strideUnits[Dims]) + ...)};
        return _data[idx];
    }

  public:
    Tensor();
    Tensor(std::vector<u32_t> dims) : _dimensions{dims} {};
    Tensor(std::vector<T> initializer, std::vector<u32_t> dims) : _data{initializer}, _dimensions{dims} {
        std::size_t dimsN{_dimensions.size()};
        _strideUnits.resize(dimsN);
        if (dimsN == 0) {
            throw std::runtime_error("Invalid argument. Tensor with a dimension count of 0 is not allowed.");
        }
        _strideUnits[dimsN - 1] = 1;
        for (std::size_t i{dimsN - 1}; i-- > 0;) {
            _strideUnits[i] = _strideUnits[i + 1] * _dimensions[i + 1];
        }
    };
    std::vector<T> &data() { return _data; };
    const std::vector<T> &data() const { return _data; };
    const std::vector<u32_t> &dimensions() const { return _dimensions; };

    template <typename... Indices>
    auto at(Indices... indices) const
        -> std::enable_if_t<((std::is_unsigned_v<Indices> && std::is_integral_v<Indices>) && ...), const T &> {
        return at(std::make_index_sequence<sizeof...(Indices)>(), indices...);
    }
    template <typename... Indices>
    auto at(Indices... indices)
        -> std::enable_if_t<((std::is_unsigned_v<Indices> && std::is_integral_v<Indices>) && ...), T &> {
        return at(std::make_index_sequence<sizeof...(Indices)>(), indices...);
    }

    const T &operator[](const std::size_t idx) const { return _data[idx]; }
    T &operator[](const std::size_t idx) { return _data[idx]; }

    Tensor<T> operator+(const Tensor<T> &rhs) {
        if (_dimensions != rhs.dimensions()) {
            throw std::runtime_error("Invalid arguments. Arguments' dimensions are not consistent.");
        }
        Tensor<T> out{_dimensions};
        std::size_t iters{_data.size()};
        for (std::size_t i{0}; i < iters; ++i) {
        }
    }
    Tensor<T> operator+(const T scalar) {
        Tensor<T> out{};
        for (T val : _data) {
            val += scalar;
        }
    }
    Tensor<T> operator-(const Tensor<T> &rhs);
    Tensor<T> operator-(const T scalar) {
        for (T val : _data) {
            val -= scalar;
        }
    }
    Tensor<T> operator*(const Tensor<T> &rhs);
    Tensor<T> operator*(const T scalar) {
        for (T val : _data) {
            val *= scalar;
        }
    }
    Tensor<T> operator/(const Tensor<T> &rhs);
    Tensor<T> operator/(const T scalar) {
        for (T val : _data) {
            val /= scalar;
        }
    }
    void hadamard(const Tensor<T> &rhs) {}
    void transpose();
};

/**
    Everything below this comment is a `TODO:` and is subject to change.
 */

enum class ModelLossFunction_ : u8_t {
    CROSS_ENTROPY,
    MEAN_SQUARED,
};

enum class ModelActivationFunction_ : u8_t {
    TANH,
    SIGMOID,
    RELU,
    LEAKY_RELU,
    SOFTMAX,
};

enum class ModelOptimizationFunction_ : u8_t {
    ADAM,
    GRADIENT_DESCENT,
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> struct ModelActivationFunction {
    ModelActivationFunction_ type{};
    Tensor<T> out(const Tensor<T> &logits);
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> struct ModelLossFunction {
    ModelLossFunction_ type{};
    f32_t out(const Tensor<T> &modelOut, const Tensor<T> &expected);
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> struct Layer {
    ModelActivationFunction<T> actFunc{};
    Tensor<T> weights{};
    Tensor<T> biases{};
    std::optional<Tensor<T>> intermediate{};
    Tensor<T> out(const Tensor<T> &input);
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> struct ModelOptimizationFunction {
    ModelOptimizationFunction_ type{};
    /// TODO: Define optimization dispatcher function.
    void opt(const std::vector<Layer<T>> &targets);
};

struct ModelResult {
    u32_t returnValue{};
    f32_t confidence{};
};

struct ModelParams {
    u32_t inputSize{};
    u32_t outputCount{};
    std::vector<u32_t> hLayerCounts{};
    std::vector<ModelActivationFunction_> activations{};
    ModelOptimizationFunction_ optimization{ModelOptimizationFunction_::GRADIENT_DESCENT};
};

class Model {
  private:
    const ModelParams params{};
    void forward();
    void backward();

  public:
    void train();
    void save(std::filesystem::path savePath);
    ModelResult infer();
    Model(ModelParams params);
    Model(std::filesystem::path modelPath);
};

#ifdef MINIMLP_IMPLEMENTATION

#endif

} // namespace mlp