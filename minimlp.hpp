#pragma once

// #define MINIMLP_IMPLEMENTATION // DEVELOPMENT
#ifdef MINIMLP_IMPLEMENTATION
#endif

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>
#include <stdexcept>

namespace mlp {

using f32_t = float;
using u32_t = std::uint32_t;
using u8_t = std::uint8_t;

/**
    TODO: Finish Matrix API. Currently at the at() methods.
*/
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> class Matrix {
  private:
    std::vector<u32_t> _dimensions{};
    std::vector<T> _data{};

  public:
    Matrix();
    Matrix(std::vector<T> initializer, std::vector<u32_t> dims) : _data{initializer}, _dimensions{dims} {};
    std::vector<T> &data() { return _data; };
    const T& atC(std::size_t ...) const {

    }
    T& at(std::size_t ...) {

    }
    const std::vector<T> &cData() const { return _data; };
    const std::vector<u32_t> &dimensions() const { return _dimensions; };
    T& operator[](const std::size_t idx) {
        return _data[idx];
    }
    Matrix<T> operator+(const Matrix<T> &rhs) {
        if (_dimensions != rhs.dimensions()) {
            throw std::runtime_error("Invalid arguments. Arguments' dimensions are not consistent.");
        }
        std::size_t iters{_data.size()};
        Matrix<T> out{};
        for (std::size_t i{0}; i < iters; ++i) {
            
        }
    }
    Matrix<T> operator+(const T scalar) {
        for (T val : _data) {
            val += scalar;
        }
    }
    Matrix<T> operator-(const Matrix<T> &rhs);
    Matrix<T> operator-(const T scalar) {
        for (T val : _data) {
            val -= scalar;
        }
    }
    Matrix<T> operator*(const Matrix<T> &rhs);
    Matrix<T> operator*(const T scalar) {
        for (T val : _data) {
            val *= scalar;
        }
    }
    Matrix<T> operator/(const Matrix<T> &rhs);
    Matrix<T> operator/(const T scalar) {
        for (T val : _data) {
            val /= scalar;
        }
    }
    void hadamard(const Matrix<T> &rhs) {

    }
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
    Matrix<T> out(const Matrix<T> &logits);
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> struct ModelLossFunction {
    ModelLossFunction_ type{};
    f32_t out(const Matrix<T> &modelOut, const Matrix<T> &expected);
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> struct Layer {
    ModelActivationFunction<T> actFunc{};
    Matrix<T> weights{};
    Matrix<T> biases{};
    std::optional<Matrix<T>> intermediate{};
    Matrix<T> out(const Matrix<T> &input);
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