#pragma once
// #define MINIMLP_IMPLEMENTATION

#ifdef MINIMLP_IMPLEMENTATION
#include <iostream>
#endif

#include <cstdint>
#include <filesystem>
#include <vector>

namespace mlp {

using f32_t = float;
using u32_t = std::uint32_t;
using u8_t = std::uint8_t;

enum class ModelLossFunction : u8_t {
    CROSS_ENTROPY,
    MEAN_SQUARED,
};

enum class ModelActivationFunction : u8_t {
    TANH,
    SIGMOID,
    RELU,
    LEAKY_RELU,
    SOFTMAX,
};

enum class ModelOptimizationFunction : u8_t {
    ADAM,
    GRADIENT_DESCENT,
};

template <typename T> class Matrix {
  private:
    std::vector<u32_t> _dimensions{};
    std::vector<T> _data{};

  public:
    Matrix();
    Matrix(std::vector<T> initializer);
    std::vector<T> &data() { return _data; };
    const std::vector<T> &cData() { return _data; };
    const std::vector<u32_t> &dimensions() const { return _dimensions; };
    Matrix<T> operator+(const Matrix<T> &rhs);
    Matrix<T> operator-(const Matrix<T> &rhs);
    Matrix<T> operator*(const Matrix<T> &rhs);
    Matrix<T> operator/(const Matrix<T> &rhs);
    void hadamard(const Matrix<T> &rhs);
    void transpose();
};

struct ModelResult {
    u32_t returnValue{};
    f32_t confidence{};
};

struct ModelParams {
    u32_t inputSize{};
    u32_t outputCount{};
    f32_t learningRate{};
    std::vector<u32_t> layerCounts{};
    std::vector<ModelActivationFunction> activations{};
    ModelOptimizationFunction optimization{ModelOptimizationFunction::GRADIENT_DESCENT};
};

class Model {
  private:
    const ModelParams params{};
    void forward();
    void backward();

  public:
    void train();
    ModelResult infer();
    Model(ModelParams params);
    Model(std::filesystem::path modelPath);
};

#ifdef MINIMLP_IMPLEMENTATION

#endif

} // namespace mlp