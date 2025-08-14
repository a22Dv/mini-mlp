#pragma once

// #define MINIMLP_IMPLEMENTATION // DEVELOPMENT
// #define MINIMLP_NO_REQUIRE // DEVELOPMENT
// #define MINIMLP_AT_NO_BOUNDS_CHECK // DEVELOPMENT
#include <cstddef>
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
using u8_t = std::uint8_t;
using u16_t = std::uint16_t;
using u32_t = std::uint32_t;
using u64_t = std::uint64_t;

/**
    TODO: Finish Tensor API. Test at() methods first and foremost.
*/

namespace {

enum class Error : u16_t {
    TENSOR_0_DIMENSIONS,
    TENSOR_MISMATCHED_DIMENSIONS,
    TENSOR_UNDEFINED_MULTIPLICATION,
    TENSOR_INVALID_DIMENSIONS
};
constexpr const char *getMessage(Error e) {
    return [&] {
        switch (e) {
        case Error::TENSOR_0_DIMENSIONS: return "Tensors with a dimension count of 0 is invalid.";
        case Error::TENSOR_MISMATCHED_DIMENSIONS: return "Tensor operation invalid because of mismatched dimensions.";
        case Error::TENSOR_UNDEFINED_MULTIPLICATION:
            return "Tensor operation invalid for (*) overload beyond 2 dimensions.";
        case Error::TENSOR_INVALID_DIMENSIONS: return "Invalid tensor dimensions.";
        }
    }();
}
#ifndef MINIMLP_NO_REQUIRE
inline void require(bool cond, Error e) {
    if (!cond) {
        throw std::runtime_error(getMessage(e));
    }
}
#else
inline void require(bool cond, Error e) { return; }
#endif

} // namespace

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> class Tensor {
  private:
    std::vector<u32_t> _dimensions{};
    std::vector<u32_t> _strideUnits{};
    std::vector<T> _data{};
    template <typename... Indices, std::size_t... Dims> T &at(std::index_sequence<Dims...>, Indices... indices) {
        std::size_t idx{((indices * _strideUnits[Dims]) + ...)};
        return _data[idx];
    }
    void getStride(const std::vector<u32_t> &dims) {
        std::size_t dimsN{dims.size()};
        require(dimsN == 0, Error::TENSOR_0_DIMENSIONS);
        _strideUnits.resize(dimsN);
        _strideUnits[dimsN - 1] = 1;
        for (std::size_t i{dimsN - 1}; i-- > 0;) {
            _strideUnits[i] = _strideUnits[i + 1] * dims[i + 1];
        }
    }
    Tensor<T> defaultOrder() {
        std::vector<u32_t> nlOrder(_data.size());
        for (std::size_t i{_data.size()}, j{}; i-- > 0; ++j) {
            nlOrder[j] = i;
        }
        return nlOrder;
    }

  public:
    Tensor() {};
    Tensor(std::vector<u32_t> &dims) : _dimensions{std::move(dims)} { getStride(dims); };
    Tensor(std::vector<T> &initializer, std::vector<u32_t> &dims)
        : _data{std::move(initializer)}, _dimensions{std::move(dims)} {
        getStride(dims);
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
    Tensor<T> operator+(const Tensor<T> &rhs) const {
        require(_dimensions == rhs._dimensions, Error::TENSOR_MISMATCHED_DIMENSIONS);
        Tensor<T> out{_dimensions};
        const std::size_t iters{_data.size()};
        for (std::size_t i{0}; i < iters; ++i) {
            out[i] = _data[i] + rhs[i];
        }
        return out;
    }
    Tensor<T> operator+(const T scalar) const {
        Tensor<T> out{_dimensions};
        const std::size_t iters{_data.size()};
        for (std::size_t i{0}; i < iters; ++i) {
            out[i] = _data[i] + scalar;
        }
        return out;
    }
    Tensor<T> operator-(const Tensor<T> &rhs) const {
        require(_dimensions == rhs._dimensions, Error::TENSOR_MISMATCHED_DIMENSIONS);
        Tensor<T> out{_dimensions};
        const std::size_t iters{_data.size()};
        for (std::size_t i{0}; i < iters; ++i) {
            out[i] = _data[i] - rhs[i];
        }
        return out;
    }
    Tensor<T> operator-(const T scalar) const {
        Tensor<T> out{_dimensions};
        const std::size_t iters{_data.size()};
        for (std::size_t i{0}; i < iters; ++i) {
            out[i] = _data[i] - scalar;
        }
        return out;
    }
    Tensor<T> operator*(const Tensor<T> &rhs) {
        require(_dimensions.size() < 3 && rhs._dimensions.size() < 3, Error::TENSOR_UNDEFINED_MULTIPLICATION);
        require(_dimensions.size() < 3 && rhs._dimensions.size() < 3, Error::TENSOR_MISMATCHED_DIMENSIONS);

        /// TODO: Matrix multiplication.
    }
    Tensor<T> operator*(const T scalar) {
        std::size_t iters{_data.size()};
        Tensor<T> out{_dimensions};
        for (std::size_t i{0}; i < iters; ++i) {
            out[i] = _data[i] * scalar;
        }
        return out;
    }
    Tensor<T> operator/(const Tensor<T> &rhs) {
        std::size_t iters{_data.size()};
        Tensor<T> out{_dimensions};
        for (std::size_t i{0}; i < iters; ++i) {
            out[i] = _data[i] / rhs[i];
        }
        return out;
    }
    Tensor<T> operator/(const T scalar) {
        std::size_t iters{_data.size()};
        Tensor<T> out{_dimensions};
        for (std::size_t i{0}; i < iters; ++i) {
            out[i] = _data[i] / scalar;
        }
        return out;
    }
    Tensor<T> hadamard(const Tensor<T> &rhs) {
        require(_dimensions == rhs._dimensions, Error::TENSOR_MISMATCHED_DIMENSIONS);
        Tensor<T> out{_dimensions};
        const std::size_t iters{_data.size()};
        for (std::size_t i{0}; i < iters; ++i) {
            out[i] = _data[i] * rhs[i];
        }
        return out;
    }
    void transpose(const std::vector<u32_t> &order = {}, const bool materialize = false) {
        const std::vector<u32_t> nOrder{order.empty() ? defaultOrder() : order};
        require(nOrder.size() == _dimensions.size(), Error::TENSOR_MISMATCHED_DIMENSIONS);
        const std::size_t orderSize{nOrder.size()};
        std::vector<bool> field(orderSize, false);
        for (const u32_t dim : nOrder) {
            require(dim < orderSize && !field[dim], Error::TENSOR_INVALID_DIMENSIONS);
            field[dim] = true;
        }
        std::vector<u32_t> nDims(_dimensions.size());
        std::vector<u32_t> nStrides(_strideUnits.size());
        for (std::size_t i{0}; i < orderSize; ++i) {
            nDims[i] = _dimensions[nOrder[i]];
            nStrides[i] = _strideUnits[nOrder[i]];
        }
        if (!materialize) {
            _dimensions = std::move(nDims);
            _strideUnits = std::move(nStrides);
            return;
        }
        /*
            Materialization. Physically moves the data buffer
            according to the new dimensions.
        */
        const std::vector<u32_t> oDims{_dimensions};
        const std::vector<u32_t> oStride{_strideUnits};
        const std::size_t dims{_dimensions.size()};
        const std::size_t iters{_data.size()};
        
        _dimensions = std::move(nDims);
        getStride(_dimensions);

        std::vector<u32_t> dimIdx(dims, 0);
        std::vector<u32_t> nDimIdx(dims, 0);
        std::vector<T> data(iters);
        for (std::size_t i{}; i < iters; ++i) {
            std::size_t oFIdx{};
            std::size_t nFIdx{};
            for (std::size_t j{0}; j < dims; ++j) {
                /*
                    The jth dimension now
                    stores dimIdx's value
                    taken according to the new order.
                */
                nDimIdx[j] = dimIdx[nOrder[j]];

                oFIdx += dimIdx[j] * oStride[j];
                nFIdx += nDimIdx[j] * _strideUnits[j];
            }
            data[nFIdx] = _data[oFIdx];
            for (std::size_t j{dims}; j-- > 0;) {
                dimIdx[j]++;
                if (dimIdx[j] < oDims[j]) {
                    break;
                }
                dimIdx[j] = 0;
            }
        }
        _data = std::move(data);
    }
    Tensor<T> transposeN(const std::vector<u32_t> &order = {}) const {
        Tensor<T> out{*this};
        out.transpose(order, true);
        return out;
    }
    Tensor<T> contract(const Tensor<T> &rhs, const std::vector<u32_t> &axesA, const std::vector<u32_t> &axesB) const {
        /// TODO: Need to brush up on theory. What even is an Einstein-sum.
    }
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