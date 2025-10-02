#ifndef __CBUFFER_DATA_H__
#define __CBUFFER_DATA_H__

#include <boost/circular_buffer.hpp>
#include <map>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <numeric>
#include <stdexcept>

template <typename T>
class CBufferData
{
public:
    CBufferData() = default;
    ~CBufferData() = default;

    // Register a new key with specified buffer size
    void register_key(const std::string& key, size_t buffer_size)
    {
        if (mBuffers.find(key) == mBuffers.end())
        {
            mBuffers[key] = boost::circular_buffer<T>(buffer_size);
        }
    }

    // Check if key exists
    bool key_exists(const std::string& key) const
    {
        return mBuffers.find(key) != mBuffers.end();
    }

    // Push value to buffer
    void push(const std::string& key, const T& value)
    {
        if (!key_exists(key))
        {
            throw std::runtime_error("CBufferData: Key '" + key + "' not registered");
        }
        mBuffers[key].push_back(value);
    }

    // Get entire buffer for a key
    std::vector<T> get(const std::string& key) const
    {
        if (!key_exists(key))
        {
            return std::vector<T>();
        }
        const auto& buffer = mBuffers.at(key);
        return std::vector<T>(buffer.begin(), buffer.end());
    }

    // Get range of values [start, end)
    std::vector<T> get_range(const std::string& key, size_t start, size_t end) const
    {
        if (!key_exists(key))
        {
            return std::vector<T>();
        }
        const auto& buffer = mBuffers.at(key);
        if (start >= buffer.size())
        {
            return std::vector<T>();
        }
        end = std::min(end, buffer.size());
        return std::vector<T>(buffer.begin() + start, buffer.begin() + end);
    }

    // Get buffer size
    size_t size(const std::string& key) const
    {
        if (!key_exists(key))
        {
            return 0;
        }
        return mBuffers.at(key).size();
    }

    // Get capacity
    size_t capacity(const std::string& key) const
    {
        if (!key_exists(key))
        {
            return 0;
        }
        return mBuffers.at(key).capacity();
    }

    // Clear buffer
    void clear(const std::string& key)
    {
        if (key_exists(key))
        {
            mBuffers[key].clear();
        }
    }

    // Clear all buffers
    void clear_all()
    {
        for (auto& pair : mBuffers)
        {
            pair.second.clear();
        }
    }

    // Get all registered keys
    std::vector<std::string> get_keys() const
    {
        std::vector<std::string> keys;
        for (const auto& pair : mBuffers)
        {
            keys.push_back(pair.first);
        }
        return keys;
    }

    // Get latest value
    T get_latest(const std::string& key) const
    {
        if (!key_exists(key))
        {
            throw std::runtime_error("CBufferData: Key '" + key + "' not registered");
        }
        const auto& buffer = mBuffers.at(key);
        if (buffer.empty())
        {
            throw std::runtime_error("CBufferData: Buffer for key '" + key + "' is empty");
        }
        return buffer.back();
    }

    // Get moving average (for double type)
    template <typename U = T>
    typename std::enable_if<std::is_same<U, double>::value, double>::type
    get_moving_average(const std::string& key, size_t window_size) const
    {
        if (!key_exists(key))
        {
            return 0.0;
        }
        const auto& buffer = mBuffers.at(key);
        if (buffer.empty())
        {
            return 0.0;
        }

        size_t count = std::min(window_size, buffer.size());
        double sum = 0.0;
        for (size_t i = buffer.size() - count; i < buffer.size(); ++i)
        {
            sum += buffer[i];
        }
        return sum / count;
    }

private:
    std::map<std::string, boost::circular_buffer<T>> mBuffers;
};

#endif
