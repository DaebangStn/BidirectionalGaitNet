#ifndef INCREMENTAL_LEAST_SQUARES_H
#define INCREMENTAL_LEAST_SQUARES_H

#include <deque>
#include <utility>
#include <cmath>

/**
 * @brief Incremental Least Squares Regression for sliding window
 *
 * Efficiently computes linear regression slope over a moving window of data points.
 * Uses O(1) updates by maintaining running sums: S_x, S_y, S_xx, S_xy
 *
 * When a new point (x_new, y_new) arrives and the oldest point (x_old, y_old)
 * is removed, the sums are updated incrementally without full recomputation.
 *
 * Slope formula: b = (W*S_xy - S_x*S_y) / (W*S_xx - S_x^2)
 * where W is the window size (number of points in buffer).
 */
class IncrementalLeastSquares {
public:
    /**
     * @brief Constructor
     * @param window_size Maximum number of points to keep in the sliding window
     */
    IncrementalLeastSquares(int window_size = 1000)
        : W(window_size), S_x(0.0), S_y(0.0), S_xx(0.0), S_xy(0.0) {}

    /**
     * @brief Add a new data point and update regression incrementally
     * @param x_new X-coordinate of new point (e.g., time)
     * @param y_new Y-coordinate of new point (e.g., position)
     */
    void update(double x_new, double y_new) {
        // Add new point to buffer and sums
        buffer.push_back({x_new, y_new});
        S_x += x_new;
        S_y += y_new;
        S_xx += x_new * x_new;
        S_xy += x_new * y_new;

        // Remove oldest point if buffer exceeds window size
        if (static_cast<int>(buffer.size()) > W) {
            auto [x_old, y_old] = buffer.front();
            buffer.pop_front();
            S_x -= x_old;
            S_y -= y_old;
            S_xx -= x_old * x_old;
            S_xy -= x_old * y_old;
        }
    }

    /**
     * @brief Get the current regression slope (velocity if x=time, y=position)
     * @return Slope of the least squares line, or 0.0 if insufficient data
     */
    double getSlope() const {
        int n = buffer.size();
        if (n < 2) return 0.0;  // Need at least 2 points for regression

        double denom = n * S_xx - S_x * S_x;
        if (std::abs(denom) < 1e-10) return 0.0;  // Avoid division by zero

        return (n * S_xy - S_x * S_y) / denom;
    }

    /**
     * @brief Get the current intercept of the regression line
     * @return Intercept of the least squares line
     */
    double getIntercept() const {
        int n = buffer.size();
        if (n < 2) return 0.0;

        double slope = getSlope();
        return (S_y - slope * S_x) / n;
    }

    /**
     * @brief Reset all data and clear the buffer
     */
    void reset() {
        buffer.clear();
        S_x = 0.0;
        S_y = 0.0;
        S_xx = 0.0;
        S_xy = 0.0;
    }

    /**
     * @brief Get the number of points currently in the buffer
     * @return Current buffer size
     */
    int size() const {
        return buffer.size();
    }

private:
    int W;                                          // Window size
    double S_x, S_y, S_xx, S_xy;                   // Running sums
    std::deque<std::pair<double, double>> buffer;  // Circular buffer of (x, y) pairs
};

#endif // INCREMENTAL_LEAST_SQUARES_H
