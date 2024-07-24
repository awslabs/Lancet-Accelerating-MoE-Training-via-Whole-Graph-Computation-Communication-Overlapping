#pragma once

#include <cassert>
#include <vector>
#include <algorithm>

namespace raf {
namespace pass {
namespace lerp {

template <typename T>
class LinearInterpolator {
public:
    LinearInterpolator() {}
    LinearInterpolator(const std::vector<T>& x, const std::vector<T>& y, bool pos=true): pos_(pos) {
        // sort the x y pairs and remove duplicates
        std::vector<std::pair<T, T>> xy;
        for (int i = 0; i < x.size(); ++i) {
            xy.push_back(std::make_pair(x[i], y[i]));
        }
        std::sort(xy.begin(), xy.end(), [](const std::pair<T, T>& a, const std::pair<T, T>& b) {
            return a.first < b.first;
        });
        for (int i = 0; i < xy.size(); ++i) {
            if (i > 0 && (xy[i].first - xy[i - 1].first < 1e-6)) {
                continue;
            }
            x_.push_back(xy[i].first);
            y_.push_back(xy[i].second);
        }
        assert(x.size() == y.size());
        m_.resize(x.size());
        for (int i = 0; i < x.size() - 1; ++i) {
            m_[i] = (y_[i + 1] - y_[i]) / (x_[i + 1] - x_[i]);
        }
    }

    T operator()(const T& x) {
        assert (x_.size() > 0);
        int i = 0;
        while (x_[i] < x) {
            ++i;
        }
        if (i > x_.size() - 2) {
            i = x_.size() - 2;
        }
        auto result = y_[i] + m_[i] * (x - x_[i]);
        if (pos_) {
            return std::max(result, static_cast<T>(0));
        } else {
            return result;
        }
    }

private:
    std::vector<T> x_;
    std::vector<T> y_;
    std::vector<T> m_;
    bool pos_;
};

} // namespace lerp
} // namespace pass
} // namespace raf