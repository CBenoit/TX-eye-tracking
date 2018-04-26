#pragma once

#include "config.hpp"

// Resizes a matrix for faster computing time
matrix<unsigned char> rescale(const matrix<unsigned char>& original);

// Undo the op for one point
cv::Point unsale_pt(cv::Point p, const cv::Rect& original_size);

template <typename T>
constexpr bool is_in(cv::Point p, const matrix<T>& mat);

matrix<double> magnitude(const matrix<double> &x_mag, const matrix<double> &y_mag);

double compute_threshold(const matrix<double>& magnitude, double stdDevFactor);


struct less_cmp {
	bool operator()(const std::pair<cv::Point, float>& lhs, const std::pair<cv::Point, float>& rhs) {
		return lhs.first.x != rhs.first.x ? lhs.first.x < rhs.first.x : lhs.first.y < rhs.first.y;
	}
};






inline matrix<unsigned char> rescale(const matrix<unsigned char>& original) {
	matrix<unsigned char> result;
	cv::resize(original, result, cv::Size(static_cast<int>(config::eye_rescale_width)
			, static_cast<int>(original.height() * config::eye_rescale_width / original.width())));
	return result;
}

inline cv::Point unscale_pt(cv::Point p, const cv::Rect& original_size) {
	float ratio = config::eye_rescale_width / original_size.width;
	return {static_cast<int>(std::lround(p.x / ratio)), static_cast<int>(std::lround(p.y / ratio))};
}

template <typename T>
constexpr bool is_in(cv::Point p, const matrix<T>& mat) {
	return p.x >= 0 && p.x < mat.width() && p.y >= 0 && p.y < mat.height();
}

inline double compute_threshold(const matrix<double>& magnitude, double stdDevFactor) {
	cv::Scalar mean_grandient_norm, std_dev;
	cv::meanStdDev(magnitude, mean_grandient_norm, std_dev);

	double normalized_std_dev = std_dev[0] / std::sqrt(magnitude.rows * magnitude.cols);
	return stdDevFactor * normalized_std_dev + mean_grandient_norm[0];
}


inline matrix<double> magnitude(const matrix<double>& x_mag, const matrix<double>& y_mag) {
	assert(x_mag.height() == y_mag.height() && x_mag.width() == y_mag.width());

	matrix<double> mags(x_mag.height(), x_mag.width(), CV_64F);
	for (auto x = 0u ; x < x_mag.width() ; ++x) {
		for (auto y = 0u ; y < x_mag.height() ; ++y) {
			mags(x, y) = std::hypot(x_mag(x, y), y_mag(x, y));
		}
	}
	return mags;
}
