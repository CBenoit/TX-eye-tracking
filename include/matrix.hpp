#pragma once

#include <opencv2/core/mat.hpp>
#include <type_traits>

template <typename T>
class matrix : public cv::Mat {

public:

	matrix() : cv::Mat{} {}
	matrix(int rows, int cols, int type) : cv::Mat(rows, cols, type) {}
	matrix(cv::Size size, int type) : cv::Mat(size, type) {}
	matrix(int rows, int cols, int type, const cv::Scalar& s) : cv::Mat(rows, cols, type, s) {}
	matrix(cv::Size size, int type, const cv::Scalar& s) : cv::Mat(size, type, s) {}
	matrix(int ndims, const int* sizes, int type) : cv::Mat(ndims, sizes, type) {}
	matrix(const std::vector<int>& sizes, int type) : cv::Mat(sizes, type) {}
	matrix(int ndims, const int* sizes, int type, const cv::Scalar& s) : cv::Mat(ndims, sizes, type, s) {}
	matrix(const std::vector<int>& sizes, int type, const cv::Scalar& s) : cv::Mat(sizes, type, s) {}
	matrix(int rows, int cols, int type, T* data, size_t step=AUTO_STEP) : cv::Mat(rows, cols, type, data, step) {}
	matrix(cv::Size size, int type, T* data, size_t step=AUTO_STEP) : cv::Mat(size, type, data, step) {}
	matrix(int ndims, const int* sizes, int type, T* data, const size_t* steps = nullptr) : cv::Mat(ndims, sizes, type, data, steps) {}
	matrix(const std::vector<int>& sizes, int type, T* data, const size_t* steps = nullptr) : cv::Mat(sizes, type, data, steps) {}
	matrix(const cv::Mat& m, const cv::Range& rowRange, const cv::Range& colRange=cv::Range::all()) : cv::Mat(m, rowRange, colRange) {}
	matrix(const cv::Mat& m, const cv::Rect& roi) : cv::Mat(m, roi) {}
	matrix(const cv::Mat& m, const cv::Range* ranges) : cv::Mat(m, ranges) {}
	matrix(const cv::Mat& m, const std::vector<cv::Range>& ranges) : cv::Mat(m, ranges) {}
	explicit matrix(const std::vector<T>& vec, bool copyData = false) : cv::Mat(vec, copyData) {}

	explicit matrix(const cv::Mat& m) : cv::Mat(m) {}
	matrix(const matrix<T>& m) : cv::Mat(m) {}

	matrix<T> operator() (const cv::Rect& c) const {
		return matrix<T>(cv::Mat::operator()(c));
	}

	T& operator() (unsigned int x, unsigned int y) {
		return at<T>(y,x);
	}

	const T& operator() (unsigned int x, unsigned int y) const {
		return at<T>(y,x);
	}

	T& operator() (const cv::Point& p) {
		return at<T>(p.y, p.x);
	}

	const T& operator() (const cv::Point& p) const {
		return at<T>(p.y, p.x);
	}

	matrix<T> transpose() const {
		return matrix<T>{t()};
	}

	template <typename output>
	matrix<output> compute_x_gradient() {
		matrix<output> out(height(), width(), CV_64F);

		for (auto y = 0u ; y < height() ; ++y) {

			out(0, y) = output(2) * ((*this)(1, y) - (*this)(0, y));
			for (auto x = 1u ; x < width() - 1 ; ++x) {
				out(x, y) = (*this)(x + 1, y) - (*this)(x - 1, y);
			}
			out(width() - 1, y) = output(2) * ((*this)(width() - 1, y) - (*this)(width() - 2, y));
		}
		return out;
	}

	template <typename output>
	matrix<output> compute_y_gradient() {
		matrix<output> out(height(), width(), CV_64F);

		for (auto x = 0u ; x < width() ; ++x) {

			out(x, 0) = output(2) * ((*this)(x, 1) - (*this)(x, 0));
			for (auto y = 1u ; y < height() - 1 ; ++y) {
				out(x, y) = (*this)(x, y + 1) - (*this)(x, y - 1);
			}
			out(x, height() - 1) = output(2) * ((*this)(x, height() - 1) - (*this)(x, height() - 2));
		}
		return out;
	}

	unsigned int height() const {
		return static_cast<unsigned int>(rows);
	}

	unsigned int width() const {
		return static_cast<unsigned int>(cols);
	}
};