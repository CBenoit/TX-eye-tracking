#pragma once

#include <opencv2/core/types.hpp>
#include <cv.hpp>

#include <string_view>
#include <set>

#include "matrix.hpp"
#include "twin_matrixes.hpp"
#include "config.hpp"
#include "utils.hpp"

struct eye {
	cv::Rect eye_region;
	cv::Point eye_position;

private:
	cv::Point position_in_local;
	mutable unsigned int consecutive_match = 0;

	friend class eye_finder;
};

struct face {
	cv::Rect face_region;
	std::pair<eye, eye> eyes;
};

class eye_finder {

public:
	explicit eye_finder(const std::string& face_cascade_file) : face_classifier_{} {
		if (!face_classifier_.load(face_cascade_file)) {
			throw std::runtime_error("Failed to load " + face_cascade_file);
		}
	}

	std::optional<face> find_eyes(const matrix<unsigned char>& picture, const std::optional<face>& previous_face = {});

	void cast_centers_rays(cv::Point point, const matrix<unsigned char>& weights, twin_el<double> gradient,
	                       matrix<double>& out);

private:

	// Finds a face (does not detect eyes)
	std::optional<std::pair<matrix<unsigned char>, face>> find_face(const matrix<unsigned char>& picture);

	cv::Point eye_center(const matrix<unsigned char>& face, const cv::Rect& eye_region);

	matrix<float> compute_gradient_matrix(const matrix<unsigned char>& face, const cv::Rect& eye_region);

	std::pair<cv::Point, bool> eye_center_from(const matrix<unsigned char>& face, const cv::Rect& eye_region, const eye& previous_pos);

	std::set<std::pair<cv::Point, float>, less_cmp> find_maximas(const matrix<float>& matrix);

	void compute_eye_region(const cv::Rect& left_eye_region, const cv::Rect& right_eye_region, eye& left_eye, eye& right_eye) const;

	std::pair<cv::Rect, cv::Rect> precompute_eye_region(const cv::Rect& face_rect);

	// Computes a mask used by open-cv. Remove outer area of the eye region.
	matrix<unsigned char> filter_out_edges(matrix<float>& eye_mat);

	cv::CascadeClassifier face_classifier_;
};
