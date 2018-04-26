#include <opencv2/imgproc.hpp>
#include <twin_matrixes.hpp>
#include <utils.hpp>
#include <queue>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "eye_finder.hpp"

std::optional<face> eye_finder::find_eyes(const matrix<unsigned char>& picture, const std::optional<face>& previous_face) {

	std::optional<std::pair<matrix<unsigned char>, face>> detected_face = find_face(picture);

	//-- face detected
	if (detected_face) {

		eye left_eye, right_eye;
		matrix<unsigned char> faceROI = detected_face->first(detected_face->second.face_region);
		cv::GaussianBlur(faceROI, faceROI, cv::Size(0, 0), config::face_smoothing_factor * detected_face->second.face_region.width);

		// find eye regions
		auto [left_eye_region, right_eye_region] = precompute_eye_region(detected_face->second.face_region);

		// find eyes centers
		if (previous_face) {
			auto [l_pos_in_local, l_did_not_move] = eye_center_from(faceROI, left_eye_region, previous_face->eyes.first);
			auto [r_pos_in_local, r_did_not_move] = eye_center_from(faceROI, right_eye_region, previous_face->eyes.second);
			left_eye.position_in_local = l_pos_in_local;
			right_eye.position_in_local = r_pos_in_local;

			if (l_did_not_move) {
				left_eye.consecutive_match = previous_face->eyes.first.consecutive_match + 1;
			} else {
				left_eye.consecutive_match = 0u;
			}

			if (r_did_not_move) {
				right_eye.consecutive_match = previous_face->eyes.second.consecutive_match + 1;
			} else {
				right_eye.consecutive_match = 0u;
			}

		} else {
			left_eye.position_in_local = eye_center(faceROI, left_eye_region), left_eye_region;
			right_eye.position_in_local = eye_center(faceROI, right_eye_region), right_eye_region;
		}

		// Compute a more accurate eye region
		compute_eye_region(left_eye_region, right_eye_region, left_eye, right_eye);

		// draw eye region
//		rectangle(faceROI, left_eye_region, 255);
//		rectangle(faceROI, right_eye_region, 255);
//		cv::circle(faceROI, left_eye.eye_position, 3, 255);
//		cv::circle(faceROI, right_eye.eye_position, 3, 255);

		face f;
		f.face_region = detected_face->second.face_region;
		f.eyes.first = left_eye;
		f.eyes.second = right_eye;
		return f;

	} else {
		return {};
	}
}

std::pair<cv::Rect, cv::Rect> eye_finder::precompute_eye_region(const cv::Rect& face_rect) {
	auto eye_region_width = static_cast<int>(face_rect.width * config::eye_percent_width);
	auto eye_region_height = static_cast<int>(face_rect.width * config::eye_percent_height);
	auto eye_region_top = static_cast<int>(face_rect.height * config::eye_percent_top);
	auto eye_region_left_left = static_cast<int>(face_rect.width * config::eye_percent_side);
	auto eye_region_left_right = static_cast<int>(face_rect.width - eye_region_width - face_rect.width * config::eye_percent_side);


	cv::Rect left_eye_region(eye_region_left_left,   eye_region_top, eye_region_width, eye_region_height);
	cv::Rect right_eye_region(eye_region_left_right, eye_region_top, eye_region_width, eye_region_height);

	return {left_eye_region, right_eye_region};
};

void eye_finder::compute_eye_region(const cv::Rect& left_eye_region, const cv::Rect& right_eye_region, eye& left_eye, eye& right_eye) const {

	left_eye.eye_position = unscale_pt(left_eye.position_in_local, left_eye_region);
	right_eye.eye_position = unscale_pt(right_eye.position_in_local, right_eye_region);

	cv::Rect leftRightCornerRegion(left_eye_region);
	leftRightCornerRegion.width -= left_eye.eye_position.x;
	leftRightCornerRegion.x += left_eye.eye_position.x;
	leftRightCornerRegion.height /= 2;
	leftRightCornerRegion.y += leftRightCornerRegion.height / 2;

	cv::Rect leftLeftCornerRegion(left_eye_region);
	leftLeftCornerRegion.width = left_eye.eye_position.x;
	leftLeftCornerRegion.height /= 2;
	leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;

	leftLeftCornerRegion.width += leftRightCornerRegion.width;
	left_eye.eye_region = leftLeftCornerRegion;

	cv::Rect rightLeftCornerRegion(right_eye_region);
	rightLeftCornerRegion.width = right_eye.eye_position.x;
	rightLeftCornerRegion.height /= 2;
	rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;

	cv::Rect rightRightCornerRegion(right_eye_region);
	rightRightCornerRegion.width -= right_eye.eye_position.x;
	rightRightCornerRegion.x += right_eye.eye_position.x;
	rightRightCornerRegion.height /= 2;
	rightRightCornerRegion.y += rightRightCornerRegion.height / 2;

	rightLeftCornerRegion.width += rightRightCornerRegion.width;
	right_eye.eye_region = rightLeftCornerRegion;

	// change eye centers to face coordinates
	right_eye.eye_position.x += right_eye_region.x;
	right_eye.eye_position.y += right_eye_region.y;
	left_eye.eye_position.x += left_eye_region.x;
	left_eye.eye_position.y += left_eye_region.y;
}

cv::Point eye_finder::eye_center(const matrix<unsigned char>& face, const cv::Rect& eye_region) {

	cv::Point maximum;
	matrix<float> gradient_matrix = compute_gradient_matrix(face, eye_region);
	cv::minMaxLoc(gradient_matrix, nullptr, nullptr, nullptr, &maximum, filter_out_edges(gradient_matrix));

	return maximum;
}

std::pair<cv::Point, bool> eye_finder::eye_center_from(const matrix<unsigned char>& face, const cv::Rect& eye_region, const eye& previous_pos) {
	matrix<float> gradient_matrix = compute_gradient_matrix(face, eye_region);

	auto local_maxima = find_maximas(gradient_matrix);

	auto heuristic = [e_p = previous_pos.position_in_local, e_cmc = previous_pos.consecutive_match]
			(const std::pair<cv::Point, float>& proposed_pos) {


		return proposed_pos.second * 1e-38f// * (1.f / (e_cmc + 1))
		       - std::hypot(proposed_pos.first.x - e_p.x, proposed_pos.first.y - e_p.y) * (e_cmc); // (e_cmc + 1.f);
	};

	std::pair<cv::Point, float> best_match;
	best_match.second = -std::numeric_limits<float>::infinity();
	for (const std::pair<cv::Point, float>& local_maximum : local_maxima) {
		if (heuristic(local_maximum) > heuristic(best_match)) {
			best_match.first = local_maximum.first;
			best_match.second = local_maximum.second;
		}
	}

//	std::cout << previous_pos.consecutive_match << '\n';

	return {
			best_match.first
			, std::hypot(previous_pos.position_in_local.x - best_match.first.x,
			             previous_pos.position_in_local.y - best_match.first.y)
			  < 15
	};
}

std::set<std::pair<cv::Point, float>, less_cmp> eye_finder::find_maximas(const matrix<float>& matrix) {
	double max_val;
	cv::Point max_loc;
	cv::minMaxLoc(matrix, nullptr, &max_val, nullptr, &max_loc);

	if (max_val - matrix(0,0) < 1.f && max_val - matrix(matrix.width() - 1, matrix.height() - 1) < 1.f) {
		return {};
	} else {
		std::set<std::pair<cv::Point, float>, less_cmp> set;
		set.emplace(max_loc, max_val);
		if (matrix.width() >= 10 && matrix.height() >= 10) {
			set.merge(find_maximas(matrix(cv::Rect(0, 0, matrix.width() / 2, matrix.height() / 2))));

			auto subset = find_maximas(matrix(cv::Rect(matrix.width() / 2, 0, matrix.width() - matrix.width() / 2, matrix.height() / 2)));
			for (const std::pair<cv::Point, float>& item : subset) {
				set.emplace(cv::Point(item.first.x + matrix.width() / 2, item.first.y), item.second);
			}

			subset = find_maximas(matrix(cv::Rect(0, matrix.height() / 2, matrix.width() / 2, matrix.height() - matrix.height() / 2)));
			for (const std::pair<cv::Point, float>& item : subset) {
				set.emplace(cv::Point(item.first.x, item.first.y + matrix.height() / 2), item.second);
			}

			subset = find_maximas(matrix(cv::Rect(matrix.width() / 2, matrix.height() / 2, matrix.width() - matrix.width() / 2, matrix.height() - matrix.height() / 2)));
			for (const std::pair<cv::Point, float>& item : subset) {
				set.emplace(cv::Point(item.first.x + matrix.width() / 2, item.first.y + matrix.height() / 2), item.second);
			}
		}
		return set;
	}
}

matrix<float> eye_finder::compute_gradient_matrix(const matrix<unsigned char>& face, const cv::Rect& eye_region) {


	using dpair = std::pair<double, double>;

	matrix<unsigned char> eyeRoi = rescale(face(eye_region));


	//-- Find the gradient
	twin_matrixes<double> gradient(eyeRoi.compute_x_gradient<double>(), eyeRoi.compute_y_gradient<double>());


	//-- Normalize and threshold the gradient
	matrix<double> magnitudes = magnitude(gradient.first(), gradient.second());
	double threshold = compute_threshold(magnitudes, config::gradient_treshold);

	for (auto x = 0u ; x < eyeRoi.width() ; ++x) {
		for (auto y = 0u ; y < eyeRoi.height() ; ++y) {

			if (magnitudes(x, y) > threshold) {
				gradient(x, y) /= magnitudes(x, y);
			} else {
				gradient(x, y) = {0.0, 0.0};
			}
		}
	}

	//-- Create a blurred and inverted image for weighting
	matrix<unsigned char> weight;
	cv::GaussianBlur(eyeRoi, weight, cv::Size(config::weight_blur_size, config::weight_blur_size), 0, 0);

	for (auto y = 0u ; y < weight.height() ; ++y) {
		for (auto x = 0u ; x < weight.width() ; ++x) {
			weight(x, y) = static_cast<unsigned char>(255u) - weight(x, y);
		}
	}

	// eval centers for each gradient
	matrix<double> gradient_rays(cv::Mat::zeros(eyeRoi.rows, eyeRoi.cols, CV_64F));
	for (auto y = 0u ; y < weight.height() ; ++y) {
		for (auto x = 0u ; x < weight.width() ; ++x) {

			if (gradient(x, y) != dpair{0., 0.}) {
				cast_centers_rays(cv::Point(x, y), weight, gradient(x, y), gradient_rays);
			}
		}
	}

	// scale all the values down, basically averaging them
	double numGradients = weight.height() * weight.width();
	matrix<float> scaled_gradients;
	gradient_rays.convertTo(scaled_gradients, CV_32F, 1.0 / numGradients);

	//-- Find the maximum value
	cv::Point maximum;
	double maxVal;
	cv::minMaxLoc(scaled_gradients, nullptr, &maxVal, nullptr, &maximum);

	//-- Post process to improve accuracy
	matrix<float> thresholded;
	cv::threshold(scaled_gradients, thresholded, maxVal * config::post_process_treshold, 0.0f, cv::THRESH_TOZERO);

	return thresholded;
}

void eye_finder::cast_centers_rays(cv::Point point, const matrix<unsigned char>& weight, twin_el<double> gradient,
                                   matrix<double>& out) {

	for (auto x = 0u ; x < out.width() ; ++x) {
		for (auto y = 0u ; y < out.height() ; ++y) {
			if (point.x == x && point.y == y) {
				continue;
			}

			// x->px vector
			double x_px = static_cast<int>(point.x - x); // signed result
			double y_py = static_cast<int>(point.y - y);

			// normalize
			double norm = std::hypot(x_px, y_py);
			double dot_prod = std::max(0., x_px * gradient.first() / norm + y_py * gradient.second() / norm);

			// square and multiply by the weight
			out(x, y) += dot_prod * dot_prod * (weight(x, y) / config::weight_divisor);
		}
	}
}

matrix<unsigned char> eye_finder::filter_out_edges(matrix<float>& eye_mat) {

	rectangle(eye_mat, cv::Rect(0, 0, eye_mat.cols, eye_mat.rows), 255);

	matrix<unsigned char> mask(eye_mat.rows, eye_mat.cols, CV_8U, 255);
	std::queue<cv::Point> todo;
	todo.push(cv::Point(0, 0));

	while (!todo.empty()) {
		cv::Point p = todo.front();
		todo.pop();
		if (eye_mat(p) == 0.0f) {
			continue;
		}

		// exploring all directions

		cv::Point np(p.x + 1, p.y);
		if (is_in(np, eye_mat)) {
			todo.push(np);
		}

		np.x = p.x - 1; // left
		if (is_in(np, eye_mat)) {
			todo.push(np);
		}

		np.x = p.x;
		np.y = p.y + 1; // down
		if (is_in(np, eye_mat)) {
			todo.push(np);
		}

		np.y = p.y - 1; // up
		if (is_in(np, eye_mat)) {
			todo.push(np);
		}

		// remove point from research
		eye_mat(p) = 0.f;
		mask(p) = 0;
	}
	return mask;
}

std::optional<std::pair<matrix<unsigned char>, face>> eye_finder::find_face(const matrix<unsigned char>& picture) {

	// detect face
	std::vector<cv::Rect> faces;

	std::vector<cv::Mat> rgbChannels(3);
	cv::split(picture, rgbChannels);
	matrix<unsigned char> frame_red(rgbChannels[2]);

	//-- Detect faces
	face_classifier_.detectMultiScale(
			frame_red
			, faces
			, 1.1
			, 2
			, CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT // NOLINT
			, cv::Size(150, 150)
	);

	if (faces.empty()) {
		return {};
	} else {
		face face;
		face.face_region = faces.front();
		return std::pair{std::move(frame_red), face};
	}
}
