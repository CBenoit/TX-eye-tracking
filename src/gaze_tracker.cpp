#include <string_view>
#include <iostream>

#include "gaze_tracker.hpp"

void gaze_tracker::configure(const face& centered_gaze, const face& top_left_gaze,
                             const face& top_right_gaze, const face& bottom_right_gaze,
                             const face& bottom_left_gaze, const face& centered_gaze_2) {
	m_centered_gaze.first.x = (centered_gaze.eyes.first.eye_position.x + centered_gaze_2.eyes.first.eye_position.x) / 2;
	m_centered_gaze.first.y = (centered_gaze.eyes.first.eye_position.y + centered_gaze_2.eyes.first.eye_position.y) / 2;
	m_centered_gaze.second.x = (centered_gaze.eyes.second.eye_position.x + centered_gaze_2.eyes.second.eye_position.x) / 2;
	m_centered_gaze.second.y = (centered_gaze.eyes.second.eye_position.y + centered_gaze_2.eyes.second.eye_position.y) / 2;

	m_bottom_right_gaze.first.x = (bottom_right_gaze.eyes.first.eye_position.x + top_right_gaze.eyes.first.eye_position.x) / 2;
	m_bottom_right_gaze.first.y = (bottom_right_gaze.eyes.first.eye_position.y + bottom_left_gaze.eyes.first.eye_position.y) / 2;
	m_bottom_right_gaze.second.x = (bottom_right_gaze.eyes.second.eye_position.x + top_right_gaze.eyes.second.eye_position.x) / 2;
	m_bottom_right_gaze.second.y = (bottom_right_gaze.eyes.second.eye_position.y + bottom_left_gaze.eyes.second.eye_position.y) / 2;

	m_top_left_gaze.first.x = (top_left_gaze.eyes.first.eye_position.x + bottom_left_gaze.eyes.first.eye_position.x) / 2;
	m_top_left_gaze.first.y = (top_left_gaze.eyes.first.eye_position.y + top_right_gaze.eyes.first.eye_position.y) / 2;
	m_top_left_gaze.second.x = (top_left_gaze.eyes.second.eye_position.x + bottom_left_gaze.eyes.second.eye_position.x) / 2;
	m_top_left_gaze.second.y = (top_left_gaze.eyes.second.eye_position.y + top_right_gaze.eyes.second.eye_position.y) / 2;
}

void gaze_tracker::configure(std::string_view configure_window_name, cv::VideoCapture& camera) {

	auto find_at = [&camera, configure_window_name, this] (float x_fact, float y_fact) -> face {
		matrix<unsigned char> frame;
		std::optional<face> f;
		bool done = false;
		camera.read(frame);
		cv::Rect rectangle(static_cast<int>((frame.width() - 25) * x_fact) + 5, static_cast<int>((frame.height() - 25) * y_fact) + 5, 14, 14);
		while (!done) {

			cv::flip(frame, frame, 1);
			f = m_ef.find_eyes(frame);
			cv::rectangle(frame, rectangle, CV_RGB(0, 0, 0), 4);
			cv::rectangle(frame, rectangle, CV_RGB(255, 255, 255), CV_FILLED);
			if (f) {
				f->eyes.first.eye_position.x += f->face_region.x;
				f->eyes.first.eye_position.y += f->face_region.y;
				f->eyes.second.eye_position.x += f->face_region.x;
				f->eyes.second.eye_position.y += f->face_region.y;
				cv::circle(frame, f->eyes.first.eye_position, 2, CV_RGB(127,0,255));
				cv::circle(frame, f->eyes.second.eye_position, 2, CV_RGB(127,0,255));
			}
			cv::imshow(configure_window_name.data(), frame);
			camera.read(frame);

			auto t = static_cast<char>(cv::waitKey(1));
			if (t == ' ' && f) {
				done = true;
			}
		}
		return *f;
	};

	face center = find_at(.5f, .5f);
	face top_left = find_at(0.f, 0.f);
	face top_right = find_at(1.f, 0.f);
	face bottom_right = find_at(1.f, 1.f);
	face bottom_left = find_at(0.f, 1.f);
	face center2 = find_at(.5f, .5f);

	configure(center, top_left, top_right, bottom_right, bottom_left, center2);

	cv::destroyWindow(configure_window_name.data());
}

std::optional<cv::Point2i> gaze_tracker::track(const matrix<unsigned char>& pic) {

	std::optional<face> f = m_ef.find_eyes(pic);
	if (!f) {
		return {};
	}

// Geometry
//	float meters_to_pixel_ratio = (f->face_region.height / cst::face_height_meters + f->face_region.width / cst::face_width_meters) / 2.f;
//
//	cv::Point2i left_gaze, right_gaze;
//	auto left_eye = f->eyes.first.eye_position;
//	auto right_eye = f->eyes.second.eye_position;
//
//	left_gaze.x = static_cast<int>((left_eye.x * cst::distance_to_screen_meters * meters_to_pixel_ratio)
//	                               / std::sqrt(std::pow(cst::eye_ball_size_meters * meters_to_pixel_ratio, 2) + std::pow(left_eye.x - m_centered_gaze.first.x, 2)));
//
//	left_gaze.y = static_cast<int>((left_eye.y * cst::distance_to_screen_meters * meters_to_pixel_ratio)
//	                               / std::sqrt(std::pow(cst::eye_ball_size_meters * meters_to_pixel_ratio, 2) + std::pow(left_eye.y - m_centered_gaze.first.y, 2)));
//
//	right_gaze.x = static_cast<int>((right_eye.x * cst::distance_to_screen_meters * meters_to_pixel_ratio)
//	                               / std::sqrt(std::pow(cst::eye_ball_size_meters * meters_to_pixel_ratio, 2) + std::pow(right_eye.x - m_centered_gaze.first.x, 2)));
//
//	right_gaze.y = static_cast<int>((right_eye.y * cst::distance_to_screen_meters * meters_to_pixel_ratio)
//	                               / std::sqrt(std::pow(cst::eye_ball_size_meters * meters_to_pixel_ratio, 2) + std::pow(right_eye.y - m_centered_gaze.first.y, 2)));
//
//
//	return {{left_gaze, right_gaze}};

	cv::Point2i left_eye = f->eyes.first.eye_position;
	cv::Point2i right_eye = f->eyes.second.eye_position;

	std::pair<float, float> eyes_relative_pos;

	if (is_left({left_eye, right_eye})) {

		if (is_top({left_eye, right_eye})) {
			eyes_relative_pos = interpol_topleft({left_eye, right_eye});
		} else {
			eyes_relative_pos = interpol_botleft({left_eye, right_eye});
		}

	} else {

		if (is_top({left_eye, right_eye})) {
			eyes_relative_pos = interpol_topright({left_eye, right_eye});
		} else {
			eyes_relative_pos = interpol_botright({left_eye, right_eye});
		}
	}

	auto min_x = std::min(m_top_left_gaze.first.x, m_top_left_gaze.second.x);
	auto min_y = std::min(m_top_left_gaze.first.y, m_top_left_gaze.second.y);
	auto max_x = std::max(m_bottom_right_gaze.first.x, m_bottom_right_gaze.second.x);
	auto max_y = std::max(m_bottom_right_gaze.first.y, m_bottom_right_gaze.second.y);

	auto x_length = max_x - min_x;
	auto y_length = max_y - min_y;

	std::cout << m_bottom_right_gaze.first << " " << m_bottom_right_gaze.second << " " << m_top_left_gaze.first << " " << m_top_left_gaze.second << '\n';
	std::cout << x_length << " - " << y_length << '\n';
	std::cout << eyes_relative_pos.first << " - " << eyes_relative_pos.second << '\n';
	std::cout << min_x << " - " << min_y << "\n\n\n";

	return std::optional{cv::Point2i{
			        static_cast<int>(eyes_relative_pos.first * x_length) + min_x,
			        static_cast<int>(eyes_relative_pos.second * y_length) + min_y
	        }};
}

bool gaze_tracker::is_left(const std::pair<cv::Point2i, cv::Point2i>& eyes) const noexcept {
	auto l_left = eyes.first.x - m_centered_gaze.first.x;
	auto r_left = eyes.second.x - m_centered_gaze.second.x;
	return (l_left + r_left) <= 0;
}

bool gaze_tracker::is_right(const std::pair<cv::Point2i, cv::Point2i>& eyes) const noexcept {
	return !is_left(eyes);
}

bool gaze_tracker::is_top(const std::pair<cv::Point2i, cv::Point2i>& eyes) const noexcept {
	auto l_right = eyes.first.y - m_centered_gaze.first.y;
	auto r_right = eyes.second.y - m_centered_gaze.second.y;
	return (l_right + r_right) <= 0;
}

bool gaze_tracker::is_bottom(const std::pair<cv::Point2i, cv::Point2i>& eyes) const noexcept {
	return !is_top(eyes);
}

std::pair<float, float> gaze_tracker::interpol_topleft(const std::pair<cv::Point2i, cv::Point2i>& eyes) const {
	auto l_x_length = m_centered_gaze.first.x - m_top_left_gaze.first.x;
	auto r_x_length = m_centered_gaze.second.x - m_top_left_gaze.second.x;

	auto l_x_dist = eyes.first.x - m_top_left_gaze.first.x;
	auto r_x_dist = eyes.second.x - m_top_left_gaze.second.x;

	auto l_y_length = m_centered_gaze.first.y - m_top_left_gaze.first.y;
	auto r_y_length = m_centered_gaze.second.y - m_top_left_gaze.second.y;

	auto l_y_dist = eyes.first.y - m_top_left_gaze.first.y;
	auto r_y_dist = eyes.second.y - m_top_left_gaze.second.y;

	float x = (static_cast<float>(l_x_dist) + static_cast<float>(r_x_dist))
	          / (static_cast<float>(l_x_length) + static_cast<float>(r_x_length));

	float y = (static_cast<float>(l_y_dist) + static_cast<float>(r_y_dist))
	          / (static_cast<float>(l_y_length) + static_cast<float>(r_y_length));

	return {x / 2.f, y / 2.f};

}

std::pair<float, float> gaze_tracker::interpol_botleft(const std::pair<cv::Point2i, cv::Point2i>& eyes) const {
	auto l_x_length = m_centered_gaze.first.x - m_top_left_gaze.first.x;
	auto r_x_length = m_centered_gaze.second.x - m_top_left_gaze.second.x;

	auto l_x_dist = eyes.first.x - m_top_left_gaze.first.x;
	auto r_x_dist = eyes.second.x - m_top_left_gaze.second.x;

	auto l_y_length = m_bottom_right_gaze.first.y - m_centered_gaze.first.y;
	auto r_y_length = m_bottom_right_gaze.second.y - m_centered_gaze.second.y;

	auto l_y_dist = eyes.first.y - m_centered_gaze.first.y;
	auto r_y_dist = eyes.second.y - m_centered_gaze.second.y;

	float x = (static_cast<float>(l_x_dist) + static_cast<float>(r_x_dist))
	          / (static_cast<float>(l_x_length) + static_cast<float>(r_x_length));

	float y = (static_cast<float>(l_y_dist) + static_cast<float>(r_y_dist))
	          / (static_cast<float>(l_y_length) + static_cast<float>(r_y_length));

	return {x / 2.f, y / 2.f + 0.5f};

}

std::pair<float, float> gaze_tracker::interpol_topright(const std::pair<cv::Point2i, cv::Point2i>& eyes) const {
	auto l_x_length = m_bottom_right_gaze.first.x - m_centered_gaze.first.x;
	auto r_x_length = m_bottom_right_gaze.second.x - m_centered_gaze.second.x;

	auto l_x_dist = eyes.first.x - m_centered_gaze.first.x;
	auto r_x_dist = eyes.second.x - m_centered_gaze.second.x;

	auto l_y_length = m_centered_gaze.first.y - m_top_left_gaze.first.y;
	auto r_y_length = m_centered_gaze.second.y - m_top_left_gaze.second.y;

	auto l_y_dist = eyes.first.y - m_top_left_gaze.first.y;
	auto r_y_dist = eyes.second.y - m_top_left_gaze.second.y;

	float x = (static_cast<float>(l_x_dist) + static_cast<float>(r_x_dist))
	          / (static_cast<float>(l_x_length) + static_cast<float>(r_x_length));

	float y = (static_cast<float>(l_y_dist) + static_cast<float>(r_y_dist))
	          / (static_cast<float>(l_y_length) + static_cast<float>(r_y_length));

	return {x / 2.f + 0.5f, y / 2.f};
}

std::pair<float, float> gaze_tracker::interpol_botright(const std::pair<cv::Point2i, cv::Point2i>& eyes) const {
	auto l_x_length = m_bottom_right_gaze.first.x - m_centered_gaze.first.x;
	auto r_x_length = m_bottom_right_gaze.second.x - m_centered_gaze.second.x;

	auto l_x_dist = eyes.first.x - m_centered_gaze.first.x;
	auto r_x_dist = eyes.second.x - m_centered_gaze.second.x;

	auto l_y_length = m_bottom_right_gaze.first.y - m_centered_gaze.first.y;
	auto r_y_length = m_bottom_right_gaze.second.y - m_centered_gaze.second.y;

	auto l_y_dist = eyes.first.y - m_centered_gaze.first.y;
	auto r_y_dist = eyes.second.y - m_centered_gaze.second.y;

	float x = (static_cast<float>(l_x_dist) + static_cast<float>(r_x_dist))
	          / (static_cast<float>(l_x_length) + static_cast<float>(r_x_length));

	float y = (static_cast<float>(l_y_dist) + static_cast<float>(r_y_dist))
	          / (static_cast<float>(l_y_length) + static_cast<float>(r_y_length));

	return {x / 2.f + 0.5f, y / 2.f + 0.5f};
}


































