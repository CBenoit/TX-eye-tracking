#ifndef TX_EYE_TRACKER_GAZE_TRACKER_HPP
#define TX_EYE_TRACKER_GAZE_TRACKER_HPP


#include "eye_finder.hpp"

namespace cst {

	// unused
	inline const float face_height_meters = 0.25f;
	inline const float face_width_meters = 0.21f;
	inline const float distance_to_screen_meters = 1.f;
	inline const float eye_ball_size_meters = 0.012f;
}

class gaze_tracker {

public:
	explicit gaze_tracker(std::string_view cascade_path) : m_ef(cascade_path) {}

	void configure(const face& centered_gaze, const face& top_left_gaze,
	               const face& top_right_gaze, const face& bottom_right_gaze,
	               const face& bottom_left_gaze, const face& centered_gaze_2);

	void configure(std::string_view configure_window_name, cv::VideoCapture& camera);

	std::optional<cv::Point2i> track(const matrix<unsigned char>& pic);


private:

	bool is_left(const std::pair<cv::Point2i, cv::Point2i>& eyes) const noexcept;
	bool is_right(const std::pair<cv::Point2i, cv::Point2i>& eyes) const noexcept;
	bool is_top(const std::pair<cv::Point2i, cv::Point2i>& eyes) const noexcept;
	bool is_bottom(const std::pair<cv::Point2i, cv::Point2i>& eyes) const noexcept;

	std::pair<float, float> interpol_topleft(const std::pair<cv::Point2i, cv::Point2i>& eyes) const;
	std::pair<float, float> interpol_botleft(const std::pair<cv::Point2i, cv::Point2i>& eyes) const;
	std::pair<float, float> interpol_topright(const std::pair<cv::Point2i, cv::Point2i>& eyes) const;
	std::pair<float, float> interpol_botright(const std::pair<cv::Point2i, cv::Point2i>& eyes) const;

	eye_finder m_ef;
	std::pair<cv::Point2i, cv::Point2i> m_top_left_gaze{};
	std::pair<cv::Point2i, cv::Point2i> m_bottom_right_gaze{};
	std::pair<cv::Point2i, cv::Point2i> m_centered_gaze{};
};


#endif //TX_EYE_TRACKER_GAZE_TRACKER_HPP
