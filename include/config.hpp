#pragma once

namespace config {

	constexpr double eye_percent_top       = 0.25;
	constexpr double eye_percent_side      = 0.13;
	constexpr double eye_percent_height    = 0.30;
	constexpr double eye_percent_width     = 0.35;
	constexpr double face_smoothing_factor = 0.005;
	constexpr double weight_divisor        = 1.0;
	constexpr double gradient_treshold     = 50.0;
	constexpr double post_process_treshold = 0.97;

	constexpr float eye_rescale_width      = 50.f;

	constexpr int weight_blur_size         = 5;
}
