#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <cmath>
#include <matrix.hpp>
#include <eye_finder.hpp>

void process(const matrix<unsigned char>& pic, face f);

int main(int, const char*[]) {

	matrix<unsigned char> frame;
	eye_finder ef("res/haarcascade_frontalface_alt.xml");

	cv::VideoCapture capture{0};
	if (capture.isOpened()) {
		capture.read(frame);
		cv::flip(frame, frame, 1);
		std::optional<face> f = ef.find_eyes(frame);
		while (true) {
			capture.read(frame);

			// mirror it
			cv::flip(frame, frame, 1);

			// Apply the classifier to the frame
			if (!frame.empty()) {
				f = ef.find_eyes(frame, f);
				if (f) {
					process(frame, f.value());
				}
			} else {
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			// escape quits
//			int c = cv::waitKey(1);
//			if ((char) c == 27) { break; }

		}
	}

	return 0;
}

void process(const matrix<unsigned char>& pic, face f) {

	static std::pair<cv::Point,cv::Point> corners[4];
	static std::optional<uint> to_draw;
	static unsigned int count = 0;

	cv::Rect rects[] =
			{
					{0,0, pic.width() / 2, pic.height() / 2},
					{0,pic.height() / 2, pic.width() / 2, pic.height() / 2},
					{pic.width() / 2, pic.height() / 2, pic.width() / 2, pic.height() / 2},
					{pic.width() / 2, 0, pic.width() / 2, pic.height() / 2},
			};

	char t = static_cast<char>(cv::waitKey(1));
	if (t == ' ') {
		if (count < 4) {
			corners[count++] = {f.eyes.first.eye_position, f.eyes.second.eye_position};
		} else {
			if (to_draw) {
				to_draw.reset();
			} else {
				double min = std::numeric_limits<double>::infinity();
				for (auto i = 0u ; i < 4 ; ++i) {
					double dist = std::hypot(f.eyes.first.eye_position.x - corners[i].first.x,
					                         f.eyes.first.eye_position.y - corners[i].first.y);
					dist += std::hypot(f.eyes.second.eye_position.x - corners[i].second.x,
					                   f.eyes.second.eye_position.y - corners[i].second.y);

					if (dist < min) {
						min = dist;
						to_draw = i;
					}
				}
			}
		}
	} else if (t == 27) {
        exit(0);
    }

	f.eyes.first.eye_position.x += f.face_region.x;
	f.eyes.first.eye_position.y += f.face_region.y;
	f.eyes.second.eye_position.x += f.face_region.x;
	f.eyes.second.eye_position.y += f.face_region.y;

	matrix<unsigned char> display(pic.clone());

	if (to_draw) {
		cv::rectangle(display, rects[to_draw.value()], CV_RGB(40,220,40), -1);
	}

	cv::rectangle(display, f.face_region, CV_RGB(200, 0, 200));
	cv::circle(display, f.eyes.first.eye_position, 2, CV_RGB(40,40,200), 2);
	cv::circle(display, f.eyes.second.eye_position, 2, CV_RGB(200,40,40), 2);

	cv::imshow("aiue", display);
}
