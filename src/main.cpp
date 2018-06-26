#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <cmath>
#include <matrix.hpp>
#include <eye_finder.hpp>
#include <gaze_tracker.hpp>

void process(const matrix<unsigned char>& pic, face f);

int main(int, const char*[]) {

	matrix<unsigned char> frame;
	gaze_tracker gaze("res/haarcascade_frontalface_alt.xml");

	cv::VideoCapture capture{0};
	if (capture.isOpened()) {
		gaze.configure("Configuring", capture);
		while (true) {
			capture.read(frame);
			cv::flip(frame, frame, 1);
			matrix<unsigned char> copy(frame.clone());
			auto g = gaze.track(copy);
			if (g) {
				std::cout << g->x << " - " << g->y << '\n';
				cv::circle(frame, *g, 3, CV_RGB(40,40,200), 2);
			} else {
				cv::blur(frame, frame, cv::Size(20, 20));
			}
			cv::imshow("demo", frame);
			if (cv::waitKey(1) == 27) {
				exit(0);
			}
		}

	}

	return 0;
}
