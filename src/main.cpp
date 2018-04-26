#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <cmath>
#include <matrix.hpp>
#include <eye_finder.hpp>

/** Global variables */

const char * face_cascade_name = "res/haarcascade_frontalface_alt.xml";


cv::CascadeClassifier face_cascade; // NOLINT

int main(int, const char*[]) {

	matrix<unsigned char> frame;
	eye_finder ef(face_cascade_name);

	cv::VideoCapture capture{0};
	if (capture.isOpened()) {
		while (true) {
			capture.read(frame);

			// mirror it
			cv::flip(frame, frame, 1);

			// Apply the classifier to the frame
			if (!frame.empty()) {
				ef.find_eyes(frame);
			} else {
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			// escape quits
			int c = cv::waitKey(1);
			if ((char) c == 27) { break; }

		}
	}
	
	return 0;
}
