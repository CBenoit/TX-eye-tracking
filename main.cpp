#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <thread>

using namespace std;
using namespace cv;

int it = 0;

/* Function Headers */
Point detectFace(Mat frame, Point priorCenter);
void detect_eyes(const cv::Mat& frame);

CascadeClassifier face_cascade, eyes_cascade;

String face_window = "Face View";

int main() {

	VideoCapture cap(0); // capture from default camera
	Mat frame;
	Point priorCenter(0, 0);

	face_cascade.load("haarcascade_frontalface_alt.xml"); // load face classifiers
//	eyes_cascade.load("haarcascade_eye_tree_eyeglasses.xml"); // load eye classifiers
	eyes_cascade.load("haarcascade_eye.xml"); // load eye classifiers

	namedWindow(face_window,
	            CV_WINDOW_AUTOSIZE |
	            CV_WINDOW_FREERATIO |
	            CV_GUI_EXPANDED);

	// Loop to capture frames
	while(cap.read(frame)) {

		// Apply the classifier to the frame, i.e. find face
		priorCenter = detectFace(frame, priorCenter);

		if(waitKey(30) >= 0) // spacebar
			break;
		std::this_thread::sleep_for(100ms);
	}
	return 0;
}

/**
 * Output a frame of only the the rectangle centered at point
 */
Mat outputFrame(Mat frame, Point center, int w, int h) {

	int x = (center.x - w/2);
	int y = (center.y - 3*h/5);

	if(x + w > frame.size().width - 2 || x < 0 ||
	   y + h > frame.size().height - 2 || y < 0 &&
	                                      frame.size().width > 16 &&
	                                      frame.size().height > 16)
		return frame(Rect(5, 5, 10, 10));

	// output frame of only face
	return frame(Rect(x, y, w, h));
}

// Find face from eyes
Point faceFromEyes(Point priorCenter, const Mat& face) {

	std::vector<Rect> eyes;
	int avg_x = 0;
	int avg_y = 0;

	// Try to detect eyes, if no face is found
	eyes_cascade.detectMultiScale(face, eyes, 1.1, 2,
	                              0 |CASCADE_SCALE_IMAGE, Size(30, 30));

	// Iterate over eyes
	for(size_t j = 0; j < eyes.size(); j++) {

		// centerpoint of eyes
		Point eye_center(priorCenter.x + eyes[j].x + eyes[j].width/2,
		                 priorCenter.y + eyes[j].y + eyes[j].height/2);

		// Average center of eyes
		avg_x += eye_center.x;
		avg_y += eye_center.y;
	}

	// Use average location of eyes
	if(eyes.size() > 0) {

		priorCenter.x = avg_x / eyes.size();
		priorCenter.y = avg_y / eyes.size();
	}

	return priorCenter;
}

// Rounds up to multiple
int roundUp(int numToRound, int multiple) {

	if (multiple == 0)
		return numToRound;

	int remainder = abs(numToRound) % multiple;
	if (remainder == 0)
		return numToRound;
	if (numToRound < 0)
		return -(abs(numToRound) - remainder);
	return numToRound + multiple - remainder;
}

// Detect face and display it
Point detectFace(Mat frame, Point priorCenter) {

	std::vector<Rect> faces;
	Mat frame_gray, frame_lab, output, temp;
	int h = frame.size().height - 1;
	int w = frame.size().width - 1;
	int minNeighbors = 2;
	bool faceNotFound = false;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);   // Convert to gray
	equalizeHist(frame_gray, frame_gray);          // Equalize histogram

	// Detect face with open source cascade
	face_cascade.detectMultiScale(frame_gray, faces,
	                              1.1, minNeighbors,
	                              0|CASCADE_SCALE_IMAGE, Size(30, 30));


	// iterate over faces
	for(size_t i = 0; i < faces.size(); i++) {

		// Find center of face
		Point center(faces[i].x + faces[i].width/2,
		             faces[i].y + faces[i].height/2);

		// Generate width and height of face, round to closest 1/4 of frame height
		h = roundUp(faces[i].height, frame.size().height / 4);
		w = 3 * h / 5;

		// If priorCenter not yet initialized, initialize
		if(priorCenter.x == 0) {
			priorCenter = center;
			temp = outputFrame(frame, center, w, h);
			break;
		}

		// Check to see if it's probably the same user
		if(abs(center.x - priorCenter.x) < frame.size().width / 6 &&
		   abs(center.y - priorCenter.y) < frame.size().height / 6) {

			// Check to see if the user moved enough to update position
			if(abs(center.x - priorCenter.x) < 7 &&
			   abs(center.y - priorCenter.y) < 7){
				center = priorCenter;
			}

			// Smooth new center compared to old center
			center.x = (center.x + 2*priorCenter.x) / 3;
			center.y = (center.y + 2*priorCenter.y) / 3;
			priorCenter = center;

			// output frame of only face
			temp = outputFrame(frame, center, w, h);

			break; // exit, primary users face probably found

		} else {
			faceNotFound = true;
		}
	}

	if(faceNotFound) {

		// Findface from eyes
		Rect r(priorCenter.x, priorCenter.y, w, h);
		if(priorCenter.x + w > frame_gray.size().width - 2 &&
		   priorCenter.y + h > frame_gray.size().height - 2){

			priorCenter = faceFromEyes(priorCenter, frame_gray(r));

			// Generate temporary face location
			temp = outputFrame(frame, priorCenter, w, h);
		}
	}

	// Check to see if new face found
	if(temp.size().width > 2)
		output = temp;
	else
		output = frame;

	// Display only face
	//faceFromEyes(priorCenter, frame_gray);
	//imshow(face_window, output);
	detect_eyes(output);

	return priorCenter;
}


void detect_eyes(const cv::Mat& frame) {

		// Invert the source image and convert to grayscale
		cv::Mat gray;
		cv::cvtColor(~frame, gray, CV_BGR2GRAY);

		// Convert to binary image by thresholding it
		cv::threshold(gray, gray, 210, 255, cv::THRESH_BINARY);

		// Find all contours
		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		// Fill holes in each contour
		cv::drawContours(gray, contours, -1, CV_RGB(255, 255, 255), -1);

		for (int i = 0; i < contours.size(); i++) {
			double area = cv::contourArea(contours[i]);
			cv::Rect rect = cv::boundingRect(contours[i]);
			int radius = rect.width / 2;

			// If contour is big enough and has round shape
			// Then it is the pupil
			if (area >= 30 &&
			    std::abs(1 - ((double) rect.width / (double) rect.height)) <= 0.2 &&
			    std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2) {
				cv::circle(frame, cv::Point(rect.x + radius, rect.y + radius), radius, CV_RGB(255, 0, 0), 2);
			}
		}

		cv::imshow("image", frame);
}
