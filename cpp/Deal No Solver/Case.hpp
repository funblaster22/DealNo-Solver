#ifndef CASE_H
#define CASE_H

#include "Box.hpp"
#include "FixedQueue.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

cv::Scalar hsv2bgr(int h, int s, int v) {
	// Don't question it. ChatGPT converted this from Python
	cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, s, v));
	cv::Mat bgr;
	cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
	return cv::Scalar(bgr.at<cv::Vec3b>(0, 0));
}

struct Case : public Box {
	int value;
	// How many frames to abstain from moving (set after swapping)
	int cooldown = 0;
	cv::Vec<double, 2> momentum = {0, 0};
	cv::Scalar color = hsv2bgr(rand() % 255, 255, 255);
	FixedQueue<cv::Vec<double, 2>, 3> momentum_history;

	Case(int cx, int cy, int w, int h, int value) : Box(cx, cy, w, h), value(value) {}
};

#endif