#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <random>
#include <vector>
#include "BS_thread_pool.hpp"
//#include <algorithm>

#include "Case.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

// Resize binary, threshold img height to this amount, maintain 16:9 ratio
const int SCALE_HEIGHT = 144;  // 96x54 or 256x144 both seems reasonable
// Multiply by this factor to convert 720p scale to the current scale.Divide to undo
const double CONVERSION_720P = (double)SCALE_HEIGHT / 720;

vector<Case> cases;
Mat debug_frame;  // Global scope OK since only used for debugging


Mat preprocess_frame(Mat frame) {
    Mat bin_frame;
    cvtColor(frame, bin_frame, COLOR_BGR2GRAY);
    threshold(bin_frame, bin_frame, 100, 255, THRESH_BINARY);
    // cv2.imshow("bin", binary)  # For reference
    int kernel_size = 60;  // 80x80 does not work b / c prev centroid outside case after moving
    // int kernel_size = int(kernel_size_720p * (frame.shape[0] / 720));  // convert kernel size for 720p to current resolution
    Mat kernel = getStructuringElement(MORPH_RECT, Size(kernel_size, kernel_size));
    // Ref: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    erode(bin_frame, bin_frame, kernel);

    // Scale to remove unnecessary info(assumes 9:16 ratio)
    resize(bin_frame, bin_frame, Size((int)(SCALE_HEIGHT * (16.0 / 9)), SCALE_HEIGHT));
    return bin_frame;
}


vector<future<Mat>> preprocess(cv::String vid_src) {
    BS::thread_pool pool;
    cout << "Your computer has " << pool.get_thread_count() << " threads" << endl;
    vector<future<Mat>> frame_tasks;
    VideoCapture cap(vid_src);
    cap.set(CAP_PROP_POS_FRAMES, 900);
    while (true) {
        Mat frame;
        bool ret = cap.read(frame);
        if (!ret) break;
        if ((int)cap.get(CAP_PROP_POS_FRAMES) % 2 == 1)
            continue;

        //pool.push_task(preprocess_frame, frame);
        //frame_tasks.push_back(pool.submit(preprocess_frame, frame));
    }
    cap.release();
    return frame_tasks;
}


bool tick(future<Mat>& bin_frame, bool debug) {
    // TODO

    if (debug) {
        // imshow("debug_frame", debug_frame);
        imshow("bin_frame", bin_frame.get());
        char key;
        if (cases.size() < 16)
            key = waitKey(1);
        else
            key = waitKey(1);
        cout << ".";
        if (key == 'q')
            return false;
    }
    return true;
}

int main() {
    cases.clear();
    cout << "Start!" << endl;
    auto start = high_resolution_clock::now();
    auto vid_src = "../../IMG_4383.MOV";
    auto bin_frames = preprocess(vid_src);
    auto preprocess_end = high_resolution_clock::now();
    cout << "Preprocessed in " << duration_cast<seconds>(preprocess_end - start).count() << " seconds";
    for (auto& frame : bin_frames) {
        if (!tick(frame, true))
            break;
    }
    auto stop = high_resolution_clock::now();
    cout << endl;
    printf("Finished in %d seconds", (int)duration_cast<seconds>(stop - start).count());
    //cout << "Finished in " << duration_cast<seconds>(stop - start) << " seconds";
    destroyAllWindows();
}
