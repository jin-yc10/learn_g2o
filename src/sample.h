#include <cmath>
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace std;

class Sample {
public:
  static int uniform(int from, int to);
  static double uniform();
  static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr){
  return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma){
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to){
  return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform(){
  return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma){
  return gauss_rand(0., sigma);
}

DEFINE_bool(noisy_cam, false, "add noise to camera");
DEFINE_bool(noisy_3d, false, "add noise to 3d pts");
DEFINE_bool(noisy_2d, false, "add noise to 2d pts");

class ObservationSet {
    struct Params {
        int nOb;
        int nPt;
        cv::Size sz;
        cv::Mat K; bool noisy_cam;
        cv::Mat pt_3d; bool noisy_pt_3d;
        bool noisy_pt_2d;
    };
    struct Observe {
    };
};