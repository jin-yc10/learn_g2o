//
// Created by 金宇超 on 16/8/23.
//

#include "sample.h"

#include <opencv2/opencv.hpp>
#include <fstream>

#define VAL(v) std::cout << #v" = " << v << std::endl

class FundamentalTest: public ObservationSet {
public:
    FundamentalTest(const ObservationSet::Params params) :
            ObservationSet(params) {
    }

    void generate() {

        Observe ob1;
        cv::Mat R1 = cv::Mat::eye(3,3,CV_32F);
        cv::Mat t1 = cv::Mat::zeros(3,params.pt_3d.rows,CV_32F);
        cv::Mat p1 = params.K * ( R1 * params.pt_3d.t() + t1 );

        ob1.real_R = R1;
        ob1.real_t = t1;

        for(int i=0; i<params.pt_3d.rows; i++) {
            cv::Point2f pt;
            if(FLAGS_noisy_2d) {
                pt.x = p1.at<float>(0,i)/p1.at<float>(2,i) + Sample::uniform(-2, 2);
                pt.y = p1.at<float>(1,i)/p1.at<float>(2,i) + Sample::uniform(-2, 2);
            } else {
                pt.x = p1.at<float>(0,i)/p1.at<float>(2,i);// + Sample::uniform(-2, 2);
                pt.y = p1.at<float>(1,i)/p1.at<float>(2,i);// + Sample::uniform(-2, 2);
            }
            ob1.pt_2d.push_back(pt);
        }

        Observe ob2;
        float r_raw[] = {0.0f, 3.14159f*30.0/180.0f, 0.0f};
        cv::Mat r2 = cv::Mat(1,3,CV_32F, r_raw);
        cv::Mat R2;
        cv::Rodrigues(r2, R2);
        float t[] = {1.0f, 2.0f, 3.0f};
        cv::Mat t2 = cv::Mat(3,1,CV_32F,t);
        cv::repeat(t2, 1, 21, t2);
        cv::Mat p2 = params.K * ( R2 * params.pt_3d.t() + t2 );

        ob2.real_R = R2;
        ob2.real_t = t2;

        for(int i=0; i<params.pt_3d.rows; i++) {
            cv::Point2f pt;
            if(FLAGS_noisy_2d) {
                pt.x = p2.at<float>(0,i)/p2.at<float>(2,i) + Sample::uniform(-2, 2);
                pt.y = p2.at<float>(1,i)/p2.at<float>(2,i) + Sample::uniform(-2, 2);
            } else {
                pt.x = p2.at<float>(0,i)/p2.at<float>(2,i);// + Sample::uniform(-2, 2);
                pt.y = p2.at<float>(1,i)/p2.at<float>(2,i);// + Sample::uniform(-2, 2);
            }
            ob2.pt_2d.push_back(pt);
        }

        obs.push_back(ob1);
        obs.push_back(ob2);
    }

    void report() {
        std::cout << obs[0].real_R << std::endl;
        std::cout << obs[1].real_R << std::endl;

        for(auto it = obs[0].pt_2d.begin(); it != obs[0].pt_2d.end(); it++ ) {
            std::cout << *it << std::endl;
        }
    }
};

void get3D(const vector<vector<cv::Point2f> > &pt_2d,
           const vector<cv::Mat> Ps,
           const int point_idx,
           cv::Mat &pt_3d) {
    int numViews = pt_2d.size();
    cv::Mat matrA(numViews * 2, 4, CV_32FC1);
    for (int i = 0; i < numViews; ++i) {
        const cv::Mat &P = Ps[i];
        const cv::Point2f &pt = pt_2d[i][point_idx];
        matrA.row(2 * i) = pt.x * P.row(2) - P.row(0);
        matrA.row(2 * i + 1) = pt.y * P.row(2) - P.row(1);
    }
    cv::Mat matrW, matrU, matrV;
    cv::SVD::compute(matrA, matrW, matrU, matrV);
    pt_3d = matrV(cv::Range(3, 4), cv::Range(0, 3)) / matrV.at<float>(3, 3);
}

bool DecomposeEtoRandT(
        const cv::Mat& E,
        cv::Mat& R1,
        cv::Mat& R2,
        cv::Mat& t1,
        cv::Mat& t2) {
    cv::SVD svd(E,cv::SVD::MODIFY_A);
    cv::Mat svd_u = svd.u;
    cv::Mat svd_vt = svd.vt;
    cv::Mat svd_w = svd.w;
    double singular_values_ratio = fabsf(svd_w.at<float>(0) / svd_w.at<float>(1));
    if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
    if (singular_values_ratio < 0.7) {
        cout << "singular values are too far apart\n";
        return false;
    }
    cv::Matx33f W(0,-1,0,	//HZ 9.13
                1,0,0,
                0,0,1);
    cv::Matx33f Wt(0,1,0,
                -1,0,0,
                0,0,1);
    R1 = svd_u * cv::Mat(W) * svd_vt; //HZ 9.19
    R2 = svd_u * cv::Mat(Wt) * svd_vt; //HZ 9.19
    t1 = svd_u.col(2); //u3
    t2 = -svd_u.col(2); //u3

    return true;
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    ObservationSet::Params params;

    float raw_k[9] = {
            800, 0.0, 320,
            0.0, 800, 240,
            0.0, 0.0, 1.0f,
    };

    params.K = cv::Mat(3, 3, CV_32F, raw_k);

    std::ifstream ifs("../data/mean.txt");
    float mean_pose[3 * 21];
    cv::Mat MeanPoseMat;
    for (int i = 0; i < 21; i++) {
        ifs >> mean_pose[3 * i + 0]
        >> mean_pose[3 * i + 1]
        >> mean_pose[3 * i + 2];
    }
    MeanPoseMat = cv::Mat(21, 3, CV_32F, mean_pose);
    params.pt_3d = MeanPoseMat;

    params.nOb = 2;
    params.nPt = 21;
    params.sz = cv::Size(640, 480);

    FundamentalTest t(params);
    t.generate();
    t.report();

    std::vector<uchar> masks;
    cv::Mat F = cv::findFundamentalMat(t.obs[0].pt_2d, t.obs[1].pt_2d, CV_FM_RANSAC, 3.0, 0.99, masks);
    F.convertTo(F, CV_32F);
    cv::Mat fu, fvt, fw;
    cv::SVD svdF(F, cv::SVD::MODIFY_A);
    VAL(svdF.vt);
    VAL(F);
    cv::Mat E = params.K.t() * F * params.K;
    VAL(E);
    cv::Mat R1, R2;
    cv::Mat t1, t2;
    DecomposeEtoRandT(E, R1, R2, t1, t2);
    VAL(t.obs[1].real_R);
    VAL(R1);
    VAL(R2);
    VAL(t1/0.26725966);
    VAL(t2);

    return 0;
}

