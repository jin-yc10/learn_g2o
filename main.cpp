#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "sample.h"

#define N_PTS 21

DEFINE_int32(obs_number, 100, "observation count");

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <iostream>

#include "g2o/stuff/command_args.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"

#include "EXTERNAL/ceres/autodiff.h"

#if defined G2O_HAVE_CHOLMOD
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#elif defined G2O_HAVE_CSPARSE
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#endif

#define NO_2D_NOISE 1
#define NO_3D_NOISE 1

DEFINE_bool(3D_noise, false, "add noise to 3d pose");
DEFINE_bool(2D_noise, false, "add noise to 2d observation");
DEFINE_bool(Cam_noise, false, "add noise to camera intrinsics");

using namespace g2o;
using namespace std;
using namespace Eigen;

// rx, ry, rz, tx, ty, tz, f, k1, k2
class VertexCameraBAL : public BaseVertex<7, Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() { }

    virtual bool read(std::istream& /*is*/) {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    virtual bool write(std::ostream& /*os*/) const {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    virtual void setToOriginImpl() {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
    }

    virtual void oplusImpl(const double* update) {
        Eigen::VectorXd::ConstMapType v(update, VertexCameraBAL::Dimension);
        _estimate += v;
    }
};

// x,y,z
class VertexPointBAL : public BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {
    }

    virtual bool read(std::istream& /*is*/) {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    virtual bool write(std::ostream& /*os*/) const {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    virtual void setToOriginImpl() {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
    }

    virtual void oplusImpl(const double* update) {
        Eigen::Vector3d::ConstMapType v(update);
        _estimate += v;
    }
};

/**
 * \brief edge representing the observation of a world feature by a camera
 *
 * see: http://grail.cs.washington.edu/projects/bal/
 * We use a pinhole camera model; the parameters we estimate for each camera
 * area rotation R, a translation t, a focal length f and two radial distortion
 * parameters k1 and k2. The formula for projecting a 3D point X into a camera
 * R,t,f,k1,k2 is:
 * P  =  R * X + t       (conversion from world to camera coordinates)
 * p  = -P / P.z         (perspective division)
 * p' =  f * r(p) * p    (conversion to pixel coordinates) where P.z is the third (z) coordinate of P.
 *
 * In the last equation, r(p) is a function that computes a scaling factor to undo the radial
 * distortion:
 * r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
 *
 * This gives a projection in pixels, where the origin of the image is the
 * center of the image, the positive x-axis points right, and the positive
 * y-axis points up (in addition, in the camera coordinate system, the positive
 * z-axis points backwards, so the camera is looking down the negative z-axis,
 * as in OpenGL).
 */
class EdgeObservationBAL : public BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() { }
    virtual bool read(std::istream& /*is*/) {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }
    virtual bool write(std::ostream& /*os*/) const {
        cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
        return false;
    }

    template<typename T>
    inline void cross(const T x[3], const T y[3], T result[3]) const {
        result[0] = x[1] * y[2] - x[2] * y[1];
        result[1] = x[2] * y[0] - x[0] * y[2];
        result[2] = x[0] * y[1] - x[1] * y[0];
    }

    template<typename T>
    inline T dot(const T x[3], const T y[3]) const { return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]); }

    template<typename T>
    inline T squaredNorm(const T x[3]) const { return dot<T>(x, x); }

    /**
     * templatized function to compute the error as described in the comment above
     */
    template<typename T>
    bool operator()(const T* camera, const T* point, T* error) const {
        // Rodrigues' formula for the rotation
        // v_rot = v dot cos(th) + (k cross v) sin th + k(k dot v)(1-cos th)
        T p[3];
        T theta = sqrt(squaredNorm(camera));
        if (theta > T(0)) {
            T v[3];
            v[0] = camera[0] / theta;
            v[1] = camera[1] / theta;
            v[2] = camera[2] / theta;
            T cth = cos(theta);
            T sth = sin(theta);

            T vXp[3];
            cross(v, point, vXp);
            T vDotp = dot(v, point);
            T oneMinusCth = T(1) - cth;

            for (int i = 0; i < 3; ++i)
                p[i] = point[i] * cth + vXp[i] * sth + v[i] * vDotp * oneMinusCth;
        } else {
            // taylor expansion for theta close to zero
            T aux[3];
            cross(camera, point, aux);
            for (int i = 0; i < 3; ++i)
                p[i] = point[i] + aux[i];
        }

        // translation of the camera
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];



        // conversion to pixel coordinates
//        T radiusSqr = projectedPoint[0]*projectedPoint[0] + projectedPoint[1]*projectedPoint[1];
        T f         = T(camera[6]);
        T cx        = T(camera[7]);
        T cy        = T(camera[8]);
        T r_p       = T(1);// + k1 * radiusSqr + k2 * radiusSqr * radiusSqr;
        // perspective division
        T projectedPoint[2];
        projectedPoint[0] = p[0] / p[2];
        projectedPoint[1] = p[1] / p[2];
        // prediction
        T prediction[2];
        prediction[0] = f * projectedPoint[0] + T(320);
        prediction[1] = f * projectedPoint[1] + T(240);

        error[0] = prediction[0] - T(measurement()(0));
        error[1] = prediction[1] - T(measurement()(1));

        return true;
    }

    void computeError() {
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*>(vertex(0));
        const VertexPointBAL* point = static_cast<const VertexPointBAL*>(vertex(1));

        (*this)(cam->estimate().data(), point->estimate().data(), _error.data());
    }

    void linearizeOplus() {
        // use numeric Jacobians
        // BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
        // return;

        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*>(vertex(0));
        const VertexPointBAL* point = static_cast<const VertexPointBAL*>(vertex(1));
        typedef ceres::internal::AutoDiff<
                EdgeObservationBAL,
                double,
                VertexCameraBAL::Dimension,
                VertexPointBAL::Dimension> BalAutoDiff;

        Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera;
        Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint;
        double *parameters[] = { const_cast<double*>(cam->estimate().data()), const_cast<double*>(point->estimate().data()) };
        double *jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };
        double value[Dimension];
        bool diffState = BalAutoDiff::Differentiate(*this, parameters, Dimension, value, jacobians);

        // copy over the Jacobians (convert row-major -> column-major)
        if (diffState) {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        } else {
            assert(0 && "Error while differentiating");
            _jacobianOplusXi.setZero();
            _jacobianOplusXi.setZero();
        }
    }
};

class observation {

public:
    // 2d pts
    float pts_2d[2*N_PTS];
    vector<cv::Point2f> vpts;
    // camera view arguments
    float observation_cam[6]; // r1 r2 r3 t1 t2 t3
    float real_cam[6];
};

class Problem {

    // known arguments
    float mean_pose[3*21];
    cv::Mat MeanPoseMat;

// Problem arguments
    // camera arguments
    float intrinsics[5] = {800.0f, 320.0f, 240.0f, 0.0f, 0.0f}; // focal, principle x, y, k1, k2
    float noisy_intrinsic[5] = {1000.0f, 320.0f, 240.0f, 0.0f, 0.0f};
//            {810.123f, 320.0f, 240.0f, 0.0f, 0.0f};
    float noisy_pose[3*21]; // rough assumption
    cv::Mat NoisePoseMat;

    float raw_k[9] = {
            intrinsics[0], 0.0, intrinsics[1],
            0.0, intrinsics[0], intrinsics[2],
            0.0, 0.0,           1.0f,
    };

    cv::Mat K;
    cv::Mat dist;

    float noisy_raw_k[9] = {
            noisy_intrinsic[0], 0.0, intrinsics[1],
            0.0, noisy_intrinsic[0]+12.7f, intrinsics[2],
            0.0, 0.0,           1.0f,
    };
    cv::Mat noisy_K;
    // observations
    std::vector<observation> obs;

    void fill(observation& ob) {
        static float dt[3] = {-300.0f, 0.0, 1000.0f};
//        dt[2] += Sample::uniform(-30, 30);
        dt[0] += 6.0f;

        static float dr[3] = {0.0, -3.1415926*0.5, 0.0}; // Rodrigues form
        dr[1] += 3.1415926f * 0.01f;

        cv::Mat R, rvec(1,3,CV_32F,dr);
        cv::Rodrigues(rvec, R);
        cv::Mat t = cv::Mat(3,1,CV_32F,dt);
        cv::repeat(t, 1, 21, t);
        cv::Mat project = K*(R*MeanPoseMat.t()+t);

        for(int i=0; i<21; i++) {
            cv::Point2f pt;
            if(FLAGS_2D_noise) {
                pt.x = project.at<float>(0,i)/project.at<float>(2,i) + Sample::uniform(-2, 2);
                pt.y = project.at<float>(1,i)/project.at<float>(2,i) + Sample::uniform(-2, 2);
            } else {
                pt.x = project.at<float>(0,i)/project.at<float>(2,i);// + Sample::uniform(-2, 2);
                pt.y = project.at<float>(1,i)/project.at<float>(2,i);// + Sample::uniform(-2, 2);
            }
            ob.pts_2d[i*2+0] = pt.x;
            ob.pts_2d[i*2+1] = pt.y;
            ob.vpts.push_back(pt);
        }
        LOG(WARNING) << ob.pts_2d[17]-ob.pts_2d[16];

        memcpy(ob.real_cam, dr, 3*sizeof(float));
        memcpy(ob.real_cam+3, dt, 3*sizeof(float));

        static cv::Mat raux, taux;
        static bool useExtrinsicGuess = false;

        cv::solvePnP(NoisePoseMat, ob.vpts,
                     noisy_K,
                     dist,
                     raux, taux, useExtrinsicGuess, CV_ITERATIVE);

        if (!useExtrinsicGuess) {
            useExtrinsicGuess = true;
        }
        cv::Mat pnp_rvec;
        cv::Mat pnp_tvec;

        // solvePnP get [double] result, convert to float
        raux.convertTo(pnp_rvec, CV_32F);
        taux.convertTo(pnp_tvec, CV_32F);

        std::cout << rvec << "\t" << t.col(0).t() << "\t" <<
                pnp_rvec.t() << "\t" << pnp_tvec.t() << std::endl;

        // fill the solvePnP data
        memcpy(ob.observation_cam, pnp_rvec.data, 3*sizeof(float));
        memcpy(ob.observation_cam+3, pnp_tvec.data, 3*sizeof(float));
    }

    void show(observation ob) {
        cv::Mat img(480, 640, CV_8UC3);
        img.setTo(cv::Scalar(30,30,30));
        for(int i=0; i<21; i++) {
            cv::circle(img, cv::Point2f( ob.pts_2d[i*2+0], ob.pts_2d[i*2+1]), 2, cv::Scalar(0,255,0), 1);
        }
        cv::imshow("ob", img);
    }

public:
    Problem(std::string file_path) :
            K(3,3,CV_32F,raw_k),
            dist(1,5,CV_32F) {
        std::ifstream ifs(file_path);
        if(FLAGS_Cam_noise) {
            noisy_K = cv::Mat(3,3,CV_32F,noisy_raw_k);
        } else {
            noisy_K = cv::Mat(3,3,CV_32F,raw_k); // use the real camera intrinsic
        }
        string outputFilename = "mean.wrl";
        if (outputFilename.size() > 0) {
            ofstream fout(outputFilename.c_str()); // loadable with meshlab
            fout
            << "#VRML V2.0 utf8\n"
            << "Shape {\n"
            << "  appearance Appearance {\n"
            << "    material Material {\n"
            << "      diffuseColor " << 0 << " " << 1 << " " << 0 << "\n"
            << "      ambientIntensity 0.2\n"
            << "      emissiveColor 0.0 0.0 0.0\n"
            << "      specularColor 0.0 0.0 0.0\n"
            << "      shininess 0.2\n"
            << "      transparency 0.0\n"
            << "    }\n"
            << "  }\n"
            << "  geometry PointSet {\n"
            << "    coord Coordinate {\n"
            << "      point [\n";
            for(int i=0; i<21; i++) {
                ifs >> mean_pose[3*i+0]
                >> mean_pose[3*i+1]
                >> mean_pose[3*i+2];
                fout << mean_pose[3*i+0] << " " << mean_pose[3*i+1] << " " << mean_pose[3*i+2] << endl;
                // add some noise to the mean pose
                for(int j=0; j<3; j++)
                    noisy_pose[3*i+j] = mean_pose[3*i+j];
                if( FLAGS_3D_noise ) {
                    int j=2;
                    noisy_pose[3*i+j] = mean_pose[3*i+j] + Sample::uniform(-1,1);
                }
            }
            fout << "    ]\n" << "  }\n" << "}\n" << "  }\n";
            fout.flush();
            fout.close();
        }

        MeanPoseMat = cv::Mat(21, 3, CV_32F, mean_pose);
        NoisePoseMat = cv::Mat(21, 3, CV_32F, noisy_pose);
        std::cout << MeanPoseMat << std::endl << NoisePoseMat << std::endl;

        LOG(WARNING) << "FLAGS_obs_number = " << FLAGS_obs_number;
        std::cout << "noisy_K=" << noisy_K << std::endl;
        dist.setTo(0);
        std::cout << "dist=" << dist << std::endl;

        for(int i=0; i<FLAGS_obs_number; i++) {
            observation ob;
            fill(ob);
//            show(ob);
//            cv::waitKey(100);
            obs.push_back(ob);
        }
    }

    void solve() {

        int maxIterations = 20;
        typedef g2o::BlockSolver< g2o::BlockSolverTraits<7, 3> >  BalBlockSolver;
        string choleskySolverName = "CHOLMOD";
        typedef g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType> BalLinearSolver;
        g2o::SparseOptimizer optimizer;
        g2o::LinearSolver<BalBlockSolver::PoseMatrixType>* linearSolver = 0;
        cout << "Using Cholesky: " << choleskySolverName << endl;
        BalLinearSolver* cholesky = new BalLinearSolver();
        cholesky->setBlockOrdering(true);
        linearSolver = cholesky;
        BalBlockSolver* solver_ptr = new BalBlockSolver(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        LOG(WARNING) << "Init Solver Done.";

        vector<VertexPointBAL*> points;
        vector<VertexCameraBAL*> cameras;

        int numCameras = obs.size();
        int numPoints = N_PTS;
        int numObservations = obs.size()*N_PTS; // currently we have 21 observation in all case

        int id = 0;
        cameras.reserve(numCameras);
        for (int i = 0; i < numCameras; ++i, ++id) {
            VertexCameraBAL* cam = new VertexCameraBAL;
            cam->setId(id);
            optimizer.addVertex(cam);
            cameras.push_back(cam);
        }
        LOG(WARNING) << "Add cameras done.";

        points.reserve(numPoints);
        for (int i = 0; i < numPoints; ++i, ++id) {
            VertexPointBAL* p = new VertexPointBAL;
            p->setId(id);
            p->setMarginalized(true);
            bool addedVertex = optimizer.addVertex(p);
            if (! addedVertex) {
                cerr << "failing adding vertex" << endl;
            }
            points.push_back(p);
        }
        LOG(WARNING) << "Add points done.";

        for (int i = 0; i < numObservations; ++i) {

            int camIndex, pointIndex;
            double obsX, obsY;
            camIndex = i/21;
            pointIndex = i%21;

            obsX = obs[camIndex].pts_2d[pointIndex*2+0];
            obsY = obs[camIndex].pts_2d[pointIndex*2+1];

            VertexCameraBAL* cam = cameras[camIndex];
            VertexPointBAL* point = points[pointIndex];
//            std::cout << i << "\t" << cam->id() << "\t" << point->id() << std::endl;

            EdgeObservationBAL* e = new EdgeObservationBAL;
            e->setVertex(0, cam);
            e->setVertex(1, point);
            e->setInformation(Eigen::Matrix2d::Identity());
            e->setMeasurement(Eigen::Vector2d(obsX, obsY));
            bool addedEdge = optimizer.addEdge(e);
            if (! addedEdge) {
                cerr << "error adding edge" << endl;
            }
        }
        LOG(WARNING) << "Add Edges done.";

        // fill the estimation
        Eigen::VectorXd cameraParameter(7);
        for (int i = 0; i < numCameras; ++i) {
            for(int j=0; j<6; j++) {
                cameraParameter(j) = obs[i].observation_cam[j];
            }
            cameraParameter(6) = intrinsics[0];
            // cameraParameter(7) = noisy_intrinsic[1];
            // cameraParameter(8) = noisy_intrinsic[2];

            VertexCameraBAL* cam = cameras[i];
            cam->setEstimate(cameraParameter);
            if( i==0 ) {
                cout << "Camera #0 parameters: " << endl;
                for (int j = 0; j < 7; ++j)
                    cout << cameraParameter(j) << " ";
            }
        }
        Eigen::Vector3d p;
        for (int i = 0; i < numPoints; ++i) {
            for(int j=0 ; j<3; j++) {
                p(j) = mean_pose[3*i+j];
            }
            VertexPointBAL* point = points[i];
            point->setEstimate(p);
        }
        LOG(WARNING) << "Fill Estimations done.";
        cout << "Build BA problem done." << endl;

        cout << "Initializing ... " << flush;
        optimizer.initializeOptimization();
        cout << "done." << endl;
        optimizer.setVerbose(true);
        cout << "Start to optimize" << endl;
        optimizer.optimize(maxIterations);

        std::cout << "Ground Truth\tNoisy\tOptimize" << std::endl;
        for (int i=0; i<21; i++) {
            cout << MeanPoseMat.row(i) << "\t" <<
                    NoisePoseMat.row(i) << "\t" <<
                    points[i]->estimate().transpose() << endl;
        }
        string outputFilename = "pose.wrl";
        if (outputFilename.size() > 0) {
            ofstream fout(outputFilename.c_str()); // loadable with meshlab
            fout
            << "#VRML V2.0 utf8\n"
            << "Shape {\n"
            << "  appearance Appearance {\n"
            << "    material Material {\n"
            << "      diffuseColor " << 1 << " " << 0 << " " << 0 << "\n"
            << "      ambientIntensity 0.2\n"
            << "      emissiveColor 0.0 0.0 0.0\n"
            << "      specularColor 0.0 0.0 0.0\n"
            << "      shininess 0.2\n"
            << "      transparency 0.0\n"
            << "    }\n"
            << "  }\n"
            << "  geometry PointSet {\n"
            << "    coord Coordinate {\n"
            << "      point [\n";
            for (vector<VertexPointBAL*>::const_iterator it = points.begin(); it != points.end(); ++it) {
                fout << (*it)->estimate().transpose() << endl;
            }
            fout << "    ]\n" << "  }\n" << "}\n" << "  }\n";
            fout.flush();
            fout.close();
        }
        int idx = 0;
        for(auto cam_it=cameras.begin();
            cam_it != cameras.end();
            cam_it ++){

            cout << "/ ";
            for (int j = 0; j < 7; ++j) {
                cout << std::setw(12) << (*cam_it)->estimate()[j] << " ";
            }
            cout << "\n| ";
            for (int j = 0; j < 6; ++j) {
                cout << std::setw(12) << obs[idx].observation_cam[j] << " ";
            }
            cout << "\n\\ ";
            for (int j = 0; j < 6; ++j) {
                cout << std::setw(12) << obs[idx].real_cam[j] << " ";
            }
            cout << std::setw(12) << "800";
            cout << endl;
            idx ++;
        }
    }

    void summary() {
        std::cout << " ========== REPORT ========== " << std::endl;
    }
};

int main(int argc, char** argv) {
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);
    Problem problem("../data/mean.txt");
    problem.solve();
    problem.summary();
    return 0;
}