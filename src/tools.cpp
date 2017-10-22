#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

namespace Tools {

    VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                           const vector<VectorXd> &ground_truth) {
        VectorXd rmse(4);
        rmse << 0, 0, 0, 0;

        // check the validity of the following inputs:
        //  * the estimation vector size should not be zero
        //  * the estimation vector size should equal ground truth vector size
        if (estimations.size() != ground_truth.size()
            || estimations.empty()) {
            cout << "Invalid estimation or ground_truth data" << endl;
            return rmse;
        }

        //accumulate squared residuals
        for (unsigned int i = 0; i < estimations.size(); ++i) {
            VectorXd residual = estimations[i] - ground_truth[i];
            residual = residual.array() * residual.array();
            rmse += residual;
        }
        rmse = rmse / estimations.size(); //calculate the mean
        rmse = rmse.array().sqrt(); //calculate the squared root
        return rmse;
    }

    double normalize(double angle_rad) {
        return atan2(sin(angle_rad), cos(angle_rad));
    }


    MatrixXd covariance(
            const MatrixXd& XSig,
            const VectorXd& x,
            const VectorXd& w,
            int   norm_idx)
    {
        MatrixXd S = MatrixXd::Zero(XSig.rows(), XSig.rows());

        for (int i = 0; i < XSig.cols(); ++i)
        {
            VectorXd x_diff = XSig.col(i) - x;
            x_diff(norm_idx) = Tools::normalize(x_diff(norm_idx));
            S += w(i) * x_diff * x_diff.transpose();
        }

        return S;
    }
}