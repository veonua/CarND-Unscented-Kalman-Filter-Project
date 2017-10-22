#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

namespace Tools {
    /**
    * A helper method to calculate RMSE.
    */
    VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

    double normalize(double angle_rad);

    MatrixXd covariance(
            const MatrixXd& XSig,
            const VectorXd& x,
            const VectorXd& w,
            int norm_idx);
}
#endif /* TOOLS_H_ */
