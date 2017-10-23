#include "ukf.h"
#include "tools.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
        : is_initialized_(false)
        , use_laser_(true)
        , use_radar_(true)
        , n_x_(5)
        , n_aug_(7)
        , n_sigma_(2*n_aug_ + 1)
        , lambda_(3 - n_aug_)
{
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 2;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = M_PI / 4;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    // Matrix of predicted sigma points
    Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

    P_ = MatrixXd::Identity(n_x_, n_x_);
    // initial state vector
    x_ = VectorXd::Zero(5);

    // Weights for calculating new mean from sigma points
    weights_ = VectorXd(n_sigma_);
    weights_.fill(0.5 / (lambda_ + n_aug_));
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    const int dims = 2;

    H_laser_ = MatrixXd(dims, n_x_);
    H_laser_ << 1, 0, 0, 0, 0,
                0, 1, 0, 0, 0;

    R_laser_ = MatrixXd(dims, dims);
    R_laser_ << 0.0225, 0,
                0, 0.0225;

    const int n_z = 3;
    // add measurement noise covariance matrix
    R_radar_ = MatrixXd(n_z, n_z);
    R_radar_ <<    std_radr_ * std_radr_, 0, 0,
                   0, std_radphi_ * std_radphi_, 0,
                   0, 0, std_radrd_ * std_radrd_;

    // identity matrix for laser measurements
    I_ = MatrixXd::Identity(n_x_, n_x_);
}

UKF::~UKF() = default;

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage &meas_package) {
    if (!is_initialized_) {
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { //PolarToCartesian
            double rho = meas_package.raw_measurements_[0];
            double theta = meas_package.raw_measurements_[1];
            x_ << rho * std::cos(theta), rho * std::sin(theta), meas_package.raw_measurements_[2], 0., 0.;
        } else {
            x_ << meas_package.raw_measurements_, 0., 0., 0.;
        }

        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return; // early
    }

    // calculate delta t
    auto delta_t = (meas_package.timestamp_ - time_us_) / 1e6; // microseconds to seconds
    time_us_ = meas_package.timestamp_;

    // predict
    Prediction(delta_t);

    // update
    if (MeasurementPackage::LASER == meas_package.sensor_type_ && use_laser_) {
        UpdateLidar(meas_package);
    } else if (MeasurementPackage::RADAR == meas_package.sensor_type_ && use_radar_) {
        UpdateRadar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    const MatrixXd Xsig = createAugmentedSigmaPoints();
    predictSigmaPoints(Xsig, delta_t);
    x_ = Xsig_pred_* weights_;

    P_ = Tools::covariance(Xsig_pred_, x_, weights_, 3);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage &meas_package) {
    // Standard linear laser measurement update
    const VectorXd z_pred = H_laser_ * x_;
    const VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    const MatrixXd Ht = H_laser_.transpose();
    const MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
    const MatrixXd PHt = P_ * Ht;
    const MatrixXd K = PHt * S.inverse();
    x_ += K * z_diff;
    P_ -= K * H_laser_ * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage &meas_package) {
    const VectorXd& z = meas_package.raw_measurements_; // length 3
    auto n_z = static_cast<int>(z.size());
    const MatrixXd& Zsig = sigmaPointsInMeasurementSpace(n_z);

    // mean predicted measurement
    VectorXd z_pred = Zsig * weights_;

    // measurement covariance matrix S
    MatrixXd S = Tools::covariance(Zsig, z_pred, weights_, 1) + R_radar_;

    // calculate cross correlation matrix
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    for (int i = 0; i < n_sigma_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;    // residual
        z_diff(1) = Tools::normalize(z_diff(1));

        VectorXd x_diff = Xsig_pred_.col(i) - x_;  // state difference
        x_diff(3) = Tools::normalize(x_diff(3));

        Tc +=  weights_(i) * x_diff * z_diff.transpose();
    }

    MatrixXd K = Tc * S.inverse();       // Kalman gain K;

    VectorXd z_diff = z - z_pred;
    z_diff(1) = Tools::normalize(z_diff(1));

    // update state mean and covariance matrix
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
}

MatrixXd UKF::createAugmentedSigmaPoints() const {
    // create augmented covariance matrix
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(P_.rows(), P_.cols()) << P_;
    P_aug.bottomRightCorner(2,2) << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;

    // create augmented mean state
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug << x_, 0, 0;

    MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);
    MatrixXd L = P_aug.llt().matrixL();
    const auto f = sqrt(lambda_ + n_aug_);

    // create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1)          = x_aug + f * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - f * L.col(i);
    }

    return Xsig_aug;
}

void UKF::predictSigmaPoints(const MatrixXd& Xsig_aug, double delta_t) {
    const double dt2 = delta_t * delta_t;

    for (int i = 0; i < n_sigma_; i++) {
        // extract values for better readability
        const double p_x = Xsig_aug(0, i);
        const double p_y = Xsig_aug(1, i);
        const double v = Xsig_aug(2, i);
        const double yaw = Xsig_aug(3, i);
        const double yawd = Xsig_aug(4, i);
        const double nu_a = Xsig_aug(5, i);
        const double nu_yawdd = Xsig_aug(6, i);

        // predicted state values
        double px_p, py_p;

        // avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        // add noise
        px_p = px_p + 0.5 * nu_a * dt2 * cos(yaw);
        py_p = py_p + 0.5 * nu_a * dt2 * sin(yaw);
        v_p += nu_a * delta_t;

        yaw_p += 0.5 * nu_yawdd * dt2;
        yawd_p += nu_yawdd * delta_t;

        // write predicted sigma point into right column
        Xsig_pred_.col(i) << px_p, py_p, v_p, yaw_p, yawd_p;
    }
}


MatrixXd UKF::sigmaPointsInMeasurementSpace(int n_z) {
    MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

    for (int i = 0; i < n_sigma_; ++i) {
        // extract values for better readability
        const double p_x = Xsig_pred_(0, i);
        const double p_y = Xsig_pred_(1, i);
        const double v =   Xsig_pred_(2, i);
        const double yaw = Xsig_pred_(3, i);

        const double v1 = cos(yaw) * v;
        const double v2 = sin(yaw) * v;

        // radar measurement model
        Zsig.col(i) << sqrt(p_x * p_x + p_y * p_y), // rho
                       atan2(p_y, p_x),             // phi
                       (p_x * v1 + p_y * v2) / std::max(sqrt(p_x * p_x + p_y * p_y), 1.0e-5); //rho dot
    }
    return Zsig;
}
