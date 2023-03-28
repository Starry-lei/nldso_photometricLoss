#pragma once

#include <ceres/cubic_interpolation.h>
#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include <Eigen/Core>

const size_t PATTERN_SIZE = 8;
const int PATTERN_OFFSETS[PATTERN_SIZE][2] = {{0, 0}, {1, -1}, {-1, 1}, {-1, -1},
                                {2, 0}, {0, 2},  {-2, 0}, {0, -2}};

class PhotometricBundleAdjustment_vanilla {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::unique_ptr<ceres::Grid2D<double, 1> > image_grid;
    std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > > compute_interpolation;
    size_t image_width;
    size_t image_height;
    float fx;
    float fy;
    float cx;
    float cy;
    double x_host_normalized;
    double y_host_normalized;
    std::vector<double> patch_values;
    double mean_patch_value;

    PhotometricBundleAdjustment_vanilla(std::vector<double>& image_vectorized,
                                int img_width, int img_height,
                                std::vector<double>& patch,
                                double x_norm, double y_norm,
                                float fx_new, float fy_new,
                                float cx_new, float cy_new){
        fx = fx_new;
        fy = fy_new;
        cx = cx_new;
        cy = cy_new;

        x_host_normalized = x_norm;
        y_host_normalized = y_norm;

        image_width = img_width;
        image_height = img_height;

        image_grid.reset(new ceres::Grid2D<double, 1>(&image_vectorized[0], 0, image_height, 0, image_width));
        compute_interpolation.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> >(*image_grid));
        patch_values = patch;

        mean_patch_value = 0.0;
        for (auto& p : patch){
            mean_patch_value += p;
        }
        if (!patch.empty()){
            mean_patch_value /= patch.size();
        }

        if (mean_patch_value == 0.0){
            mean_patch_value = 1.0;
        }
    }

    template<typename T>
    bool operator()(const T* const cam, const T* const pidepth, T* presiduals) const {
        Eigen::Map<Eigen::Array<T, PATTERN_SIZE, 1>> residuals(presiduals);
        // cam[0:3] represent rotation
        T quaternion[4] = {cam[0], cam[1], cam[2], cam[3]};
        const T& idepth(*pidepth);

        Eigen::Array<T, PATTERN_SIZE, 1> patch_values_target;
        for (size_t i = 0; i < PATTERN_SIZE; i++){
            int du = PATTERN_OFFSETS[i][0];
            int dv = PATTERN_OFFSETS[i][1];

            T p_host_normalized[3] = {T(x_host_normalized), T(y_host_normalized), T(1.f)};
            p_host_normalized[0] += T(du * 1.0 / fx);
            p_host_normalized[1] += T(dv * 1.0 / fy);

            T p[3];
            ceres::UnitQuaternionRotatePoint(quaternion, p_host_normalized, p);
            // cam[4-6] represent translation
            p[0] += cam[4] * idepth;
            p[1] += cam[5] * idepth;
            p[2] += cam[6] * idepth;

            T u = T(fx)*(p[0] / p[2]) + T(cx);
            T v = T(fy)*(p[1] / p[2]) + T(cy);
//                std::cerr<< "u: " << u << " v: " << v << std::endl;
            compute_interpolation->Evaluate(v, u, &patch_values_target[i]);
        }

        T mean_patch_value_target = patch_values_target.mean();
        for (size_t i = 0; i < PATTERN_SIZE; i++){
            // different approaches to treat a, b
            // I_t - I_h
            // I_t / mean_patch(I_t) - I_h / mean_patch(I_h) (adapt the 0.2-0.5 * 8 per patch Huber norm scale)
            // I_t - mean(I_t) / mean(I_h)*I_h
            residuals[i] =patch_values_target[i] - T(patch_values[i]);
            // Mariia orig: residuals[i] = patch_values_target[i] - (mean_patch_value_target / mean_patch_value)*T(patch_values[i]);

            //residuals[i] = patch_values_target[i] / mean_patch_value_target - T(patch_values[i]) / mean_patch_value;
        }
        return true;
    }

    static ceres::CostFunction* Create(std::vector<double>& image_vectorized, int img_width, int img_height,
                                       std::vector<double>& patch,double xx, double yy, float fx_new, float fy_new,
                                       float cx_new, float cy_new){
        ceres::CostFunction* cost_fun = new ceres::AutoDiffCostFunction<PhotometricBundleAdjustment_vanilla, PATTERN_SIZE, 7, 1>
                (new PhotometricBundleAdjustment_vanilla(image_vectorized, img_width, img_height, patch, xx, yy, fx_new, fy_new, cx_new, cy_new));
        return cost_fun;
    }
};

class PhotometricBundleAdjustment {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::unique_ptr<ceres::Grid2D<double, 1> > image_grid;
    std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > > compute_interpolation;
    size_t image_width;
    size_t image_height;
    float fx;
    float fy;
    float cx;
    float cy;
    double x_host_normalized;
    double y_host_normalized;
    std::vector<double> patch_values;
    std::vector<double> patch_weigts;
    double mean_patch_value;

    PhotometricBundleAdjustment(std::vector<double>& image_vectorized,
                                int img_width, int img_height,
                                std::vector<double>& patch,
                                std::vector<double>& patch_weights,
                                double x_norm, double y_norm,
                                float fx_new, float fy_new,
                                float cx_new, float cy_new){
        fx = fx_new;
        fy = fy_new;
        cx = cx_new;
        cy = cy_new;

        x_host_normalized = x_norm;
        y_host_normalized = y_norm;

        image_width = img_width;
        image_height = img_height;

        image_grid.reset(new ceres::Grid2D<double, 1>(&image_vectorized[0], 0, image_height, 0, image_width));
        compute_interpolation.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> >(*image_grid));
        patch_values = patch;
        patch_weigts = patch_weights;

        mean_patch_value = 0.0;
        for (auto& p : patch){
            mean_patch_value += p;
        }
        if (!patch.empty()){
            mean_patch_value /= patch.size();
        }

        if (mean_patch_value == 0.0){
            mean_patch_value = 1.0;
        }
    }

    template<typename T>
    bool operator()(const T* const cam, const T* const pidepth, T* presiduals) const {
        Eigen::Map<Eigen::Array<T, PATTERN_SIZE, 1>> residuals(presiduals);
        // cam[0:3] represent rotation
        T quaternion[4] = {cam[0], cam[1], cam[2], cam[3]};
        const T& idepth(*pidepth);

        Eigen::Array<T, PATTERN_SIZE, 1> patch_values_target;
        for (size_t i = 0; i < PATTERN_SIZE; i++){
            int du = PATTERN_OFFSETS[i][0];
            int dv = PATTERN_OFFSETS[i][1];

            T p_host_normalized[3] = {T(x_host_normalized), T(y_host_normalized), T(1.f)};
            p_host_normalized[0] += T(du * 1.0 / fx);
            p_host_normalized[1] += T(dv * 1.0 / fy);

            T p[3];
            ceres::UnitQuaternionRotatePoint(quaternion, p_host_normalized, p);
            // cam[4-6] represent translation
            p[0] += cam[4] * idepth;
            p[1] += cam[5] * idepth;
            p[2] += cam[6] * idepth;

            T u = T(fx)*(p[0] / p[2]) + T(cx);
            T v = T(fy)*(p[1] / p[2]) + T(cy);
//                std::cerr<< "u: " << u << " v: " << v << std::endl;
            compute_interpolation->Evaluate(v, u, &patch_values_target[i]);
        }

        T mean_patch_value_target = patch_values_target.mean();
        for (size_t i = 0; i < PATTERN_SIZE; i++){
            // different approaches to treat a, b
            // I_t - I_h
            // I_t / mean_patch(I_t) - I_h / mean_patch(I_h) (adapt the 0.2-0.5 * 8 per patch Huber norm scale)
            // I_t - mean(I_t) / mean(I_h)*I_h
            residuals[i] = T(patch_weigts[i])*(patch_values_target[i] - T(patch_values[i])) ;
            // Mariia : residuals[i] = patch_values_target[i] - (mean_patch_value_target / mean_patch_value)*T(patch_values[i]);

            //residuals[i] = patch_values_target[i] / mean_patch_value_target - T(patch_values[i]) / mean_patch_value;
        }
        return true;
    }

    static ceres::CostFunction* Create(std::vector<double>& image_vectorized, int img_width, int img_height,
                                       std::vector<double>& patch, std::vector<double>& patch_weigts,double xx, double yy, float fx_new, float fy_new,
                                       float cx_new, float cy_new){
        ceres::CostFunction* cost_fun = new ceres::AutoDiffCostFunction<PhotometricBundleAdjustment, PATTERN_SIZE, 7, 1>
                (new PhotometricBundleAdjustment(image_vectorized, img_width, img_height, patch, patch_weigts, xx, yy, fx_new, fy_new, cx_new, cy_new));
        return cost_fun;
    }
};