#ifndef malp_convex_task_h
#define malp_convex_task_h

#include "matrix.h"

typedef std::vector<size_t> IdxVector;
typedef std::vector<IdxVector> IdxMatrix;

class Task {
    size_t max_iter_;
    double step_;
    bool verbose_;
    size_t iter_start_;
    double iter_exp_;
    
    FVector bounds_;
    FVector coeffs_;
    FMatrix constraints_;
    IdxMatrix cons_idx_;
    
    FMatrix A_;
    FVector B_;
    FVector C_;
    FMatrix pinvA_;
    FVector yC_;
    
    void assert_idx_(size_t idx)
    {
        assert(idx >= 0);
        assert(idx <= coeffs_.size());
    }
    
    
    
public:
    Task(size_t max_iter = 100, double step = 1.0, bool verbose = true, size_t iter_start = 0, double iter_exp = 1.0) :
    max_iter_(max_iter),
    step_(step),
    verbose_(verbose),
    iter_start_(iter_start),
    iter_exp_(iter_exp) {}
    
    size_t new_variable()
    {
        coeffs_.push_back(0.0);
        return coeffs_.size() - 1;
    }
    
    void add_constraint(const IdxVector& idx_list, const FVector& coeff_list, double bound)
    {
        for (auto idx : idx_list)
            assert_idx_(idx);
        assert(idx_list.size() == coeff_list.size());
        bounds_.push_back(bound);
        constraints_.push_back(coeff_list);
        cons_idx_.push_back(idx_list);
    }
    
    void add_coeff(size_t idx, double val)
    {
        assert_idx_(idx);
        coeffs_[idx] = val;
    }
    
    void set_task_()
    {
        C_ = coeffs_;
        B_ = bounds_;
        auto m = bounds_.size();
        auto n = coeffs_.size();
        A_ = FMatrix(m, FVector(n, 0.0));
        for (auto i : range(m))
        {
            auto& coeff_list = constraints_[i];
            auto& idx_list = cons_idx_[i];
            auto sz = coeff_list.size();
            assert(sz == idx_list.size());
            for (auto j : range(sz))
                A_[i][idx_list[j]] = coeff_list[j];
        }
        pinvA_ = pinv(A_);
        yC_ = dot(transpose(pinvA_), C_);
    }
    
    static bool feasible_(const FVector& y)
    {
        return (*std::min_element(y.begin(), y.end()) > 0);
    }
    
    FVector y_(const FVector& x)
    {
        return sum(dot(A_, x), B_);
    }
    
    FVector x_(const FVector& y)
    {
        return dot(pinvA_, diff(y, B_));
    }
    
    FVector gradient_(const FVector& z, size_t iter)
    {
        auto m = z.size();
        assert(m == B_.size());
        // compute gradient of the corresponding quadratic function in global Euclidean space
        auto zC = yC_;
        for (auto i : range(m))
            zC[i] *= z[i];
        // compute iteration decay multiplier (yields to 0 as iteration number increases)
        auto iter_decay = 1.0 / (iter_start_ + std::powf(iter, iter_exp_));
        // update the gradient to stay away from boundaries
        for (auto i : range(m))
            zC[i] += iter_decay / z[i];
        // get manifold tangent space in the current point
        auto diag_invZ = eye(m);
        for (auto i : range(m))
            diag_invZ[i][i] = 1.0 / z[i];
        auto tg = dot(transpose(A_), diag_invZ);
        auto tgT = transpose(tg);
        // project the gradient onto the tangent space
        auto grad_coeffs = conjgrad(tgT, zC);
        auto grad_z = dot(tgT, grad_coeffs);
        // map the gradient back onto the polytope affine image
        auto grad_y = grad_z;
        for (auto i : range(m))
            grad_y[i] *= z[i];
        return grad_y;
    }
    
    FVector solve(const FVector& start)
    {
        set_task_();
        auto m = B_.size();
        auto n = C_.size();
        auto x = start;
        assert(x.size() == n);
        auto y = y_(x);
        assert (y.size() == m);
        size_t iterations = 1;
        while (true)
        {
            auto z = FVector(m, 0.0);
            for (auto i : range(m))
                z[i] = std::sqrt(y[i]);
            auto new_y = sum(y, dot(step_, gradient_(z, iterations)));
            assert(feasible_(new_y));
            if (verbose_)
            {
                std::cout << "Iteration " << iterations << std::endl;
                std::cout << "Solution vector" << std::endl;
                print(x_(y));
            }
            if (iterations == max_iter_)
                break;
            iterations++;
            y = new_y;
        }
        return x_(y);
    }
    
};


#endif
