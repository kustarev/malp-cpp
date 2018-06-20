#ifndef malp_matrix_h
#define malp_matrix_h

#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <cmath>

#include "range.h"

typedef std::vector<double> FVector;
typedef std::vector<FVector> FMatrix;

double sum(const FVector& a)
{
    return std::accumulate(a.begin(), a.end(), 0.0);
}

void assert_vector(const FVector& v)
{
    assert(std::isfinite(sum(v)));
}

double mean(const FVector& a)
{
    return sum(a) / a.size();
}


void assert_matrix(const FMatrix& a)
{
    for (auto &i : range(a.size()))
    {
        assert_vector(a[i]);
        assert(a[0].size() == a[i].size());
    }
}

void print(const FVector& v)
{
    for (auto &x : v)
        std::cout << x << " ";
    std::cout << std::endl;
}

void print(const FMatrix& a) {
    for (auto &v : a)
        print(v);
}

double norm(const FVector &v)
{
    assert_vector(v);
    double res = 0.0;
    for (auto &x : v)
        res += std::abs(x);
    return res;
}

double norm(const FMatrix &a)
{
    assert_matrix(a);
    auto m = a.size();
    auto norms = FVector(m, 0.0);
    for (auto i : range(m))
        norms[i] = norm(a[i]);
    return *std::max_element(norms.begin(), norms.end());
}

FMatrix eye(size_t n)
{
    FMatrix res(n, FVector(n, 0.0));
    for (auto i : range(n))
        res[i][i] = 1.0;
    return res;
}

FVector dot(const FMatrix& a, const FVector& b)
{
    assert_matrix(a);
    auto m = a.size();
    auto n = a[0].size();
    assert(b.size() == n);
    auto res = FVector(m, 0.0);
    for (auto i : range(m))
        for (auto j : range(n))
            res[i] += a[i][j] * b[j];
    return res;
}

FVector dot(double c, const FVector& v)
{
    assert_vector(v);
    auto n = v.size();
    auto res = v;
    for (auto i : range(n))
        res[i] *= c;
    return res;
}

FMatrix dot(double c, const FMatrix& a)
{
    assert_matrix(a);
    auto m = a.size();
    auto n = a[0].size();
    auto res = a;
    for (auto i : range(m))
        for (auto j : range(n))
            res[i][j] *= c;
    return res;
}

FVector sum(const FVector& a, const FVector& b)
{
    assert_vector(a);
    assert_vector(b);
    auto n = a.size();
    assert(n == b.size());
    auto res = FVector(n, 0.0);
    for (auto i : range(n))
        res[i] = a[i] + b[i];
    return res;
}

FVector prod(const FVector& a, const FVector& b)
{
    assert_vector(a);
    assert_vector(b);
    auto n = a.size();
    assert(n == b.size());
    auto res = FVector(n, 0.0);
    for (auto i : range(n))
        res[i] = a[i] * b[i];
    return res;
}

double corr(const FVector& a, const FVector& b)
{
    auto n = a.size();
    assert(n == b.size());
    auto a_mean = mean(a);
    auto b_mean = mean(b);
    FVector am = a;
    FVector bm = b;
    for (auto i : range(n))
    {
        am[i] -= a_mean;
        bm[i] -= b_mean;
    }
    return sum(prod(am, bm)) / std::sqrt(sum(prod(am, am)) * sum(prod(bm, bm)));
}

FVector diff(const FVector& a, const FVector& b)
{
    return sum(a, dot(-1.0, b));
}

FMatrix sum(const FMatrix& a, const FMatrix& b)
{
    assert_matrix(a);
    assert_matrix(b);
    auto m = a.size();
    auto n = a[0].size();
    assert(b.size() == m);
    assert(b[0].size() == n);
    auto res = a;
    for (auto i : range(m))
        for (auto j : range(n))
            res[i][j] += b[i][j];
    return res;
}

FMatrix diff(const FMatrix& a, const FMatrix& b)
{
    return sum(a, dot(-1.0, b));
}

FMatrix dot(const FMatrix& a, const FMatrix& b)
{
    assert_matrix(a);
    assert_matrix(b);
    auto m = a.size();
    auto n = a[0].size();
    assert(b.size() == n);
    auto k = b[0].size();
    auto res = FMatrix(m, FVector(k, 0.0));
    for (auto i : range(m))
        for (auto j : range(n))
            for (auto l : range(k))
                res[i][l] += a[i][j] * b[j][l];
    return res;
}

FMatrix transpose(const FMatrix &a)
{
    assert_matrix(a);
    auto m = a.size();
    auto n = a[0].size();
    auto res = FMatrix(n, FVector(m, 0.0));
    for (auto i : range(m))
        for (auto j : range(n))
            res[j][i] = a[i][j];
    return res;
}

bool are_close(const FMatrix &a, const FMatrix &b, double eps)
{
    auto rel_diff = norm(diff(a, b)) / std::min(norm(a), norm(b));
    return rel_diff < eps;
}

FMatrix pinv(const FMatrix &a, double eps = 1e-3, double pinv_alpha = 0.5)
{
    assert(eps > 0.0);
    assert(pinv_alpha > 0.0);
    assert(pinv_alpha < 1.0);
    auto m = a.size();
    auto n = a[0].size();
    auto aT = transpose(a);
    if (m < n)
        return transpose(pinv(aT, eps, pinv_alpha));
    auto aTa = dot(aT, a);
    auto aaT = dot(a, aT);
    auto aTa_norm = std::min(norm(aTa), norm(transpose(aTa)));
    auto aaT_norm = std::min(norm(aaT), norm(transpose(aaT)));
    auto alpha = pinv_alpha * 2.0 / std::min(aTa_norm, aaT_norm);
    auto res = dot(alpha, aT);
    while (!are_close(dot(res, a), eye(n), eps))
        res = diff(dot(2.0, res), dot(res, dot(a, res)));
    return res;
}

FVector conjgrad(const FMatrix &a, const FVector& b, double eps = 1e-3)
{
    auto m = a.size();
    auto n = a[0].size();
    assert(m >= n);
    assert(b.size() == m);
    auto aT = transpose(a);
    auto aTa = dot(aT, a);
    auto aTb = dot(aT, b);
    auto x = FVector(n, 0.0);
    auto r = aTb;
    auto init_r_norm = norm(r);
    auto p = r;
    while (norm(r) / init_r_norm > eps)
    {
        auto alpha = sum(prod(r, r)) / sum(prod(p, dot(aTa, p)));
        x = sum(x, dot(alpha, p));
        auto old_r_dot = sum(prod(r, r));
        r = diff(r, dot(alpha, dot(aTa, p)));
        auto new_r_dot = sum(prod(r, r));
        auto beta = new_r_dot / old_r_dot;
        p = sum(r, dot(beta, p));
    }
    return x;
}

void test_pinv()
{
    FMatrix a = { { 2.0, 1.0 }, { 1.0, 1.0 } };
    auto res = pinv(a);
    FMatrix expected = { { 1.0, -1.0 }, { -1.0, 2.0 } };
    auto tolerance = 1e-3;
    assert(are_close(res, expected, tolerance));
}

#endif
