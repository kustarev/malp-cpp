#include "convex_task.h"

enum class TASKS { TASK_SIMPLEX, TASK_MAXRET_PORTFOLIO };
auto RUN = TASKS::TASK_SIMPLEX;

void run_task_simplex()
{
    size_t max_iter = 1000;
    double step = 0.01;
    bool verbose = true;
    size_t start_iter = 0;
    double iter_exp = 3.0;
    
    auto t = Task(max_iter, step, verbose, start_iter, iter_exp);
    
    size_t dim = 100;
    FVector coeffs(dim, 0.0);
    for (auto i : range(dim))
        coeffs[i] = i + 1.0;
    
    for (auto i : range(dim))
    {
        t.new_variable();
        t.add_constraint(IdxVector({i}), FVector({1.0}), 0.0);
        t.add_coeff(i, coeffs[i]);
    }
    t.add_constraint(range(dim).vector(), FVector(dim, -1.0), 1.0);
    FVector start(dim, 1.0 / (1.0 + dim));
    auto res = t.solve(start);
    print(res);
    std::cout << "Verifying optimized versus expected solution.." << std::endl;
    auto atol = 1e-3;
    assert(std::abs(res[dim - 1] - 1.0) < atol);
    for (auto i : range(dim - 1))
        assert(std::abs(res[i]) < atol);
    std::cout << "done" << std::endl;
}

void run_task_maxret_portfolio()
{
    size_t max_iter = 1000;
    double step = 1.0;
    bool verbose = true;
    size_t iter_start = 1e+6;
    double iter_exp = 3.0;
    auto t = Task(max_iter, step, verbose, iter_start, iter_exp);

    size_t dim = 100;
    double gmv = 1.0;
    size_t target_num = 10;
    double max_abs_pos = gmv / target_num;
    auto ideal_pos = FVector(dim, 0);
    for (auto i : range(dim))
        ideal_pos[i] = double(rand()) / RAND_MAX;
    std::sort(ideal_pos.begin(), ideal_pos.end());
    
    IdxVector pos_list;
    IdxVector abs_pos_list;
    FVector optim_start;
    for (auto i : range(dim))
    {
        auto pos_idx = t.new_variable();
        pos_list.push_back(pos_idx);
        optim_start.push_back(0.0);
        auto abs_pos_idx = t.new_variable();
        optim_start.push_back(0.5 * gmv / dim);
        abs_pos_list.push_back(abs_pos_idx);
        t.add_constraint(IdxVector({abs_pos_idx, pos_idx}), FVector({1.0, -1.0}), 0.0);
        t.add_constraint(IdxVector({abs_pos_idx, pos_idx}), FVector({1.0, 1.0}), 0.0);
        t.add_constraint(IdxVector({abs_pos_idx}), FVector({-1.0}), max_abs_pos);
        t.add_coeff(pos_idx, ideal_pos[i]);
    }
    t.add_constraint(abs_pos_list, FVector(dim, -1.0), gmv);
    auto res = t.solve(optim_start);
    auto result_pos = FVector(dim, 0.0);
    assert(pos_list.size() == dim);
    for (auto i : range(dim))
        result_pos[i] = res[pos_list[i]];
    std::cout << "Verifying optimized versus expected solution.." << std::endl;
    auto atol = 1e-3;
    for (auto i : range(dim))
    {
        double expected = (i < dim - target_num) ? 0.0 : max_abs_pos;
        assert(std::abs(expected - result_pos[i]) < atol);
    }
    std::cout << "done" << std::endl;
}

int main(int argc, const char * argv[]) {
    srand(unsigned(0));
    test_pinv();
    if (RUN == TASKS::TASK_SIMPLEX)
        run_task_simplex();
    else if (RUN == TASKS::TASK_MAXRET_PORTFOLIO)
        run_task_maxret_portfolio();
    else
        return -1;
    return 0;
}
