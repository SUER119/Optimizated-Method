#pragma once

#include <pybind11/pybind11.h>

#include <Math/CSR_MATRIX.h>

#ifdef CHOLMOD_DIRECT_SOLVER
#include "cholmod.h"
#endif

namespace JGSL {

template <class T>
bool Solve_Direct(
    CSR_MATRIX<T> &sysMtr, 
    const std::vector<T> &rhs, 
    std::vector<T> &sol) 
{
    if (sysMtr.Get_Matrix().rows() != rhs.size()) {
        printf("sysMtr dimension does not match with rhs!\n");
        return false;
    }

#ifdef CHOLMOD_DIRECT_SOLVER
    cholmod_common cm;
    cholmod_start(&cm);

    // setup matrix
    cholmod_sparse* A = cholmod_allocate_sparse(sysMtr.Get_Matrix().rows(), 
        sysMtr.Get_Matrix().cols(), sysMtr.Get_Matrix().nonZeros(), 
        true, true, -1, CHOLMOD_REAL, &cm);
    void *Ax = A->x;
    void *Ap = A->p;
    void *Ai = A->i;
    A->i = sysMtr.Get_Matrix().innerIndexPtr();
    A->p = sysMtr.Get_Matrix().outerIndexPtr();
    A->x = sysMtr.Get_Matrix().valuePtr();

    // factorization
    cholmod_factor* L = cholmod_analyze(A, &cm);
    cholmod_factorize(A, L, &cm);
    if (cm.status == CHOLMOD_NOT_POSDEF) {
        return false;
    }

    // back solve
    cholmod_dense *b = cholmod_allocate_dense(rhs.size(), 1, rhs.size(), CHOLMOD_REAL, &cm);
    void *bx = b->x;
    b->x = const_cast<T*>(rhs.data());
    cholmod_dense* x = cholmod_solve(CHOLMOD_A, L, b, &cm);
    sol.resize(rhs.size());
    std::memcpy(sol.data(), x->x, sol.size() * sizeof(T));
    cholmod_free_dense(&x, &cm);

    // free memory
    A->i = Ai;
    A->p = Ap;
    A->x = Ax;
    cholmod_free_sparse(&A, &cm);
    cholmod_free_factor(&L, &cm);
    b->x = bx;
    cholmod_free_dense(&b, &cm);
    cholmod_finish(&cm);

    return true;
#else
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solver;
    solver.compute(sysMtr.Get_Matrix());
    if (solver.info() != Eigen::Success) {
        printf("Eigen::SimplicialLDLT factorization failed\n");
        return false;
    }
    else {
        Eigen::VectorXd rhsE(rhs.size());
        std::memcpy(rhsE.data(), rhs.data(), sizeof(T) * rhs.size());
        Eigen::VectorXd solE = solver.solve(rhsE);
        if (solver.info() != Eigen::Success) {
            printf("Eigen::SimplicialLDLT back solve failed\n");
            return false;
        }
        else {
            sol.resize(solE.size());
            std::memcpy(sol.data(), solE.data(), sizeof(T) * solE.size());
            return true;
        }
    }
#endif
}

template <class T>
bool Solve_CG(
    CSR_MATRIX<T> &sysMtr,
    const std::vector<T> &rhs,
    std::vector<T> &sol,
    T tol = (T)1.0e-5,
    int maxIter = 1000)
{
    if (sysMtr.Get_Matrix().rows() != rhs.size()) {
        printf("sysMtr dimension does not match with rhs!\n");
        return false;
    }

    if (maxIter <= 0) maxIter = 1000;

    const Eigen::SparseMatrix<T> &A = sysMtr.Get_Matrix();
    const int n = static_cast<int>(rhs.size());

    if ((int)sol.size() != n) sol.assign(n, T(0));

    // use std::vector storage and Eigen::Map for sparse MV
    std::vector<T> r(n), p(n), Ap(n);

    // compute r = b - A*x
    {
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> xMap(sol.data(), n);
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> bMap(rhs.data(), n);
        std::vector<T> Ax(n);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> AxMap(Ax.data(), n);
        AxMap.noalias() = A * xMap;
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> rMap(r.data(), n);
        rMap = bMap - AxMap;
    }

    // p = r
    p = r;

    // rsold = r.dot(r)
    T rsold = T(0);
    for (int i = 0; i < n; ++i) rsold += r[i] * r[i];
    if (rsold == T(0)) return true; // already converged

    const T eps = std::numeric_limits<T>::epsilon();
    bool truncated = false;  // 标记是否发生截断

    for (int iter = 0; iter < maxIter; ++iter) {
        // Ap = A * p
        {
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> pMap(p.data(), n);
            Eigen::Matrix<T, Eigen::Dynamic, 1> ApE = A * pMap;
            for (int i = 0; i < n; ++i) Ap[i] = ApE[i];
        }

        // denom = p.dot(Ap)
        T denom = T(0);
        for (int i = 0; i < n; ++i) denom += p[i] * Ap[i];
        if (std::abs(denom) < eps) {
            // breakdown: 处理崩溃情况，直接返回
            return false;
        }

        // 负曲率判断和方向更新
        double curv = T(0);
        // 将 std::vector 转换为 Eigen::VectorXd 计算内积
        Eigen::Map<const Eigen::VectorXd> pMap(p.data(), n);
        Eigen::Map<const Eigen::VectorXd> ApMap(Ap.data(), n);
        curv = pMap.dot(ApMap);  // p^T * A * p，计算曲率

        if (curv <= 0 || truncated) {
            Eigen::VectorXd v = pMap;  // 或者使用当前 Lanczos 向量
            double sg = (v.dot(Eigen::Map<const Eigen::VectorXd>(rhs.data(), n)) >= 0 ? 1.0 : -1.0);  // 根据方向的符号调整

            // 计算方向的尺度：|v^T * A * v| / ||v||^2
            double sc = std::abs(v.dot(A * v)) / std::max(v.squaredNorm(), eps);
            Eigen::VectorXd newDirection = -sg * sc * (v / std::max(v.norm(), eps));

            // 使用新的方向
            for (int i = 0; i < n; ++i) sol[i] = newDirection[i];
            truncated = true;
            break;
        }

        // 正常 CG 更新
        T alpha = rsold / denom;

        // x = x + alpha * p
        for (int i = 0; i < n; ++i) sol[i] += alpha * p[i];

        // r = r - alpha * Ap
        for (int i = 0; i < n; ++i) r[i] -= alpha * Ap[i];

        // rsnew = r.dot(r)
        T rsnew = T(0);
        for (int i = 0; i < n; ++i) rsnew += r[i] * r[i];

        if (std::sqrt((double)rsnew) < (double)tol) {
            return true; // converged
        }

        T beta = rsnew / rsold;
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];

        rsold = rsnew;
    }

    // did not converge within maxIter
    return false;
}




#ifdef CHOLMOD_DIRECT_SOLVER
template <class T>
class Solver_Direct_Helper {
public:
    cholmod_common cm;
    cholmod_sparse* A;
    cholmod_dense *b;
    void *Ax, *Ap, *Ai, *bx;
    cholmod_factor* L;

    Solver_Direct_Helper(CSR_MATRIX<T> &sysMtr) {
        cholmod_common cm;
        cholmod_start(&cm);

        // setup matrix
        A = cholmod_allocate_sparse(sysMtr.Get_Matrix().rows(),
                                                    sysMtr.Get_Matrix().cols(), sysMtr.Get_Matrix().nonZeros(),
                                                    true, true, -1, CHOLMOD_REAL, &cm);
        Ax = A->x;
        Ap = A->p;
        Ai = A->i;
        A->i = sysMtr.Get_Matrix().innerIndexPtr();
        A->p = sysMtr.Get_Matrix().outerIndexPtr();
        A->x = sysMtr.Get_Matrix().valuePtr();

        // factorization
        L = cholmod_analyze(A, &cm);
        cholmod_factorize(A, L, &cm);
        if (cm.status == CHOLMOD_NOT_POSDEF) {
            exit(0);
        }

        // prepare rhs data structure
        b = cholmod_allocate_dense(sysMtr.Get_Matrix().rows(), 1, sysMtr.Get_Matrix().rows(), CHOLMOD_REAL, &cm);
        bx = b->x;
    }

    ~Solver_Direct_Helper() {
        // free memory
        A->i = Ai;
        A->p = Ap;
        A->x = Ax;
        cholmod_free_sparse(&A, &cm);
        cholmod_free_factor(&L, &cm);
        b->x = bx;
        cholmod_free_dense(&b, &cm);
        cholmod_finish(&cm);
    }

    void Solve(const std::vector<T> &rhs, std::vector<T> &sol) {
        // back solve
        b->x = const_cast<T*>(rhs.data());
        cholmod_dense* x = cholmod_solve(CHOLMOD_A, L, b, &cm);
        sol.resize(rhs.size());
        std::memcpy(sol.data(), x->x, sol.size() * sizeof(T));
        cholmod_free_dense(&x, &cm);
    }
};
#else
template <class T>
class Solver_Direct_Helper {
public:
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solver;
    Solver_Direct_Helper(CSR_MATRIX<T> &sysMtr) {
        solver.compute(sysMtr.Get_Matrix());
        if (solver.info() != Eigen::Success) {
            exit(0);
        }
    }
    void Solve(const std::vector<T> &rhs, std::vector<T> &sol) {
        Eigen::VectorXd rhsE(rhs.size());
        std::memcpy(rhsE.data(), rhs.data(), sizeof(T) * rhs.size());
        Eigen::VectorXd solE = solver.solve(rhsE);
        if (solver.info() != Eigen::Success) {
            exit(0);
        }
        else {
            sol.resize(solE.size());
            std::memcpy(sol.data(), solE.data(), sizeof(T) * solE.size());
        }
    }
};
#endif

}