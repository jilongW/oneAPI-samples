//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Contents:
//     A simple matrix multiplication benchmark, using the oneAPI Math Kernel
//     Library (oneMKL).
//

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <typeinfo>

#include "utilities.hpp"

using namespace sycl;

bool test_gemv(queue &Q, int M, int N, int K, int Z, int R, int D)
{
    std::cout << "\nBenchmarking (" << M << " x " << K << ") x (" << K << " x " << N << ") matrix multiplication, " << "fp32_vec" << "\n";;

    std::cout << " -> Initializing data...\n";

    /* Allocate A/B/C matrices */
    int lda = nice_ld<float>(M);
    int ldb = nice_ld<float>(K);
    int ldc = nice_ld<float>(M);

    auto A = malloc_device<float>(lda * K, Q);
    auto B = malloc_device<float>(ldb * N, Q);
    auto C = malloc_device<float>(ldc * N, Q);

    constexpr int rd_size = 1048576;
    std::vector<float> host_vector(rd_size);
    auto host_data = host_vector.data();
    std::vector<float> correct_host_vector(rd_size);
    auto correct_host_data = correct_host_vector.data();
    /* Measure time for a given number of GEMM calls */
    bool verify = false;
    auto time_gemvs = [=, &Q, &host_data](int runs, bool verify=false) -> std::tuple<double, int> {
        using namespace oneapi::mkl;
        using namespace std::chrono;
        auto start = steady_clock::now();
        int ok = 0;
        if (verify == false){
            for (int i = 0; i < runs; i++)
                blas::column_major::gemv(Q, transpose::nontrans, M, K, 1, A, lda, B, N, 0, C, N);
            Q.wait_and_throw();
            auto end = steady_clock::now();
            return std::make_tuple(duration<float>(end - start).count(), ok);
        }
        else{
            size_t elems = std::min(ldc * N, rd_size);
            
            blas::column_major::gemv(Q, transpose::nontrans, M, K, 1, A, lda, B, N, 0, C, N);
            Q.wait_and_throw();
            Q.copy<float>(C, correct_host_data, elems).wait();
            auto end = steady_clock::now();
            auto used_time = duration<float>(end - start).count();
            int calls = int(600. / used_time);
            // correct_host_data[0] += 1.0;
            for (int i = 1; i < runs; i++){
                start = steady_clock::now();
                blas::column_major::gemv(Q, transpose::nontrans, M, K, 1, A, lda, B, N, 0, C, N);
                Q.wait_and_throw();
                end = steady_clock::now();
                used_time += duration<float>(end - start).count();
                Q.copy<float>(C, host_data, elems).wait();
                int linear_id = 0;
                for (size_t k = 0; k < M; k++) {
                    linear_id = k;
                    if (linear_id >= M) break;
                    if (std::abs(host_data[linear_id] - correct_host_data[linear_id]) > 1e-3) {
                        ok = i;
                        return std::make_tuple(duration<float>(end - start).count(), ok);
                    }
                }
                if ( i % calls == 0 ){
                    std::cout << " gemm has been running for " << (i/calls) *10 <<" minutes, and running " << i << " times\n";
                }
            }
            return std::make_tuple(used_time, ok);
        }
    };

    /* Fill A/B with all ones to verify correctness */
    generate_ones(rd_size, host_data);
    replicate_data(Q, A, lda * K, host_data, rd_size);
    replicate_data(Q, B, ldb * N, host_data, rd_size);

    /* Verify that the leading entries of C are correct */
    std::cout << " -> Verification...";
    (void) time_gemvs(1);
    size_t elems = std::min(ldc * N, rd_size);
    Q.copy(C, host_data, elems).wait();
    bool ok = true;
    int linear_id = 0;
    for (size_t j = 0; j < N; j++) {
        for (size_t i = 0; i < M; i++) {
            linear_id = j*ldc + i;
            if (linear_id >= elems) break;
            if (host_data[linear_id] != float(K)) {
                ok = false;
            }
        }
        if (linear_id >= elems) break;
    }

    std::cout << "gemv " << (ok ? " passes." : " FAILS!") << " for type: " << "fp32_vec" << "\n";
    if (!ok) { return false; }

    /* Fill A/B with random data */
    generate_random_data<float>(rd_size, host_data);
    replicate_data<float>(Q, A, lda * K, host_data, rd_size);
    replicate_data<float>(Q, B, ldb * N, host_data, rd_size);

    /* Do a warmup call with random data to initialize MKL and ensure kernels are JIT'ed if needed */
    std::cout << " -> Warmup...\n";
    (void) time_gemvs(10);

    /* Time one GEMM call, and estimate how many calls will be required to keep the
     * GPU busy for 1s. */
    auto [tare, _] = time_gemvs(1, true);
    int ncalls = std::max(4, std::min(1000, int(1. / tare)));
    if ( D != 1 ){
        ncalls = int(1. / tare);
        ncalls *= D;
    }
    else if( R != 1 ){
        ncalls = R;
    }
    /* Time that many GEMMs, subtracting the first call time to remove host overhead.
     * This gives a better idea of device performance. */
    std::cout << " -> Timing...\n";
    auto [time, result] = time_gemvs(ncalls + 1, true);
    time -= tare;
   
    auto avg = time / ncalls;

    /* Calculate and display performance */
    auto op_count = double(M) * double(N) * double(K) * 2;
    if (Z != -1){
        op_count += (double(M) * double(N) * double(Z) * 2);
    }
    auto flops = op_count / avg;

    flops *= 1e-9;
    char unit = 'G';
    if (flops >= 1000.) {
        flops *= 1e-3;
        unit = 'T';
    }
    if (flops >= 1000.) {
        flops *= 1e-3;
        unit = 'P';
    }
     if (result != 0){
        std::cout << "gemv FAILS" << " for type: " << "fp32_vec" << " on " << result <<" times run!"<< "\n";
    }
    else{
        std::cout << "gemv Passes" << " for type: " << "fp32_vec" << "!\n";
        std::cout << "\nAverage performance: " << flops << unit << 'F' << "\n";
    }
    

    /* Free data */
    free(C, Q);
    free(B, Q);
    free(A, Q);

    return true;
}


template <typename T>
static
bool test(queue &Q, int M, int N, int K, int Z, int R, int D)
{
    if ( Z == -1)
        std::cout << "\nBenchmarking (" << M << " x " << K << ") x (" << K << " x " << N << ") matrix multiplication, " << type_string<T>() << "\n";
    else
        std::cout << "\nBenchmarking (" << M << " x " << K << ") x (" << K << " x " << N << ") x (" << N << " x " << Z << ") matrix multiplication, " << type_string<T>() << "\n";;
    std::cout << " -> Initializing data...\n";

    /* Allocate A/B/C matrices */
    int lda = nice_ld<T>(M);
    int ldb = nice_ld<T>(K);
    int ldc = nice_ld<T>(M);
    int lde = nice_ld<T>(N);
    int ldf = nice_ld<T>(M);
    

    auto A = malloc_device<T>(lda * K, Q);
    auto B = malloc_device<T>(ldb * N, Q);
    auto C = malloc_device<T>(ldc * N, Q);
    auto E = malloc_device<T>(lde * Z, Q);   
    auto F = malloc_device<T>(ldf * Z, Q);
        
    int rd_size = lda * K + ldb * N + lda * N + lde * Z + lda * Z + 1;
    std::vector<T> host_vector(rd_size);
    auto host_data = host_vector.data();
    std::vector<T> correct_host_vector(rd_size);
    auto correct_host_data = correct_host_vector.data();
    /* Measure time for a given number of GEMM calls */
    bool verify = false;
    
    auto time_gemms = [=, &Q, &host_data](int runs, bool verify=false) -> std::tuple<double, int> {
        using namespace oneapi::mkl;
        using namespace std::chrono;
        auto start = steady_clock::now();
        int ok = 0;
        if (verify == false){
            if ( Z == -1){
                for (int i = 0; i < runs; i++)
                    blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
            }
            else{
                for (int i = 0; i < runs; i++){
                    blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
                    Q.wait_and_throw();
                    blas::gemm(Q, transpose::N, transpose::N, M, Z, N, 1, C, ldc, E, lde, 0, F, ldf);
                }
                    
            }
            Q.wait_and_throw();
            auto end = steady_clock::now();
            return std::make_tuple(duration<float>(end - start).count(), ok);
        }
        else{
            size_t elems;
            if ( Z == -1){
                elems = std::min(ldc * N, rd_size);
                blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
                Q.wait_and_throw();
                Q.copy(C, correct_host_data, elems).wait();
            }
            else{
                elems = std::min(ldf * Z, rd_size);
                blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
                Q.wait_and_throw();
                blas::gemm(Q, transpose::N, transpose::N, M, Z, N, 1, C, ldc, E, lde, 0, F, ldf);
                Q.wait_and_throw();
                Q.copy(F, correct_host_data, elems).wait();
            }
            auto end = steady_clock::now();
            auto used_time = duration<float>(end - start).count();

            // correct_host_data[0] += 1.0;
            if ( Z == -1){
                for (int i = 1; i < runs; i++){
                    start = steady_clock::now();
                    blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
                    Q.wait_and_throw();
                    end = steady_clock::now();
                    used_time += duration<float>(end - start).count();
                    Q.copy(C, host_data, elems).wait();
                    int linear_id = 0;
                    for (size_t j = 0; j < N; j++) {
                        for (size_t k = 0; k < M; k++) {
                            linear_id = j*ldc + k;
                            if (linear_id >= elems) break;
                            if (host_data[linear_id] != correct_host_data[linear_id]) {
                                ok = i;
                                return std::make_tuple(duration<float>(end - start).count(), ok);
                            }
                        }
                        if (linear_id >= elems) break;
                    }
                    
                }
            }
            else{
                for (int i = 1; i < runs; i++){
                    start = steady_clock::now();
                    blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
                    Q.wait_and_throw();
                    blas::gemm(Q, transpose::N, transpose::N, M, Z, N, 1, C, ldc, E, lde, 0, F, ldf);
                    Q.wait_and_throw();
                    end = steady_clock::now();
                    used_time += duration<float>(end - start).count();
                    Q.copy(F, host_data, elems).wait();
                    int linear_id = 0;
                    for (size_t j = 0; j < Z; j++) {
                        for (size_t k = 0; k < M; k++) {
                            linear_id = j*ldf + k;
                            if (linear_id >= elems) break;
                            if (host_data[linear_id] != correct_host_data[linear_id]) {
                                ok = i;
                                return std::make_tuple(duration<float>(end - start).count(), ok);
                            }
                        }
                        if (linear_id >= elems) break;
                    }
                    
                }
            }
            return std::make_tuple(used_time, ok);
        }
    };

    /* Fill A/B with all ones to verify correctness */
    generate_ones(rd_size, host_data);
    replicate_data(Q, A, lda * K, host_data, rd_size);
    replicate_data(Q, B, ldb * N, host_data, rd_size);
    if ( Z != -1){
        replicate_data(Q, E, lde * Z, host_data, rd_size);
    }
    /* Verify that the leading entries of C are correct */
    std::cout << " -> Verification...";
    (void) time_gemms(1);
    bool ok;
    if ( Z == -1){
        size_t elems = std::min(ldc * N, rd_size);
        Q.copy(C, host_data, elems).wait();
        ok = true;
        int linear_id = 0;
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < M; i++) {
                linear_id = j*ldc + i;
                if (linear_id >= elems) break;
                if (host_data[linear_id] != T(K)) {
                    ok = false;
                }
            }
            if (linear_id >= elems) break;
        }
    }
    else{
        size_t elems = std::min(ldf * Z, rd_size);
        Q.copy(F, host_data, elems).wait();
        ok = true;
        int linear_id = 0;
        for (size_t j = 0; j < Z; j++) {
            for (size_t i = 0; i < M; i++) {
                linear_id = j*ldf + i;
                if (linear_id >= elems) break;
                if (host_data[linear_id] != T(K) * N) {
                    ok = false;
                }
            }
            if (linear_id >= elems) break;
        }
    }
    
    std::cout << "gemm " << (ok ? " passes." : " FAILS!") << " for type: " << type_string<T>() << "\n";
    if (!ok) { return false; }

    /* Fill A/B with random data */
    generate_random_data(rd_size, host_data);
    replicate_data(Q, A, lda * K, host_data, rd_size);
    replicate_data(Q, B, ldb * N, host_data, rd_size);
    if ( Z != -1){
        replicate_data(Q, E, lde * Z, host_data, rd_size);
    }

    /* Do a warmup call with random data to initialize MKL and ensure kernels are JIT'ed if needed */
    std::cout << " -> Warmup...\n";
    (void) time_gemms(10);

    /* Time one GEMM call, and estimate how many calls will be required to keep the
     * GPU busy for 1s. */
    auto [tare, _] = time_gemms(1, true);
    int ncalls = std::max(4, std::min(1000, int(1. / tare)));
    if ( D != 1 ){
        ncalls = int(1. / tare);
        ncalls *= D;
    }
    else if( R != 1 ){
        ncalls = R;
    }
    /* Time that many GEMMs, subtracting the first call time to remove host overhead.
     * This gives a better idea of device performance. */
    std::cout << " -> Timing...\n";
    auto [time, result] = time_gemms(ncalls + 1, true);
    time -= tare;
   
    auto avg = time / ncalls;

    /* Calculate and display performance */
    auto op_count = double(M) * double(N) * double(K) * 2;
    if (Z != -1){
        op_count += (double(M) * double(N) * double(Z) * 2);
    }
    auto flops = op_count / avg;
    flops *= 1e-9;
    char unit = 'G';
    if (flops >= 1000.) {
        flops *= 1e-3;
        unit = 'T';
    }
    if (flops >= 1000.) {
        flops *= 1e-3;
        unit = 'P';
    }
     if (result != 0){
        std::cout << "gemm FAILS" << " for type: " << type_string<T>() << " on " << result <<" times run!"<< "\n";
    }
    else{
        std::cout << "gemm Passes" << " for type: " << type_string<T>() << "!\n";
        std::cout << "\nAverage performance: " << flops << unit << 'F' << "\n";
    }
    

    /* Free data */
    free(F, Q);
    free(E, Q);
    free(C, Q);
    free(B, Q);
    free(A, Q);

    return true;
}
template <>
bool test<std::int8_t>(queue &Q, int M, int N, int K, int Z, int R, int D)
{
    
    if ( Z == -1)
        std::cout << "\nBenchmarking (" << M << " x " << K << ") x (" << K << " x " << N << ") matrix multiplication, " << type_string<std::int8_t>() << "\n";
    else
        std::cout << "\nBenchmarking (" << M << " x " << K << ") x (" << K << " x " << N << ") x (" << N << " x " << Z << ") matrix multiplication, " << type_string<std::int8_t>() << "\n";;
    std::cout << " -> Initializing data...\n";

    /* Allocate A/B/C matrices */
    int lda = nice_ld<std::int8_t>(M);
    int ldb = nice_ld<std::int8_t>(K);
    int ldc = nice_ld<std::int32_t>(M);
    int lde = nice_ld<std::int32_t>(N);
    int ldf = nice_ld<std::int32_t>(M);

    auto A = malloc_device<std::int8_t>(lda * K, Q);
    auto B = malloc_device<std::int8_t>(ldb * N, Q);
    auto C = malloc_device<std::int32_t>(ldc * N, Q);
    auto E = malloc_device<std::int32_t>(lde * Z, Q);   
    auto F = malloc_device<std::int32_t>(ldf * Z, Q);

    int rd_size = lda * K + ldb * N + lda * N + lde * Z + lda * Z + 1;
    std::vector<std::int32_t> host_vector(rd_size);
    auto host_data = host_vector.data();
    std::vector<std::int32_t> correct_host_vector(rd_size);
    auto correct_host_data = correct_host_vector.data();
    /* Measure time for a given number of GEMM calls */
    bool verify = false;
    auto time_gemms = [=, &Q, &host_data](int runs, bool verify=false) -> std::tuple<double, int> {
        using namespace oneapi::mkl;
        using namespace std::chrono;
        auto start = steady_clock::now();
        int ok = 0;
        if (verify == false){
            if ( Z == -1){
                for (int i = 0; i < runs; i++)
                    blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
            }
            else{
                for (int i = 0; i < runs; i++){
                    blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, (float *)C, ldc);
                    Q.wait_and_throw();
                    blas::gemm(Q, transpose::N, transpose::N, M, Z, N, 1, (float *)C, ldc, (float *)E, lde, 0, (float *)F, ldf);
                }
                    
            }
            Q.wait_and_throw();
            auto end = steady_clock::now();
            return std::make_tuple(duration<float>(end - start).count(), ok);
        }
        else{
            size_t elems;
            if ( Z == -1){
                elems = std::min(ldc * N, rd_size);
                blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
                Q.wait_and_throw();
                Q.copy(C, correct_host_data, elems).wait();
            }
            else{
                elems = std::min(ldf * Z, rd_size);
                blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
                Q.wait_and_throw();
                blas::gemm(Q, transpose::N, transpose::N, M, Z, N, 1, (float *)C, ldc, (float *)E, lde, 0, (float *)F, ldf);
                Q.wait_and_throw();
                Q.copy(F, correct_host_data, elems).wait();
            }
            auto end = steady_clock::now();
            auto used_time = duration<float>(end - start).count();

            // correct_host_data[0] += 1.0;
            if ( Z == -1){
                for (int i = 1; i < runs; i++){
                    start = steady_clock::now();
                    blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
                    Q.wait_and_throw();
                    end = steady_clock::now();
                    used_time += duration<float>(end - start).count();
                    Q.copy(C, host_data, elems).wait();
                    int linear_id = 0;
                    for (size_t j = 0; j < N; j++) {
                        for (size_t k = 0; k < M; k++) {
                            linear_id = j*ldc + k;
                            if (linear_id >= elems) break;
                            if (host_data[linear_id] != correct_host_data[linear_id]) {
                                ok = i;
                                return std::make_tuple(duration<float>(end - start).count(), ok);
                            }
                        }
                        if (linear_id >= elems) break;
                    }
                    
                }
            }
            else{
                for (int i = 1; i < runs; i++){
                    start = steady_clock::now();
                    blas::gemm(Q, transpose::N, transpose::N, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
                    Q.wait_and_throw();
                    blas::gemm(Q, transpose::N, transpose::N, M, Z, N, 1, (float *)C, ldc, (float *)E, lde, 0, (float *)F, ldf);
                    Q.wait_and_throw();
                    end = steady_clock::now();
                    used_time += duration<float>(end - start).count();
                    Q.copy(F, host_data, elems).wait();
                    int linear_id = 0;
                    for (size_t j = 0; j < Z; j++) {
                        for (size_t k = 0; k < M; k++) {
                            linear_id = j*ldf + k;
                            if (linear_id >= elems) break;
                            if (host_data[linear_id] != correct_host_data[linear_id]) {
                                ok = i;
                                return std::make_tuple(duration<float>(end - start).count(), ok);
                            }
                        }
                        if (linear_id >= elems) break;
                    }
                    
                }
            }
            return std::make_tuple(used_time, ok);
        }
    };
    std::vector<std::int8_t> host_zero(rd_size);
    auto host_zero_data = host_zero.data();
    /* Fill A/B with all ones to verify correctness */
    generate_ones(rd_size, host_zero_data);
    if ( Z != -1)
        generate_ones(rd_size, host_data);
    replicate_data(Q, A, lda * K, host_zero_data, rd_size);
    replicate_data(Q, B, ldb * N, host_zero_data, rd_size);
    if ( Z != -1){
        replicate_data(Q, E, lde * Z, host_data, rd_size);
    }
    /* Verify that the leading entries of C are correct */
    std::cout << " -> Verification...";
    (void) time_gemms(1);
    size_t elems = std::min(ldc * N, rd_size);
    bool ok;
    if ( Z == -1){
        size_t elems = std::min(ldc * N, rd_size);
        Q.copy(C, host_data, elems).wait();
        ok = true;
        int linear_id = 0;
        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < M; i++) {
                linear_id = j*ldc + i;
                if (linear_id >= elems) break;
                if (host_data[linear_id] != int32_t(K)) {
                    ok = false;
                }
            }
            if (linear_id >= elems) break;
        }
    }
    else{
        size_t elems = std::min(ldf * Z, rd_size);
        Q.copy(F, host_data, elems).wait();
        ok = true;
        int linear_id = 0;
        for (size_t j = 0; j < Z; j++) {
            for (size_t i = 0; i < M; i++) {
                linear_id = j*ldf + i;
                if (linear_id >= elems) break;
                if (host_data[linear_id] != int32_t(K) * N) {
                    ok = false;
                }
            }
            if (linear_id >= elems) break;
        }
    }
    std::cout << "gemm " << (ok ? " passes." : " FAILS!") << " for type: " << type_string<std::int8_t>() << "\n";
    if (!ok) { return false; }

    /* Fill A/B with random data */
    generate_random_data(rd_size, host_zero_data);
    if ( Z != -1)
        generate_random_data(rd_size, host_data);
    replicate_data(Q, A, lda * K, host_zero_data, rd_size);
    replicate_data(Q, B, ldb * N, host_zero_data, rd_size);
    if ( Z != -1){
        replicate_data(Q, E, lde * Z, host_data, rd_size);
    }

    /* Do a warmup call with random data to initialize MKL and ensure kernels are JIT'ed if needed */
    std::cout << " -> Warmup...\n";
    (void) time_gemms(10);

    /* Time one GEMM call, and estimate how many calls will be required to keep the
     * GPU busy for 1s. */
    auto [tare, _] = time_gemms(1, true);
    int ncalls = std::max(4, std::min(1000, int(1. / tare)));
    if ( D != 1 ){
        ncalls = int(1. / tare);
        ncalls *= D;
    }
    else if( R != 1 ){
        ncalls = R;
    }
    /* Time that many GEMMs, subtracting the first call time to remove host overhead.
     * This gives a better idea of device performance. */
    std::cout << " -> Timing...\n";
    auto [time, result] = time_gemms(ncalls + 1, true);
    time -= tare;
   
    auto avg = time / ncalls;

    /* Calculate and display performance */
    auto op_count = double(M) * double(N) * double(K) * 2;
    if (Z != -1){
        op_count += (double(M) * double(N) * double(Z) * 2);
    }
    auto flops = op_count / avg;

    flops *= 1e-9;
    char unit = 'G';
    if (flops >= 1000.) {
        flops *= 1e-3;
        unit = 'T';
    }
    if (flops >= 1000.) {
        flops *= 1e-3;
        unit = 'P';
    }
     if (result != 0){
        std::cout << "gemm FAILS" << " for type: " << type_string<std::int8_t>() << " on " << result <<" times run!"<< "\n";
    }
    else{
        std::cout << "gemm Passes" << " for type: " << type_string<std::int8_t>() << "!\n";
        std::cout << "\nAverage performance: " << flops << unit << 'F' << "\n";
    }
    

    /* Free data */
    free(F, Q);
    free(E, Q);
    free(C, Q);
    free(B, Q);
    free(A, Q);

    return true;
}
static
void usage(const char *pname)
{
    std::cerr << "Simple Usage:\n"
              << "  " << pname << " [type] N                                    benchmark (NxN) x (NxN) square matrix multiplication (default: N = 4096)\n"
              << "  " << pname << " [type] M N K                                benchmark (MxK) x (KxN) square matrix multiplication\n"
              << "  " << pname << " [type] M N K                                benchmark (MxK) x (KxN) square matrix multiplication\n"
              << "  " << pname << " [type] -m 4096 -n 4096 -k 4096 -z 4096      benchmark (MxK) x (KxN) x (NxZ) square matrix multiplication\n"
              << "  " << pname << " [type] -m 4096 -n 4096 -k 4096 -c 0         benchmark (MxK) x (KxN) square matrix multiplication on device 0\n"
              << "  " << pname << " [type] -m 4096 -n 4096 -k 4096 -r 10        benchmark 10 times (MxK) x (KxN) square matrix multiplication \n"
              << "  " << pname << " [type] -m 4096 -n 4096 -k 4096 -d 10m       benchmark (MxK) x (KxN) square matrix multiplication for 10 mins\n"
              << "\n"
              << "The optional [type] selects the data type:\n"
              << "   double    [default]\n"
              << "   fp64\n"
              << "   single\n"
              << "   fp32 (or fp32_mat)\n"
              << "   half\n"
              << "   fp16\n"
              << "   bf16\n"
              << "   int8\n"
              << "   fp32_vec (gemv instead of gemm)\n"
              << "   all (runs all above)\n"
              << "\n"
              << "Option Usage:\n"
              << "  " << pname << " [type] [Options]\n"
              << "  -m                                  benchmark (MxK) x (KxN) square matrix multiplication (default: M = 4096)\n"
              << "  -n                                  benchmark (MxK) x (KxN) square matrix multiplication (default: N = 4096)\n"
              << "  -k                                  benchmark (MxK) x (KxN) square matrix multiplication (default: K = 4096)\n"
              << "  -z                                  benchmark (MxK) x (KxN) x (NxZ) square matrix multiplication (default: not use)\n"
              << "                                      fp32_vec can't use this args\n"
              << "  -c                                  running benchmark on which device (default running on all devices)\n"
              << "  -r                                  running benchmark multiple times, conflict with -d (default 1)\n"
              << "  -d                                  Duration of running benchmark, conflict with -r (default 1s)\n"
              << "                                      can be set to Xs or Xm or Xh, will overwrite -r\n"
              << "\n"
              << "This benchmark uses the default DPC++ device, which can be controlled using\n"
              << "  the ONEAPI_DEVICE_SELECTOR environment variable\n";
    std::exit(1);
}

static
bool device_has_fp64(sycl::device const& D) {
    return (D.get_info<sycl::info::device::double_fp_config>().size() != 0);
}

static
void device_info(sycl::device const& D) {
    std::cout << "oneMKL DPC++ GEMM benchmark\n"
              << "---------------------------\n"
              << "Platform:                " << D.get_platform().get_info<info::platform::name>()         << "\n"
              << "Device:                  " << D.get_info<info::device::name>()                          << "\n"
              << "Driver_version:          " << D.get_info<info::device::driver_version>()                << "\n"
              << "Core/EU count:           " << D.get_info<info::device::max_compute_units>()             << "\n"
              << "Maximum clock frequency: " << D.get_info<info::device::max_clock_frequency>() << " MHz" << "\n"
              << "FP64 capability:         " << (device_has_fp64(D) ? "yes" : "no") << "\n"
              << "\n"
              ;
}

int main(int argc, char **argv)
{
    auto pname = argv[0];
    int M = 4096, N = 4096, K = 4096, Z = -1;
    int R = 1, C = -1, D = 1;
    std::string type = "none";

    if (argc <= 1)
        usage(pname);

    if (argc > 1 && std::isalpha(argv[1][0])) {
        type = argv[1];
        if (type == "int8" || type == "bf16"){
            N *= 2;
            M *= 2;
            K *= 2;
        }
        argc--; argv++;
    }

    if (argc == 2) M = N = K = std::atoi(argv[1]);

    if (argc ==4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    for (int i = 1; i< argc;i++){
        if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
            usage(pname);
            exit(0);
        }
        else if ((strcmp(argv[i], "-m") == 0)){
            if (isdigit(argv[i + 1][0])) {
                M = std::atoi(argv[i + 1]);
            } else {
                usage(pname);
                exit(-1);
            }
            i++;
        }
        else if ((strcmp(argv[i], "-n") == 0)){
            if (isdigit(argv[i + 1][0])) {
                N = std::atoi(argv[i + 1]);
            } else {
                usage(pname);
                exit(-1);
            }
            i++;
        }
        else if ((strcmp(argv[i], "-k") == 0)){
            if (isdigit(argv[i + 1][0])) {
                K = std::atoi(argv[i + 1]);
            } else {
                usage(pname);
                exit(-1);
            }
            i++;
        }
        else if ((strcmp(argv[i], "-z") == 0)){
            if (isdigit(argv[i + 1][0])) {
                Z = std::atoi(argv[i + 1]);
                if ( Z <= 0){
                    usage(pname);
                    exit(-1);
                }
            } else {
                usage(pname);
                exit(-1);
            }
            i++;
        }
        else if ((strcmp(argv[i], "-r") == 0)){
            if (isdigit(argv[i + 1][0])) {
                R = std::atoi(argv[i + 1]);
            } else {
                usage(pname);
                exit(-1);
            }
            i++;
        }
        else if ((strcmp(argv[i], "-c") == 0)){
            if (isdigit(argv[i + 1][0])) {
                C = std::atoi(argv[i + 1]);
            } else {
                usage(pname);
                exit(-1);
            }
            i++;
        }
        else if ((strcmp(argv[i], "-d") == 0)){
            std::string str = argv[i+1];
            if (str[str.size() - 1] == 'h'){
                str.pop_back();
                D = std::atoi(str.c_str()) * 60 * 60;
            }
            else if(str[str.size() - 1] == 'm'){
                str.pop_back();
                D = std::atoi(str.c_str()) * 60;
            }
            else if(str[str.size() - 1] == 's'){
                str.pop_back();
                D = std::atoi(str.c_str());
            }
            else{
                usage(pname);
                exit(-1);
            }
            if ( D < 1 ){
                usage(pname);
                exit(-1);
            }
            i++;
        }
    }
    if (M <= 0 || N <= 0 || K <= 0 || R < 1 || D < 1 || (type == "fp32_vec" && Z >= 1))
        usage(pname);
    
    bool g_success = true;
    try { 
        device d(default_selector_v);
       
        auto P = d.get_platform();
        auto RootDevices = P.get_devices();
        auto c = context(RootDevices);
        int device_id = 0;
        
        if (C >= int(RootDevices.size()) ){
            std::cout << "Can't find device " << C <<" , please check your system" << "\n"; 
            exit(-1);
        }
        for (auto &d : RootDevices) {
            device_info(d);
            if ( C != -1 ){
                if( device_id != C){
                    device_id++;
                    continue;
                }
            }
            std::cout << "Running on device: " << device_id << "\n";
            queue Q(c, d);

            if ("none" == type)
                std::string type = device_has_fp64(d) ? "double" : "float";

            if (type == "double") {
                if (device_has_fp64(d))
                    test<double>(Q, M, N, K, Z, R, 1);
                else {
                    std::cout << "no FP64 capability on given SYCL device and type == \"double\"";
                    return 1;
                }
            }
            else if (type == "single" || type == "float")
                g_success = g_success && test<float>(Q, M, N, K, Z, R, D);
            else if (type == "half")
                g_success = g_success && test<half>(Q, M, N, K, Z, R, D);
            else if (type == "fp16")
                g_success = g_success && test<half>(Q, M, N, K, Z, R, D);
            else if (type == "fp32_mat" )
                g_success = g_success && test<float>(Q, M, N, K, Z, R, D);
            else if (type == "bf16")
                g_success = g_success && test<oneapi::mkl::bfloat16>(Q, M, N, K, Z, R, D);
            else if (type == "int8")
                g_success = g_success && test<std::int8_t>(Q, M, N, K, Z, R, D);
            else if (type == "fp32_vec"){
                g_success = g_success && test_gemv(Q, M, 1, K, Z, R, D);
                N = 1;
            }
                
            else if (type == "all") {
                type = "half";
                g_success = g_success && test<half>(Q, M, N, K, Z, R, D);

                type = "float";
                g_success = g_success && test<float>(Q, M, N, K, Z, R, D);

                if (device_has_fp64(d)) {
                    type = "double";
                    g_success = g_success && test<double>(Q, M, N, K, Z, R, D);
                }
            } else {
                type = "none";
                usage(pname);
            }
            device_id++;
            std::cout << "\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } 
        
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << "\n";
        std::cerr << " while performing GEMM for" 
            << " M=" << M 
            << ", N=" << N
            << ", K=" << K
            << ", type `" << type << "`"
            << "\n";
        return 139;
    }
    return g_success ? 0 : 1;
}

