#include <iostream>
#include "immintrin.h"
#include <chrono>
#include <future>
#include "cblas.h"
#include "omp.h"

using namespace std;

float mul(float *a, float *b, size_t size) {
    __m256 ymmA, ymmB, sum;// ymmA_2, ymmB_2, sum_2;
    float total, result=0;//total_2, result = 0,result_2=0;
    std::size_t i, k, j;// len_1 = 240000 * (size / 240000)/2;
    /*
    //#pragma omp parallel for num_threads(32) reduction(+:result)
    future<void> thd1 = async(std::launch::async, [&] {
        for (std::size_t l = 0; l < len_1-240000; l += 240000) {
            total =0;
            for (int m = l; m < l+240000; m+=8) {
                ymmA = _mm256_loadu_ps(a + m);
                ymmB = _mm256_loadu_ps(b + m);
                sum = _mm256_dp_ps(ymmA, ymmB, 0xff);
                total += sum[0] + sum[4];
            }
            result+=total;
        }
    });

    future<void> thd2 = async(std::launch::async,[&]{
        for (std::size_t i = size-len_1; i+240000 < size; i+=240000) {
            total_2 = 0;
            for (int k = i; k < i+240000; ++k) {
                ymmA_2 = _mm256_loadu_ps(a + k);
                ymmB_2 = _mm256_loadu_ps(b + k);
                sum_2 = _mm256_dp_ps(ymmA_2, ymmB_2, 0xff);
                total_2 += sum[0] + sum[4];
            }
            result_2 += total_2;
        }
    });
    thd1.wait();
    thd2.wait();
    result+=result_2;
    total=0;
    for (j = i; j + 8 < size; j += 8) {
        ymmA = _mm256_loadu_ps(a + k);
        ymmB = _mm256_loadu_ps(b + k);
        sum = _mm256_dp_ps(ymmA, ymmB, 0xff);
        total += sum[0] + sum[4];
    }
    for (; j < size; ++j) {
        total += a[i] * b[i];
    }
    result += total;
     */
    for (i = 0; i < size - 240000; i += 240000) {
        total = 0;
        for (k = i; k + 8 <= i + 240000; k += 8) {
            ymmA = _mm256_loadu_ps(a + k);
            ymmB = _mm256_loadu_ps(b + k);
            sum = _mm256_dp_ps(ymmA, ymmB, 0xff);
            total += sum[0] + sum[4];
        }
        result += total;
    }
    total = 0;
    for (j = i; j + 8 < size; j += 8) {
        ymmA = _mm256_loadu_ps(a + k);
        ymmB = _mm256_loadu_ps(b + k);
        sum = _mm256_dp_ps(ymmA, ymmB, 0xff);
        total += sum[0] + sum[4];
    }
    for (; j < size; ++j) {
        total += a[i] * b[i];
    }
    result += total;
    return result;
}

int main() {
    std::size_t arrSize = 200000000;
    auto *a = new float[arrSize];
    auto *b = new float[arrSize];
    float temp = 0.0, temp2;
    for (int i = 0; i < arrSize; ++i) {
        a[i] = 1;
        b[i] = 1;
    }
    auto begin = chrono::steady_clock::now();
    cout << mul(a, b, arrSize) << endl;
    auto end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
    begin = chrono::steady_clock::now();
#pragma omp parallel for num_threads(32) reduction(+:temp)
    for (int i = 0; i < arrSize; ++i) {
       temp+=a[i]*b[i];
    }
    cout << temp << endl;
    end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
    begin = chrono::steady_clock::now();
    cout << cblas_sdot(arrSize, a, 1, b, 1) << endl;
    end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
    return 0;
}
