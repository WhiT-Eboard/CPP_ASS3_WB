#include <iostream>
#include "immintrin.h"
#include <chrono>
#include "omp.h"
#include "cblas.h"
#include "stdlib.h"
#include <random>
using namespace std;
float mul(float *a,float *b,size_t size) {
    size_t eight_times_size = div(size,8).quot;
    eight_times_size *=8;
    float total=0;
#pragma omp parallel for num_threads(32) reduction(+:total)
    for (int i = 0; i < eight_times_size; i+=8) {
        total+= _mm256_dp_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),0xff)[0];
        total+= _mm256_dp_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),0xff)[4];
    }
    for (int i = eight_times_size; i < size; ++i) {
        total += a[i]*b[i];
    }
    return total;
}

int main(){
    size_t arrSize = 200000000;
    float temp =0;
    auto *a = new float[arrSize];
    auto *b = new float[arrSize];
    for (int i = 0; i < arrSize; ++i) {
        a[i] = i%1000000+i*0.0001;
        b[i] = i%1000000+i*0.0001;
    }
    auto begin = chrono::steady_clock::now();
    cout << fixed << mul(a,b,arrSize) << endl;
    auto end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << endl;

    //begin = chrono::steady_clock::now();
//#pragma omp parallel for num_threads(32) reduction(+:temp)
    //for (int i = 0; i < arrSize; ++i) {
    //    temp+=a[i]*b[i];
    //}
    //cout << temp << endl;
    //end = chrono::steady_clock::now();
    //cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;


    begin = chrono::steady_clock::now();
    cout << cblas_sdot(arrSize, a, 1, b, 1) << endl;
    end = chrono::steady_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;

}