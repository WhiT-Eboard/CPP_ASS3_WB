#CPP Assignment 3

*by 11911307 Li kangxin*

[gitHubSite](https://github.com/WhiT-Eboard/CPP_ASS3_WB)
##Content
1. Basic idea
2. First try with SMID and AVX2
3. OMP and Parallel processing
4. Combination of both methods
5. The Ultimate weapon:CUDA and GPU

>\-\-\-\-Efficiency, Or Accuracy, You Can Only Choose One
>


##1. Basic idea
The most obvious way to implement the the dot product is shown:

```c++
for (int i=0;i<arrary_size;++i){
    product+=a[i]*b[i];
}
```
The running time of the method is floating around `900ms` to `1300ms` on an **AMD Ryzen 5 4500U** CPU when facing a data
scale of `200M float`.
To compare with it, the `cblas_sdot()` function of **OpenBLAS** is under `100ms` under the same condition.
>Actually it is almost IMPOSSIBLE to get the correct answer from the code above. Since `float` will suffer great 
>precision lose if the one side of the addend is far bigger than another (>10000000) due to its storage. So the real 
>code will be look like this:

```c++
for (int i=0;i<1000;++i){
    for (int j=0;j<2000;++j){
        temp += a[j]*b[j];
    }
    product += temp;
    temp = 0;
}
``` 
##2. First try with SMID and AVX2
The AVX2 is an instruction set of CPU which can calculate a handful number at a time. It is included in the head file 
`immintrin.h` Which can greatly improve the 
performance of our program. Unfortunately, most of the CPUs are not support to AVX-512, so I can only use AVX-2 to 
implement the function.
````c++
float mul(float* a,float* b, size_t size){
    __m256 ymmA, ymmB, sum;//data of float 
    float total;
    std::size_t i;
    for (int i=0; i + 8 <= size; i += 8) {// 256bit can store 8-float number at a time
        ymmA = _mm256_loadu_ps(a + i);//since our data is not aligned
        ymmB = _mm256_loadups(b + i);
        sum = _mm256_dp_ps(ymmA, ymmB, 0xff);//macro of dot product 
        total += sum[0] + sum[4];
    }
    return total;
}
````
There still some problem with this code, if the input array size is not a multiple of eight we will lose some numbers, or
we will get a segment fault otherwise. Another critical problem is same as the origin code of Part 1, we will lose 
precision due to float itself. And just as what we've done to the first code, we will get an improved version like this:
````c++
float mul(float *a, float *b, size_t size) {
    __m256 ymmA, ymmB, sum;
    float total, result=0;
    std::size_t i, k, j;
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
````
>First we separate the array into pieces of length `240000`, medium sized while a multiple of eight as well.
>For the rest we do what the origin code do,calculate them! And we may get a small tail if the array length is not a 
>multiple of eight. That's what last for-loop do.

So far, we have greatly improved the efficiency of our program. The running time of it has reduced to `100ms` to `170ms`
on an **AMD Ryzen 5 4500U** CPU when facing a data scale of `200M float`. CHEERS!!!
Only a small gap with the `cblas_sdot()`.

##3. OMP and Parallel processing
Think about it, if there exists ten people who can calculate for us, ten times improvement of the speed! If we see the 
AVX as one person with eight hands, parallel processing is ten people who have eight hands. The implement of parallel
calculating is done by the pack **OpenMP**, which includes in the head file `omp.h`. If we apply this magic to our basic
idea, it will be look like this:
````c++
#pragma omp parallel for num_threads(32) reduction(+:temp)
    for (int i = 0; i < arrSize; ++i) {
       temp+=a[i]*b[i];
    }
````
>`#pragma omp parallel for`: means the following for-loop will be processed by multi-threads.
>
>`num_threads(32)`: the threads we will use is 32
>
>`reduction(+:temp)` : `reduction` is a key word of **OpenMP** which will make sure the value of the variable be counted
>correctly.
>
>see more at [**OpenMP**](https://zhuanlan.zhihu.com/p/61857547)

Thanks to this package, the optimization of the for-loop is such a easy. The improvement is also obvious: the running
time can be reduced to `300ms`, more than 60% efficiency. So we have two powerful weapons now, what if we combined them
together?
>You may wondering why I didn't use improved version to avoid precision loss. Because the loop is done by multi-threads,
>the sum of each loop will be more average. Which means huge loss has already been avoided. Since we can't avoid them 
>all, this is better enough.

##4. Combination of both methods
We are now have a clear aim: to combine both method to one, it's not an though job. The code looks this:
````c++
float mul(float *a,float *b,size_t size) {
    size_t eight_times_size = div(size,8).quot;
    eight_times_size *=8;//random name, just forgot it :P
    float total=0;
    for (int i = eight_times_size; i < size; ++i) {
        total += a[i]*b[i];
    }
#pragma omp parallel for num_threads(32) reduction(+:total)
    for (int i = 0; i < eight_times_size; i+=8) {
         total+= _mm256_dp_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),0xff)[0];
         total+= _mm256_dp_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),0xff)[4];
    }
    return total;
}
````
>The array will be tailed into a multiple of eight and throw into the loop, then calculate the tail, such easy,doesn't it?

The final result of this is amazing, it can done the calculation under `100ms`,which means the same level with 
`cblas_sdot()`! Although when data is very large, precision will lose. It is still quick enough.

##5. The Ultimate weapon:CUDA and GPU
Since only the calculation is needed, GPU is more powerful than CPU. By using CUDA，the  speed will be unbelieveable.
The kernel of the program is faster than anything, usually about `100μs`. But memory and IO is quite slow, the whole part
will run more than `600ms`. For `200M float` data, it's to slow. But for an even bigger data, 
using GPU to calculate may become the only choice.
