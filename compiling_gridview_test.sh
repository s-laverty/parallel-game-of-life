xlc gridview_test.c -c -o gridview_test.o
nvcc cuda-kernels.cu -c -o cuda-kernels-test.o
xlc -O3 gridview_test.o cuda-kernels-test.o -o test.exe -L/usr/local/cuda-11.2/lib64 -lcudadevrt -lcudart -lstdc++