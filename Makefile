all:
	# Optimized build
	mpixlc -O3 -Wall -Werror -O3 -c -o simulation.o simulation.c
	nvcc -Werror all-warnings -O3 -c -o cuda-kernels.o cuda-kernels.cu
	mpixlc -Wall -Werror -O3 -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ \
		-o simulation simulation.o cuda-kernels.o
	# Debug build
	mpixlc -Wall -Werror -g -DDEBUG -c -o simulation-debug.o simulation.c
	nvcc -Werror all-warnings -G -g -DDEBUG -c -o cuda-kernels-debug.o cuda-kernels.cu
	mpixlc -Wall -Werror -g -DDEBUG -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ \
		-o simulation-debug simulation-debug.o cuda-kernels-debug.o
