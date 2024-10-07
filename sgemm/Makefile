all: workdir
	nvcc -o bin/gemm_v0 gemm_v0.cu -lcublas
	nvcc -o bin/gemm_v1 gemm_v1.cu -lcublas
	nvcc -o bin/gemm_v2 gemm_v2.cu -lcublas
	nvcc -o bin/gemm_v3 gemm_v3.cu -lcublas
	nvcc -o bin/gemm_v4 gemm_v4.cu -lcublas
	nvcc -o bin/gemm_v5 gemm_v5.cu -lcublas

workdir:
	@mkdir bin