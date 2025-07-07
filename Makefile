CXX = nvcc

INCLUDES = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/include 
FLAGS = --expt-extended-lambda -lcufft -std=c++17 -lstdc++fs -O2 \
-gencode=arch=compute_52,code=sm_52 \
-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_86,code=sm_86 
#-gencode arch=compute_89,code=sm_89 

ACSQUAREWAVEFLAT = -DANTIPERIODICX -DDWINTHEMIDDLEIC -DTAUSQUARE=50 -DAMPSQUARE=0.05

PARAMS = $(ACSQUAREWAVEFLAT)

LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/lib64 


phi4: main.cu
	$(CXX) $(FLAGS) $(PARAMS) main.cu -o phi4 $(LDFLAGS) $(INCLUDES) 


update_git:
	git add *.cu *.py Makefile README.md *.gnu *.sh; git commit -m "program update"; git push

clean:
	rm phi4
