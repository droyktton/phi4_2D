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

#N?=1024

DISORDERAMP?=1.0

ACSQUAREWAVEFLAT = -DANTIPERIODICX -DDWINTHEMIDDLEIC -DTAUSQUAREWAVE=50 \
-DAMPSQUAREWAVE=0.05 -DMONITOR=1 -DDISORDERAMP=$(DISORDERAMP) #-DPRINTCONFIGS=50 #-DN=1024

ACSQUAREWAVECIRCULAR = -DDWCIRCULARIC=256 -DTAUSQUAREWAVE=50 \
-DAMPSQUAREWAVE=0.05 -DMONITOR=1 -DDISORDERAMP=$(DISORDERAMP) -DPRINTCONFIGS=50 #-DN=1024

PARAMS = $(ACSQUAREWAVEFLAT)
#PARAMS = $(ACSQUAREWAVECIRCULAR)

LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/lib64 


phi4: main.cu Makefile
	$(CXX) $(FLAGS) $(PARAMS) main.cu -o phi4 $(LDFLAGS) $(INCLUDES) 


update_git:
	git add *.cu Makefile README.md; git commit -m "program update"; git push

clean:
	rm -f phi4
