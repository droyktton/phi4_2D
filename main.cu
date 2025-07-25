#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>  // <-- required for std::setprecision


const int N = 1024;              // Grid size (NxN)
int Nsteps = 50000;        // Time steps
const float dt = 0.10f;
const float gamma_ = 1.0f;
const float c = 1.0f;
const float epsilon0 = 1.0f;
float h = 0.0f;
float noise_amp = 0.0f;
float disorder_amp = 0.0f;


void write_field_to_file(const thrust::host_vector<float>& phi_host,
                         int Nx, int Ny, int timestep) {
    std::ofstream file;
    std::string filename = "phi_t" + std::to_string(timestep) + ".dat";
    file.open(filename);

    file << std::fixed << std::setprecision(6);  // Optional: formatting

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            file << phi_host[j * Nx + i] << " ";
        }
        file << "\n";
    }

    file.close();
}

void write_matrix_to_file(const thrust::host_vector<float>& x,
                         int Nx, int Ny, std::ofstream &file) {
    
    file << std::fixed << std::setprecision(6);  // Optional: formatting

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            file << x[j * Nx + i] << " ";
        }
        file << "\n";
    }

    file.close();
}

struct Laplacian2D {
    int N;
    const float* phi;
    __host__ __device__
    float operator()(int idx) const {
        int i = idx / N;
        int j = idx % N;
        int ip = (i + 1) % N, im = (i - 1 + N) % N;
        int jp = (j + 1) % N, jm = (j - 1 + N) % N;

        #ifndef ANTIPERIODICX
        return phi[im*N + j] + phi[ip*N + j] + phi[i*N + jm] + phi[i*N + jp] - 4.0f * phi[idx];
        #else
        return
        ((i==0)?(-1.0f):(1.0f))*phi[im*N + j] +
        ((i==N-1)?(-1.0f):(1.0f))*phi[ip*N + j] +
        phi[i*N + jm] + phi[i*N + jp] - 4.0f * phi[idx];
        #endif
    }
};

struct GradientSquared2D {
    int N;
    const float* phi;
    __host__ __device__
    float operator()(int idx) const {
        int i = idx / N;
        int j = idx % N;
        int ip = (i + 1) % N, im = (i - 1 + N) % N;
        int jp = (j + 1) % N, jm = (j - 1 + N) % N;
        
        return    (phi[im*N + j] - phi[ip*N + j])*(phi[im*N + j] - phi[ip*N + j]) 
                + (phi[i*N + jp] - phi[i*N + jm])*(phi[i*N + jp] - phi[i*N + jm]);
    }
};


// Nonlinear and noise update
struct PhiUpdate {
    float c, epsilon0, gamma_, dt, h, noise_amp;
    const float* laplace;
    const float* phi_old;
    const float* r_disorder;

    unsigned int seed;
    __host__ __device__
    float operator()(int i) const {
        float phi = phi_old[i];
        float lap = laplace[i];
        float r = r_disorder[i];

        // Gaussian noise generator
        thrust::default_random_engine rng(seed);
        thrust::normal_distribution<float> dist(0.0f, 1.0f);
        rng.discard(i);
        float xi = noise_amp * dist(rng);

        float nonlinear = epsilon0 * ((1.0f + r) * phi - phi * phi * phi);
        float dphi = (c * lap + nonlinear + h + xi) * dt / gamma_;
        return phi + dphi;
    }
};

float positives_in_region(const thrust::device_vector<float>& phi, int imin, int imax, int jmin, int jmax) {
    
    int size = N*N;
    // Count the number of positive values in phi
    return thrust::count_if(
        thrust::make_zip_iterator(thrust::make_tuple(phi.begin(), thrust::make_counting_iterator(0))), 
        thrust::make_zip_iterator(thrust::make_tuple(phi.end(), thrust::make_counting_iterator(size))), 
        [imin, imax, jmin, jmax] __device__ (const thrust::tuple<float, int>& t) {
            float x = thrust::get<0>(t);
            int idx = thrust::get<1>(t);
            int i = idx / N;
            int j = idx % N;
            return (x > 0.0f && i >= imin && i < imax && j >= jmin && j < jmax);
        }
    );
} 
 
float ElasticEnergy(const thrust::device_vector<float>& phi) {
    int size = N * N;
    float energy = 0.0f;

    GradientSquared2D grad2_op{N, thrust::raw_pointer_cast(phi.data())};
    energy = thrust::transform_reduce(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(size),
            grad2_op,
            0.0f,
            thrust::plus<float>()
    );
    energy *= 0.5f; // Factor of 1/2 for the elastic energy
    return energy;
} 
 

int main(int argc, char **argv) {

    std::ofstream logout("log.txt");

    #ifdef TAUSQUAREWAVE
    float tausquarewave = atof(argv[1]);
    float ampsquarewave = atof(argv[2]);
    logout << "square wave = (" << tausquarewave << "," << ampsquarewave << ")" << std::endl;
    #endif

    std::ofstream monitor_out("monitor.dat");

    int size = N * N;
    thrust::device_vector<float> phi(size);
    thrust::device_vector<float> phi_new(size);
    thrust::device_vector<float> r_disorder(size);
    thrust::device_vector<float> laplace(size);

    #ifdef RANDOMWITHFRAMEIC
    // Initialize phi randomly in [-1,1]
    logout << "random initial condition with frame" << std::endl;
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(size),
        phi.begin(),
        [] __host__ __device__ (int n) {
            thrust::default_random_engine rng(1234);
            thrust::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            rng.discard(n);

            int i = n / N;
            int j = n % N;
            bool isborder = (i < 10 || i > N-10 || j < 10 || j > N-10);

            if(isborder) return -1.0f;
            //else return (dist(rng)>0)?(1.0f):(-1.0f);
            else return dist(rng);
        }
    );
    #endif
    
    #ifdef DWINTHEMIDDLEIC
    // Initialize phi randomly in [-1,1]
    logout << "flat domain wall in the middle" << std::endl;
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(size),
        phi.begin(),
        [] __host__ __device__ (int n) {
            thrust::default_random_engine rng(1234);
            thrust::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            rng.discard(n);

            int i = n / N;
            //int j = n % N;
            bool isleft = (i < N/2);

            return ((isleft) ? (1.0f) : (-1.0f));        
        }
    );    
    #endif


    #ifdef DWCIRCULARIC
    // Initialize phi randomly in [-1,1]
    logout << "circular domain wall in the middle" << std::endl;
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(size),
        phi.begin(),
        [] __host__ __device__ (int n) {
            //thrust::default_random_engine rng(1234);
            //thrust::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            //rng.discard(n);

            int i = n / N;
            int j = n % N;
            bool isincircle = ((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2) < DWCIRCULARIC*DWCIRCULARIC);

            return ((isincircle) ? (1.0f) : (-1.0f));        
        }
    );    
    #endif


    logout << "Disorder Amplitude " << DISORDERAMP << std::endl; 	

    // Initialize disorder r(x,y)
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(size),
        r_disorder.begin(),
        [] __host__ __device__ (int i) {
            thrust::default_random_engine rng(5678);
            thrust::uniform_real_distribution<float> dist(-DISORDERAMP,DISORDERAMP);
            rng.discard(i);
            return dist(rng);
        }
    );

    #ifdef MONITOR
    logout << "Monitor " << MONITOR << std::endl; 	
    #endif

    /*#ifdef TAUSQUAREWAVE	
    std::cout << "TAUSQUAREWAVE " << TAUSQUAREWAVE << std::endl; 	
    std::cout << "AMPSQUAREWAVE " << AMPSQUAREWAVE << std::endl; 	
    #endif*/
	

    #ifdef PRINTCONFIGS
    thrust::host_vector<float> x(r_disorder.size());
    
    std::ofstream out1("disorder.dat");
    thrust::copy(r_disorder.begin(), r_disorder.end(), x.begin());  // No new allocation
    write_matrix_to_file(x, N, N, out1);
    
    std::ofstream out2("initial.dat");
    thrust::copy(phi.begin(), phi.end(), x.begin());  // No new allocation
    write_matrix_to_file(x, N, N, out2);
    #endif




    thrust::host_vector<float> phi_host(size);
    //int nprint = 100;
    float t=0.0f;
    #ifdef TAUSQUAREWAVE
    h = 1.0f*ampsquarewave; //AMPSQUAREWAVE;
    #endif


    float prevmag = positives_in_region(phi, 10, N-10, 0, N);
    logout << "Initial magnetization: " << prevmag << std::endl;

    float Vac=0.0f;

    Nsteps = (20.f*tausquarewave/dt);

    bool changesign = true;
    int countsign = 0;

    for (int step = 0; step < Nsteps; ++step) {

        #ifdef MONITOR
        if(step % MONITOR == 0)
        {

	  #ifdef DWINTHEMIDDLEIC
          float mag = positives_in_region(phi, 10, N-10, 0, N);
	  #else
          float mag = positives_in_region(phi, 0, N, 0, N);
	  #endif


          float elastic_energy = ElasticEnergy(phi);

          monitor_out << t << " " << (mag-prevmag)/(N*dt*MONITOR) << " " << h << " " << elastic_energy << " " << mag << " " << changesign << std::endl;
          Vac += h * (mag-prevmag)/(N*dt*MONITOR);

          prevmag = mag;
        }
        #endif

        Laplacian2D lap_op{N, thrust::raw_pointer_cast(phi.data())};
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(size),
            laplace.begin(),
            lap_op
        );


        #ifdef PRINTCONFIGS
        //if (step % PRINTCONFIGS == 0) {
        if (changesign == true) {
            thrust::copy(phi.begin(), phi.end(), phi_host.begin());  // No new allocation
            write_field_to_file(phi_host, N, N, countsign);
        }
        #endif


        PhiUpdate update{c, epsilon0, gamma_, dt, h, noise_amp,
                         thrust::raw_pointer_cast(laplace.data()),
                         thrust::raw_pointer_cast(phi.data()),
                         thrust::raw_pointer_cast(r_disorder.data()),
                         static_cast<unsigned int>(step * 7919)};

        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(size),
            phi_new.begin(),
            update
        );

        // Swap pointers
        thrust::swap(phi, phi_new);

        t+=dt;

	#ifdef TAUSQUAREWAVE
	float hold = h;
    	h = (sin(2.0f*M_PI*t/tausquarewave)>0)?(1.0f):(-1.0f);
    	h *= ampsquarewave; //AMPSQUAREWAVE;
    	if(h*hold < 0.0f){changesign = true; countsign++;}
    	else changesign = false;
    	#endif
    }

    // Optional: copy to host and save
    thrust::host_vector<float> result = phi;
    // Save result here if needed...
    
    std::cout << ampsquarewave << " " << Vac*MONITOR/(Nsteps*ampsquarewave) << " " << DISORDERAMP << std::endl;
    //std::cout << "Simulation complete.\n";
    return 0;
}
