#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>  // <-- required for std::setprecision


const int N = 1024;              // Grid size (NxN)
const int Nsteps = 5000;        // Time steps
const float dt = 0.01f;
const float gamma_ = 1.0f;
const float c = 1.0f;
const float epsilon0 = 1.0f;
const float h = 0.0f;
const float noise_amp = 0.0f;
const float disorder_amp = 0.0f;


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


struct Laplacian2D {
    int N;
    const float* phi;
    __host__ __device__
    float operator()(int idx) const {
        int i = idx / N;
        int j = idx % N;
        int ip = (i + 1) % N, im = (i - 1 + N) % N;
        int jp = (j + 1) % N, jm = (j - 1 + N) % N;
        return phi[im*N + j] + phi[ip*N + j] + phi[i*N + jm] + phi[i*N + jp] - 4.0f * phi[idx];
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

int main() {
    int size = N * N;
    thrust::device_vector<float> phi(size);
    thrust::device_vector<float> phi_new(size);
    thrust::device_vector<float> r_disorder(size);
    thrust::device_vector<float> laplace(size);

    // Initialize phi randomly in [-1,1]
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

    // Initialize disorder r(x,y)
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(size),
        r_disorder.begin(),
        [] __host__ __device__ (int i) {
            thrust::default_random_engine rng(5678);
            thrust::uniform_real_distribution<float> dist(-disorder_amp, disorder_amp);
            rng.discard(i);
            return dist(rng);
        }
    );

    thrust::host_vector<float> phi_host(size);
    int nprint = 100;

    for (int step = 0; step < Nsteps; ++step) {

        if (step % nprint == 0) {
            thrust::copy(phi.begin(), phi.end(), phi_host.begin());  // No new allocation
            write_field_to_file(phi_host, N, N, step);
        }

        Laplacian2D lap_op{N, thrust::raw_pointer_cast(phi.data())};
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(size),
            laplace.begin(),
            lap_op
        );

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
    }

    // Optional: copy to host and save
    thrust::host_vector<float> result = phi;
    // Save result here if needed...

    std::cout << "Simulation complete.\n";
    return 0;
}
