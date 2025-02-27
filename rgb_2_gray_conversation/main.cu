#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <sstream>
// Convert RGB image into grayscale
__global__ void rgb2gray_kernel(float* d_rgb, float* d_gray, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float r = d_rgb[idx * 3];
        float g = d_rgb[idx * 3 + 1];
        float b = d_rgb[idx * 3 + 2];
        d_gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// Load a single channel from a CSV file
void read_channel_csv(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        float val;
        while (ss >> val) {
            data.push_back(val);
            if (ss.peek() == ',') ss.ignore();
        }
    }
    file.close();
}

// Load RGB data from separate channel CSV files
void read_rgb_csv(const std::string& r_file, const std::string& g_file, const std::string& b_file, 
                  std::vector<float>& data, int& width, int& height) {
    std::vector<float> r_data, g_data, b_data;
    read_channel_csv(r_file, r_data);
    read_channel_csv(g_file, g_data);
    read_channel_csv(b_file, b_data);

    if (r_data.empty() || g_data.empty() || b_data.empty()) {
        std::cerr << "Error: One or more CSV files are empty or not read correctly!\n";
        return;
    }

    // Debug print - Show first 10 RGB values
    std::cout << "First 10 RGB values:\n";
    for (int i = 0; i < std::min(10, (int)r_data.size()); ++i) {
        std::cout << "R: " << r_data[i] << ", G: " << g_data[i] << ", B: " << b_data[i] << std::endl;
    }

    int total_pixels = r_data.size();
    width = height = static_cast<int>(sqrt(total_pixels));

    for (int i = 0; i < total_pixels; ++i) {
        data.push_back(r_data[i]);
        data.push_back(g_data[i]);
        data.push_back(b_data[i]);
    }
}


// Write grayscale image to CSV
void write_csv(const std::string& filename, const std::vector<float>& data, int width, int height) {
    std::ofstream file(filename);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            file << data[i * width + j];
            if (j < width - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    std::string r_file = "r.csv";
    std::string g_file = "g.csv";
    std::string b_file = "b.csv";
    std::string output_file = "output_gray.csv";
    
    std::vector<float> h_rgb;
    int width, height;
    read_rgb_csv(r_file, g_file, b_file, h_rgb, width, height);

    int total_pixels = width * height;
    std::vector<float> h_gray(total_pixels);
    
    // Device memory allocation
    float *d_rgb, *d_gray;
    cudaMalloc(&d_rgb, h_rgb.size() * sizeof(float));
    cudaMalloc(&d_gray, total_pixels * sizeof(float));

    cudaError_t err1 =  cudaMemcpy(d_rgb, h_rgb.data(), h_rgb.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err1 != cudaSuccess) {
    	std::cerr << "CUDA malloc d_rgb failed: " << cudaGetErrorString(err1) << std::endl;
    	return -1;
	}

    std::vector<float> debug_rgb(30);
    cudaMemcpy(debug_rgb.data(), d_rgb, 30 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "First 10 RGB values in GPU memory:\n";
    for (int i = 0; i < 10; ++i) {
    	std::cout << "R: " << debug_rgb[i * 3] 
              << ", G: " << debug_rgb[i * 3 + 1] 
              << ", B: " << debug_rgb[i * 3 + 2] << std::endl;
	}

    // Kernel execution
    int threads_per_block = 256;
    int num_blocks = (total_pixels + threads_per_block - 1) / threads_per_block;
    rgb2gray_kernel<<<num_blocks, threads_per_block>>>(d_rgb, d_gray, total_pixels);
    
    // Copy grayscale data from device to host
    cudaMemcpy(h_gray.data(), d_gray, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write grayscale image to CSV
    write_csv(output_file, h_gray, width, height);
    
    cudaFree(d_rgb);
    cudaFree(d_gray);

    std::cout << "Grayscale image written to " << output_file << "\n";
    return 0;
}

