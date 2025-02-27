#include <iostream>
#include <fstream>
#include <vector>

// Function to read a single MNIST image from a binary file
std::vector<std::vector<double>> readMNISTImage(const std::string& filename, int rows = 28, int cols = 28) {
    std::ifstream file;
    file.open(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open image file: " << filename << std::endl;
        exit(1);
    }

    file.ignore(16); // Skip the 16-byte MNIST header

    std::vector<std::vector<double>> image(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image[i][j] = pixel / 255.0; // Normalize pixel to [0,1]
        }
    }

    file.close();
    return image;
}

// Function to print the MNIST image using 1s and 0s
void printBinaryMNISTImage(const std::vector<std::vector<double>>& image) {
    std::cout << "\nMNIST Image Representation (Binary 1s and 0s):\n";
    for (const auto& row : image) {
        for (double pixel : row) {
            if (pixel > 0.5) std::cout << "1";  // White (Foreground)
            else std::cout << "0";             // Black (Background)
        }
        std::cout << std::endl; // New line after each row
    }
}

int main() {
    std::string image_file = "C:\\Users\\hatef\\TestGround\\single-image.idx3-ubyte"; // Adjust if needed

    std::vector<std::vector<double>> image = readMNISTImage(image_file);
    printBinaryMNISTImage(image);  // Print image using 1s and 0s

    return 0;
}
