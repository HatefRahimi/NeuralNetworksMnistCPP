#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

// Function to read a single MNIST image
std::vector<std::vector<double>> readMNISTImage(const std::string& filename, int rows = 28, int cols = 28) {
    std::ifstream file;
    file.open(filename,ios::in);
    if (!file.is_open()) {
        std::cerr << "Error opening image file: " << filename << std::endl;
        exit(1);
    }

    file.ignore(16); // Skip the header (16 bytes)

    std::vector<std::vector<double>> image(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image[i][j] = pixel / 255.0; // Normalize to [0,1]
        }
    }

    file.close();
    return image;
}

// Function to read a single MNIST label
int readMNISTLabel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    file.open(filename,ios::in);
    if (!file.is_open()) {
        std::cerr << "Error opening label file: " << filename << std::endl;
        exit(1);
    }

    file.ignore(8); // Skip the header (8 bytes)

    unsigned char label = 0;
    file.read(reinterpret_cast<char*>(&label), 1); // Read single label

    file.close();
    return static_cast<int>(label);
}

// Function to print the image using ASCII characters
void printMNISTImage(const std::vector<std::vector<double>>& image) {
    std::cout << "\nMNIST Image Representation:\n";
    for (const auto& row : image) {
        for (double pixel : row) {
            if (pixel > 0.75) std::cout << "█";  // Dark pixel
            else if (pixel > 0.50) std::cout << "▓";
            else if (pixel > 0.25) std::cout << "░";
            else std::cout << " ";  // Light pixel
        }
        std::cout << std::endl;
    }
}

int main() {
    // // Updated paths to reflect your project structure
    // std::string image_file = "C:\\Users\\hatef\\TestGround\\single-image.idx3-ubyte";
    // std::string label_file = "C:\\Users\\hatef\\TestGround\\single-label.idx1-ubyte";
    //
    // // Read image and label
    // std::vector<std::vector<double>> image = readMNISTImage(image_file);
    // int label = readMNISTLabel(label_file);
    //
    // // Print label and image
    // std::cout << "\nLabel: " << label << std::endl;
    // printMNISTImage(image);
    //
    // return 0;
    std::string label_file = R"(C:\Users\hatef\TestGround\single-label.idx3-ubyte)";

    // Declare file stream
    std::ifstream file;

    // Explicitly open the file in binary mode
    file.open(label_file, std::ios::binary);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open label file!" << std::endl;
        return 1;
    }

    file.ignore(8); // Skip the 8-byte header

    unsigned char label = 0;
    file.read(reinterpret_cast<char*>(&label), 1); // Read 1 byte

    file.close(); // Explicitly close the file

    // Print the label
    std::cout << "Label: " << static_cast<int>(label) << std::endl;

    return 0;
}
