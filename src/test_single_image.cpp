#include <iostream>
#include <fstream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class test_single_image {
public:
    test_single_image(int rows, int cols);
    test_single_image() = default;
    MatrixXd readMNISTImage(const string& filename);
    void printBinaryMNISTImage(const MatrixXd& image) const;

private:
    int rows;
    int cols;
};

test_single_image::test_single_image(int rows, int cols) : rows(rows), cols(cols) {}

MatrixXd test_single_image::readMNISTImage(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not open image file: " << filename << endl;
        exit(1);
    }

    file.ignore(16); // Skip the 16-byte MNIST header

    MatrixXd image(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image(i, j) = static_cast<double>(pixel) / 255.0;
        }
    }

    file.close();
    return image;
}

void test_single_image::printBinaryMNISTImage(const MatrixXd& image) const {
    cout << "\nMNIST Image Representation (Binary 1s and 0s):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << (image(i, j) > 0.5 ? "1" : "0");
        }
        cout << endl;
    }
}

int main() {
    string image_file = R"(C:\Users\hatef\NeuralNetworksCPP\mnist-datasets\single-image.idx3-ubyte)";

    test_single_image mnistReader(28, 28);
    MatrixXd image = mnistReader.readMNISTImage(image_file);
    mnistReader.printBinaryMNISTImage(image);

    return 0;
}
