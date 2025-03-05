#include <iostream>
#include "data.hpp"

using namespace std;

int main() {

    DataSetImages dataset(5000); // Batch size
    dataset.readImageData("mnist-datasets/train-images.idx3-ubyte");
    dataset.writeImageToFile("image_out.txt", 0); // Write the first image to file
    cout << "Image data written to image_out.txt" << endl;

    std::cout << dataset.getBatchSize(); //12
    return 0;
}
