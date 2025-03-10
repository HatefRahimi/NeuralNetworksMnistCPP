#include <iostream>
#include "data.hpp"

using namespace std;

int main(int argc, char* argv[]) {

    string input_filepath = argv[1];
    string output_filepath = argv[2];
    size_t index = stoi(argv[3]);

    DataSetImages dataset(5000); // Batch size
    dataset.readImageData(input_filepath);
    dataset.writeImageToFile(output_filepath, index); // Write the first image to file

    return 0;
}
