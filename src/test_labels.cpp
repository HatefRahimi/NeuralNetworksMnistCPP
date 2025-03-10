#include <iostream>
#include <fstream>
#include "labels.hpp"

using namespace std;

int main(int argc, char* argv[]) {

    std::string input_filepath = argv[1];
    std::string output_filepath = argv[2];
    size_t index = std::stoi(argv[3]);

    DataSetLabels dataset_labels(5000);
    dataset_labels.readLabelData(input_filepath);
    dataset_labels.writeLabelToFile(output_filepath, index);

    return 0;
}
