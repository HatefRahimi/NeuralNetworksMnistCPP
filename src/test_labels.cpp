#include <iostream>
#include <fstream>
#include "labels.hpp"

using namespace std;

int main() {

    DataSetLabels dataset_labels(4);
    dataset_labels.readLabelData("mnist-datasets/train-labels.idx1-ubyte");

    dataset_labels.writeAllLabelsToFile("labels_out.txt");
    std::cout << "Image and label data loaded successfully." << std::endl;

    return 0;
}
