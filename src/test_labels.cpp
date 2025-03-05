#include <iostream>
#include <fstream>
#include "labels.hpp"

using namespace std;

int main() {

    DataSetLabels dataset_labels(4);
    dataset_labels.readLabelData("mnist-datasets/train-labels.idx1-ubyte");

    dataset_labels.writeAllLabelsToFile("labels_out.txt");
    cout << "Image and label data loaded successfully." << endl;

    return 0;
}
