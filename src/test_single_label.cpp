#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class mnist_label_reader {
public:
    int readMNISTLabel(const string& filename);
};

int mnist_label_reader::readMNISTLabel(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not open label file: " << filename << endl;
        exit(1);
    }

    file.ignore(8); // Skip the 8-byte MNIST label header

    unsigned char label = 0;
    file.read(reinterpret_cast<char*>(&label), 1); // Read 1 byte as the label

    file.close();
    return static_cast<int>(label); // Return the label as an integer (0-9)
}

int main() {

    string label_file = R"(C:\Users\hatef\NeuralNetworksCPP\mnist-datasets\single-label.idx1-ubyte)";

    mnist_label_reader labelReader;
    int label = labelReader.readMNISTLabel(label_file);

    cout << "Label: " << label << endl;

    return 0;
}
