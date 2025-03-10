#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using namespace std;

class DataSetImages {
private:
    size_t batch_size_;
    size_t number_of_images_;
    size_t number_of_rows_;
    size_t number_of_columns_;
    vector<MatrixXd> batches_;

public:
    DataSetImages(size_t batch_size) : batch_size_(batch_size) {}
    ~DataSetImages() {}

    void readImageData(const string& input_filepath);
    void writeImageToFile(const string& output_filepath, const size_t& index);
    size_t getBatchSize();
    MatrixXd getBatch(const size_t& index);
};

size_t DataSetImages::getBatchSize() {
    return batches_.size();
}

MatrixXd DataSetImages::getBatch(const size_t &index) {
    return batches_[index];
}

void DataSetImages::readImageData(const string& input_filepath) {
    ifstream input_file(input_filepath, ios::binary);
    if (!input_file.is_open()) {
        cerr << "Unable to open file: " << input_filepath << endl;
        return;
    }

    char bin_data[4];
    int magic_number = 0;
    int number_of_images = 0;
    int number_of_rows = 0;
    int number_of_columns = 0;

    input_file.read(bin_data, 4);
    reverse(bin_data, end(bin_data));
    memcpy(&magic_number, bin_data, sizeof(int));

    input_file.read(bin_data, 4);
    reverse(bin_data, end(bin_data));
    memcpy(&number_of_images, bin_data, sizeof(int));
    number_of_images_ = number_of_images;

    input_file.read(bin_data, 4);
    reverse(bin_data, end(bin_data));
    memcpy(&number_of_rows, bin_data, sizeof(int));
    number_of_rows_ = number_of_rows;

    input_file.read(bin_data, 4);
    reverse(bin_data, end(bin_data));
    memcpy(&number_of_columns, bin_data, sizeof(int));
    number_of_columns_ = number_of_columns;

    size_t image_size = number_of_rows_ * number_of_columns_;
    unsigned char* image_bin = new unsigned char[image_size];
    double* image = new double[image_size];
    MatrixXd image_matrix(batch_size_, image_size);
    size_t remaining_images = number_of_images_ % batch_size_;

    for (size_t i = 0; i < number_of_images_; ++i) {
        input_file.read(reinterpret_cast<char*>(image_bin), image_size);
        std::transform(image_bin, image_bin + image_size, image, [](unsigned char c) {
            return static_cast<double>(c) / 255.0;
        });
        image_matrix.row(i % batch_size_) = Eigen::Map<Eigen::VectorXd>(image, image_size);

        if ((i + 1) % batch_size_ == 0) {
            batches_.push_back(image_matrix.block(0,0,batch_size_, image_size));
        }
        else if (i == number_of_images_ - 1) {
            batches_.push_back(image_matrix.block(0, 0, batch_size_, image_size));
        }
    }

    delete[] image_bin;
    delete[] image;
    input_file.close();
}

void DataSetImages::writeImageToFile(const std::string& output_filepath, const size_t& index) {
    size_t batch_no = index / batch_size_;
    size_t image_index = index % batch_size_;
    std::ofstream output_file(output_filepath);

    if (output_file.is_open()) {
        output_file << 2 << "\n";
        output_file << number_of_rows_ << "\n";
        output_file << number_of_columns_ << "\n";

        size_t image_size = number_of_rows_ * number_of_columns_;
        for (size_t i = 0; i < image_size; ++i) {
            output_file << batches_[batch_no](image_index, i) << "\n";
        }
        output_file.close();
        cout << "Image data written to " << output_filepath << " in tensor format." << std::endl;
    }
    else {
        cerr << "Error: Unable to open file for writing: " << output_filepath << std::endl;
    }
}
