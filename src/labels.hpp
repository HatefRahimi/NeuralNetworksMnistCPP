#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace std;

class DataSetLabels {
private:
    size_t batch_size_;
    size_t number_of_labels_;
    std::vector<MatrixXd> batches_;

public:
    DataSetLabels(size_t batch_size) : batch_size_(batch_size) {}
    ~DataSetLabels() {}

    void readLabelData(const std::string& input_filepath);
    void writeAllLabelsToFile(const std::string& output_filepath);
    void printFirstBatch() const;
};

void DataSetLabels::readLabelData(const std::string& input_filepath) {
    std::ifstream input_file(input_filepath, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Unable to open file: " << input_filepath << std::endl;
        return;
    }

    char bin_data[4];
    int magic_number = 0;
    int number_of_labels = 0;

    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    std::memcpy(&magic_number, bin_data, sizeof(int));

    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    std::memcpy(&number_of_labels, bin_data, sizeof(int));
    number_of_labels_ = number_of_labels;

    Eigen::MatrixXd label_matrix(batch_size_, 10);
    label_matrix.setZero();

    for (size_t i = 0; i < number_of_labels_; ++i) {
        uint8_t byte = 0;
        input_file.read(reinterpret_cast<char*>(&byte), sizeof(byte));
        int label = static_cast<int>(byte);

        label_matrix(i % batch_size_, label) = 1;

        if ((i + 1) % batch_size_ == 0) {
            batches_.push_back(label_matrix);
            label_matrix.setZero();
        }
    }

    if (number_of_labels_ % batch_size_ != 0) {
        batches_.push_back(label_matrix);
    }

    input_file.close();
    std::cout << "Read " << number_of_labels_ << " labels into " << batches_.size() << " batches." << std::endl;
}

void DataSetLabels::writeAllLabelsToFile(const std::string& output_filepath) {
    std::ofstream output_file(output_filepath);
    if (!output_file.is_open()) {
        std::cerr << "Unable to open file: " << output_filepath << std::endl;
        return;
    }

    for (size_t batch_no = 0; batch_no < batches_.size(); ++batch_no) {
        for (size_t row = 0; row < batch_size_; ++row) {
            std::vector<int> one_hot_vector;
            for (size_t col = 0; col < 10; ++col) {
                int value = static_cast<int>(batches_[batch_no](row, col));
                one_hot_vector.push_back(value);
            }
            output_file << "[";
            for (size_t i = 0; i < one_hot_vector.size(); ++i) {
                output_file << one_hot_vector[i];
                if (i < one_hot_vector.size() - 1) {
                    output_file << ", ";
                }
            }
            output_file << "]\n";
        }
    }
    output_file.close();
    std::cout << "One-hot encoded labels written to " << output_filepath << std::endl;
}

void DataSetLabels::printFirstBatch() const {
    if (batches_.empty()) {
        std::cout << "No batches available." << std::endl;
        return;
    }
    std::cout << "First batch of labels (One-Hot Encoded):" << std::endl;
    std::cout << batches_[0] << std::endl;
}