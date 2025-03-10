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

    // The vector 'batches_' is a vector of matrices
    // Each matrix in 'batches_' represents a batch of labels
    // Each matrix has 'batch_size' rows and 10 columns
    // Each row is a one-hot encoded vector of a label
    // Each column corresponds to a digit from 0 to 9

    vector<MatrixXd> batches_;

public:
    DataSetLabels(size_t batch_size) : batch_size_(batch_size) {}
    ~DataSetLabels() {}

    void readLabelData(const string& input_filepath);
    void writeAllLabelsToFileWithBrackets(const string& output_filepath);
    void writeLabelToFile(const std::string &, const size_t &);
    void printFirstBatch() const;
    MatrixXd getBatch(const size_t& index);
    size_t getBatchSize();
};

MatrixXd DataSetLabels::getBatch(const size_t& index){

    return batches_[index];
}

size_t DataSetLabels::getBatchSize() {
    return batches_.size();
}


void DataSetLabels::readLabelData(const std::string& input_filepath) {
    ifstream input_file(input_filepath, std::ios::binary);
    if (!input_file.is_open()) {
        cerr << "Unable to open file: " << input_filepath << endl;
        return;
    }

    char bin_data[4];
    int magic_number = 0;
    int number_of_labels = 0;

    input_file.read(bin_data, 4);
    reverse(bin_data, bin_data + 4);
    memcpy(&magic_number, bin_data, sizeof(int));

    input_file.read(bin_data, 4);
    reverse(bin_data, bin_data + 4);
    memcpy(&number_of_labels, bin_data, sizeof(int));
    number_of_labels_ = number_of_labels;

    Eigen::MatrixXd label_matrix(batch_size_, 10);
    label_matrix.setZero();

    for (size_t i = 0; i < number_of_labels_; ++i) {
        uint8_t byte = 0;
        input_file.read(reinterpret_cast<char*>(&byte), sizeof(byte));
        int label =static_cast<int> (byte);

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
}

void DataSetLabels::writeAllLabelsToFileWithBrackets(const std::string& output_filepath) {
    ofstream output_file(output_filepath);
    if (!output_file.is_open()) {
        cerr << "Unable to open file: " << output_filepath << endl;
        return;
    }

    for (size_t batch_no = 0; batch_no < batches_.size(); ++batch_no) {
        for (size_t row = 0; row < batch_size_; ++row) {
            vector<int> one_hot_vector;
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
    cout << "One-hot encoded labels written to " << output_filepath << std::endl;
}

void DataSetLabels::writeLabelToFile(const std::string &output_filepath, const size_t &index)
{
    // Write in this format to be compatible with this format
    // 1
    // 10
    // 0 if label is 0
    // 1 if label is 1
    // label 0, 1 should be written for respective digit entry
    std::ofstream output_file(output_filepath, std::ios::binary);
    if (output_file.is_open())
    {

        // Write the rank of the tensor to the file
        output_file << 1 << "\n";

        // Write the shape of the tensor to the file
        output_file << 10 << "\n";

        // Write the tensor elements to the file
        size_t batch_no = index / batch_size_;
        size_t image_index = index % batch_size_;
        size_t image_size = 10;
        for (size_t i = 0; i < image_size; ++i)
        {
            output_file << batches_[batch_no](image_index, i) << "\n";
        }
        output_file.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }
}

