#pragma once

#include <iostream>
#include <codecvt>
#include <fstream>
#include <vector>
#include <cstring>

#include "tensorflow/c/c_api.h"

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/types.hpp>
#include<opencv2/imgproc.hpp>


struct Recognition {
	std::string name;
	int class_id;
	float confidence;
	std::vector<float> coordinates;
};

class Network {

public:
	void load_graph(std::string modelPath);
	std::vector<Recognition> run(cv::Mat& image_record, float confidence_trash=0.8);

	Network();
	~Network();
private:
	TF_Session* session;
	TF_Graph* graph;
	std::vector<std::string> label_map = { "cat", "striped", "squared", "tie",
		"bow", "glasses", "beard" };

	void init_input(std::vector<TF_Output>& input, std::vector<TF_Tensor*>& value, cv::Mat& image);
	void init_output(std::vector<TF_Output>& output, TF_Tensor** value);
	
	TF_Buffer* read_buffer_from_file(std::string file);
};
