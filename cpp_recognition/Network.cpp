#include "Network.h"

Network::Network() : session(nullptr), graph(nullptr) {}

TF_Buffer* Network::read_buffer_from_file(std::string file) {
	std::ifstream f(file, std::ios::binary);
	if (f.fail() || !f.is_open()) {
		return nullptr;
	}

	f.seekg(0, std::ios::end);
	const auto fsize = f.tellg();
	f.seekg(0, std::ios::beg);

	if (fsize < 1) {
		f.close();
		return nullptr;
	}

	char* data = static_cast<char*>(std::malloc(fsize));
	f.read(data, fsize);
	f.close();

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = [](void* data, size_t) {free(data); };
	return buf;
}

void Network::init_input(std::vector<TF_Output>& input, std::vector<TF_Tensor*>& value, cv::Mat& image) {
	int dims_n = 4;
	int batch_n = 1;
	int channels_n = 3;
	std::int64_t input_dims[4] = { batch_n, image.rows, image.cols, channels_n };
	int bytes_n = image.cols * image.rows * channels_n;

	input.push_back({ TF_GraphOperationByName(graph, "image_tensor"),0 });
	value.push_back(TF_NewTensor(TF_UINT8, input_dims, dims_n, image.data, bytes_n,
		[](void* data, size_t length, void* arg) {}, 0)
	);
}

void Network::init_output(std::vector<TF_Output>& output, TF_Tensor** values) {
	output.push_back({ TF_GraphOperationByName(graph, "detection_classes"),0 });
	output.push_back({ TF_GraphOperationByName(graph, "detection_scores"),0 });
	output.push_back({ TF_GraphOperationByName(graph, "detection_boxes"),0 });
	output.push_back({ TF_GraphOperationByName(graph, "num_detections"),0 });

	for (auto i = 0; i < 4; i++) {
		values[i] = nullptr;
	}
}

void Network::load_graph(std::string modelPath) {
	TF_Buffer* buffer = read_buffer_from_file(modelPath);
	if (buffer == nullptr) {
		throw std::invalid_argument("error creating the session from the given graph path");
	}

	TF_Status* status = TF_NewStatus();
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	graph = TF_NewGraph();
	TF_GraphImportGraphDef(graph, buffer, opts, status);
	TF_DeleteImportGraphDefOptions(opts);
	TF_DeleteBuffer(buffer);
	if (TF_GetCode(status) != TF_OK) {
		TF_DeleteGraph(graph);
		TF_DeleteStatus(status);
		graph = nullptr;
		throw std::invalid_argument("import graph error");
	}
	TF_DeleteStatus(status);

	status = TF_NewStatus();
	TF_SessionOptions* options = TF_NewSessionOptions();
	session = TF_NewSession(graph, options, status);
	TF_DeleteSessionOptions(options);
}

std::vector<Recognition> Network::run(cv::Mat& image, float confidence_trash) {
	std::vector<TF_Output> 	input_tensors;
	std::vector<TF_Tensor*> input_values;
	this->init_input(input_tensors, input_values, image);

	std::vector<TF_Output> output_tensors;
	auto output_values = new TF_Tensor * [4];
	this->init_output(output_tensors, output_values);

	TF_Status* status = TF_NewStatus();
	TF_SessionRun(session, nullptr,
		input_tensors.data(), input_values.data(), input_values.size(),
		output_tensors.data(), output_values, output_tensors.size(),
		nullptr, 0, nullptr, status
	);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "ERROR: SessionRun: " << TF_Message(status) << std::endl;
		return {};
	}
	auto detection_classes = static_cast<float_t*>(TF_TensorData(output_values[0]));
	auto detection_scores = static_cast<float_t*>(TF_TensorData(output_values[1]));
	auto detection_boxes = static_cast<float_t*>(TF_TensorData(output_values[2]));
	auto num_detections = static_cast<float_t*>(TF_TensorData(output_values[3]));

	std::vector<Recognition> result;
	for (int i = 0; i < *num_detections; i++) {
		if (detection_scores[i] > confidence_trash) {
			std::vector<float> coordinates{detection_boxes[4 * i + 1] * image.cols,
					detection_boxes[4 * i] * image.rows,
					(detection_boxes[4 * i + 3] - detection_boxes[4 * i + 1]) * image.cols,
					(detection_boxes[4 * i + 2] - detection_boxes[4 * i]) * image.rows };

			auto class_name = label_map[static_cast<int>(detection_classes[i]) - 1];
			result.emplace_back(Recognition{ class_name, static_cast<int>(detection_classes[i]), detection_scores[i], coordinates });

			cv::Rect rect = cv::Rect(detection_boxes[4 * i + 1] * image.cols, detection_boxes[4 * i] * image.rows, (detection_boxes[4 * i + 3] - detection_boxes[4 * i + 1]) * image.cols, (detection_boxes[4 * i + 2] - detection_boxes[4 * i]) * image.rows);
			rectangle(image, rect, cv::Scalar(0, 255, 0), 2, 8, 0);
			cv::putText(image, class_name, cv::Point(coordinates[0], coordinates[1]),
				cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);
		}
	}

	cv::imshow("test", image);
	cv::waitKey();

	TF_DeleteStatus(status);
	for (int i = 0; i < 4; i++) {
		TF_DeleteTensor(output_values[i]);
	}

	for (auto& t : input_values) {
		TF_DeleteTensor(t);
	}

	return result;
}

Network::~Network() {
	TF_DeleteGraph(graph);
	TF_Status* status = TF_NewStatus();
	TF_CloseSession(session, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "error close session" << std::endl;
	}
	TF_DeleteSession(session, status);
	TF_DeleteStatus(status);
}
