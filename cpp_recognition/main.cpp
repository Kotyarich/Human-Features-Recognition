#include "Network.h"

int main() {
	std::string modelPath = "C:\\tensorflow1\\models\\research\\object_detection\\inference_graph\\frozen_inference_graph.pb";
	std::string imagePath = "C:\\tensorflow1\\models\\research\\object_detection\\test_images\\1_1.png";

	Network ssd_network;
	ssd_network.load_graph(modelPath);

	cv::Mat image;
	image = cv::imread(imagePath);
	if (image.data == NULL) {
		std::cout << "image read ERROR" << std::endl;
		return 1;
	}
	image.convertTo(image, CV_8UC3);

	for (size_t i = 0; i < 1; i++) {
		ssd_network.run(image);
	}
}