#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include "clew.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/compute/system.hpp>
#include <boost/compute/interop/opencv/core.hpp>
#include <boost/compute/interop/opencv/highgui.hpp>
#include <boost/compute/utility/source.hpp>

#include <iostream>
#define FRAME_HEIGHT 720
#define FRAME_WIDTH 1280
#define FRAME_INTERVAL (1000/30)

namespace compute = boost::compute;
using namespace cv;

compute::image2d process_image(cv::Mat input, const char *source, compute::context &context, compute::command_queue &queue)
{


    // convert image to BGRA (OpenCL requires 16-byte aligned data)
    cv::cvtColor(input, input, CV_BGR2BGRA);

    // transfer image to gpu
    compute::image2d input_image =
        compute::opencv_create_image2d_with_mat(
            	input, compute::image2d::read_write, queue
        		);

    // create output image
    compute::image2d output_image(
        	context,
        	input_image.width(),
        	input_image.height(),
        	input_image.format(),
        	compute::image2d::write_only
    		);


    compute::program flip_program =
        compute::program::create_with_source(source, context);
    flip_program.build();

    // create flip kernel and set arguments
    compute::kernel flip_kernel(flip_program, "flip_kernel");
    flip_kernel.set_arg(0, input_image);
    flip_kernel.set_arg(1, output_image);

    // run flip kernel
    size_t origin[2] = { 0, 0 };
    size_t region[2] = { input_image.width(), input_image.height() };
    queue.enqueue_nd_range_kernel(flip_kernel, 2, origin, region, 0);

    return output_image;
}

int main(int argc, const char *argv[])
{
    bool clpresent = 0 == clewInit();
    if( !clpresent ) {
		std::cerr << "opencl library not found" << std::endl;
		return -1;
    }

    // check command line
    if(argc < 2){
        std::cerr << "usage: " << argv[0] << " FILENAME" << std::endl;
        return -1;
    }


    // read image with opencv
    cv::Mat cv_image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!cv_image.data){
        std::cerr << "failed to load image" << std::endl;
        return -1;
    }
    // get default device and setup context
    compute::device gpu = compute::system::default_device();
    std::cout << "OpenCL Device: " << gpu.name() << std::endl;
    std::cout << "OpenCL Profile: " << gpu.profile() << std::endl;
    std::cout << "OpenCL Version: " << gpu.version() << std::endl;
    compute::context context(gpu);
    compute::command_queue queue(context, gpu);

    // create flip program
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        	__kernel void flip_kernel(__read_only image2d_t input,
                __write_only image2d_t output)
        	{
            const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
            int height = get_image_height(input);
            int2 input_coord = { get_global_id(0), get_global_id(1) };
            int2 output_coord = { input_coord.x, height - input_coord.y - 1 };
            float4 value = read_imagef(input, sampler, input_coord);
            write_imagef(output, output_coord, value);
        	}
    		);


    VideoCapture cap(0); // Grab the camera
    if (!cap.isOpened()) {
		std::cerr << "OpenCV Failed to open camera";
        exit(1);
    }

	while(1) {
    	Mat frame, send;
    	cap >> frame;
    	if(frame.size().width==0)continue;//simple integrity check; skip erroneous data...
    	resize(frame, send, Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_LINEAR);

		auto output_image = process_image(send, source, context, queue);

    	// show host image
    	//cv::imshow("opencv image", cv_image);

    	// show gpu image
    	compute::opencv_imshow("filtered image", output_image, queue);

    	// wait and return
		cv::waitKey(FRAME_INTERVAL);
	}

	return 0;
}
