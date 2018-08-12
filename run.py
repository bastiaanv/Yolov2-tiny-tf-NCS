from mvnc import mvncapi as mvnc
import NeuralNetwork
import cv2
import argparse
import time

arg = argparse.ArgumentParser()
arg.add_argument("-m", "--mode", required=True, help="Mode of Neural Network, options: image, video")
arg.add_argument("-i", "--image", required=False, help="The path to the image you want to process")
arg.add_argument("-v", "--video", required=False, help="The path to the video you want to process or parse 0 if you want to use your webcam")
args = vars(arg.parse_args())

path_to_networks = './model/'
graph_filename = 'graph'
video_mode = True if args["mode"] == "video" else False

NN = NeuralNetwork.Net( video = video_mode )

mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)
devices = mvnc.enumerate_devices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.open()

print('Load graph...')
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()

graph = mvnc.Graph('graph')
fifoIn, fifoOut = graph.allocate_with_fifos(device, graphfile)

if video_mode:
	fps = 0.0

	#Webcam mode, else video file mode
	if args["video"].isdigit():
		args["video"] = int(args["video"])

	cap = cv2.VideoCapture(args["video"])

	while True:
		start = time.time()
		ret, display_image = cap.read()

		if not ret: 
			print("No image found from source, exiting")
			break

		inputs = NN.preproces_image(display_image)
		graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, inputs, 'user object')
		prediction, _ = fifoOut.read_elem()
		output_image, boxes = NN.postprocess(prediction, display_image, 0.3, 0.3)

		fps  = ( fps + ( 1 / (time.time() - start) ) ) / 2
		output_image = cv2.putText(output_image, "fps: {:.2f}".format(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4)

		cv2.imshow(NN.cv_window_name, output_image)
	
		if cv2.getWindowProperty(NN.cv_window_name, cv2.WND_PROP_ASPECT_RATIO) < 0.0:
			print("Window closed")
			break
		elif cv2.waitKey(1) & 0xFF == ord('q'):
			print("Q pressed")
			break


	cap.release()
	cv2.destroyAllWindows()

else:
	image = cv2.imread(args["image"])
	inputs = NN.preproces_image(image)

	graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, inputs, 'user object')
	prediction, _ = fifoOut.read_elem()
	output_image, boxes = NN.postprocess(prediction, image, 0.3, 0.3)

	cv2.imshow(NN.cv_window_name, output_image)
	cv2.waitKey(0)

fifoIn.destroy()
fifoOut.destroy()
graph.destroy()
device.close()

print('Finished')
