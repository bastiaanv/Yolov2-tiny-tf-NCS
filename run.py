from mvnc import mvncapi as mvnc
import NeuralNetwork
import cv2
import time

path_to_networks = './model/'
graph_filename = 'graph'
fps = 0.0
NN = NeuralNetwork.Net(video=True)

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
cap = cv2.VideoCapture(0)

while True:
	start = time.time()
	ret, display_image = cap.read()

	if not ret: 
		print("No image from from video device, exiting")
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
fifoIn.destroy()
fifoOut.destroy()
graph.destroy()
device.close()

print('Finished')
