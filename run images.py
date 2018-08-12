from mvnc import mvncapi as mvnc
import NeuralNetwork
import cv2

path_to_networks = './model/'
graph_filename = 'graph'
image = cv2.imread("./images/Monitor.jpg")
NN = NeuralNetwork.Net()

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

print('generate input and upload to NCS...')
inputs = NN.preproces_image(image)
graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, inputs, 'user object')

print('Get prediction and process image...')
prediction, _ = fifoOut.read_elem()
output_image, boxes = NN.postprocess(prediction, image, 0.3, 0.3)
cv2.imshow(NN.cv_window_name, output_image)
cv2.waitKey(0)

fifoIn.destroy()
fifoOut.destroy()
graph.destroy()
device.close()

print('Finished')
