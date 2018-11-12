from mvnc import mvncapi as mvnc
import NeuralNetwork
import cv2
import argparse
import time
import threading

#Argument parser
arg = argparse.ArgumentParser()
arg.add_argument("-m", "--mode", required=True, type=str, default="image", help="Mode of Neural Network, options: image, video")
arg.add_argument("-n", "--num", required=False, type=int, default=1, help="Number of NCS you want to use")
arg.add_argument("-i", "--image", required=False, type=str, help="The path to the image you want to process")
arg.add_argument("-v", "--video", required=False, help="The path to the video you want to process or enter a integer if you want to use your webcam")
args = vars( arg.parse_args() )

#Neural network
video_mode = True if args["mode"] == "video" else False
NN = NeuralNetwork.Net( video = video_mode )

#Intel's Neural Compute Stick
mvnc.global_set_option( mvnc.GlobalOption.RW_LOG_LEVEL, 2 )
devices = mvnc.enumerate_devices()
if len(devices) == 0:
	print( "No devices found..." )
	quit()
elif args["num"] > len(devices):
	print( "There aren't that many NCS's available..." )
	quit() 
elif args["num"] == 0:
	print( "One NCS is required to run..." )
	quit()


with open( './model/graph', mode='rb' ) as f:
	graphfile = f.read()
	graph = mvnc.Graph( 'graph' )
	
	


class feed_forward_thread( threading.Thread ):
	def __init__( self, device, args, NN, graph, delay=0, video=False ):
		threading.Thread.__init__( self )
		self.device = None
		self.fifoIn = None
		self.fifoOut = None
		self.video_mode = video
		self.args = args
		self.NN = NN
		self.graph = graph
		self.delay = delay

		self.open_device_load_graph( device )

	def open_device_load_graph( self, device ):
		self.device = mvnc.Device( device )
		self.device.open()
		self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos( self.device, graphfile )
		
	def run( self ):
		if self.delay > 0:
			time.sleep( self.delay )

		if self.video_mode:
			fps = 0.0

			#Webcam mode, else video file mode
			if self.args["video"].isdigit():
				self.args["video"] = int( self.args["video"]) 

			cap = cv2.VideoCapture( self.args["video"] )

			while True:
				start = time.time()
				ret, display_image = cap.read()

				if not ret: 
					print( "No image found from source, exiting" )
					break

				output_image, boxes = self.run_interference( display_image )

				fps  = ( fps + ( 1 / (time.time() - start) ) ) / 2
				output_image = cv2.putText( output_image, "fps: {:.1f}".format(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4 )

				cv2.imshow( self.NN.cv_window_name, output_image )
	
				if cv2.getWindowProperty( self.NN.cv_window_name, cv2.WND_PROP_ASPECT_RATIO ) < 0.0:
					print( "Window closed" )
					break
				elif cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
					print( "Q pressed" )
					break


			cap.release()
			cv2.destroyAllWindows()

		else:
			start = time.time()
			image = cv2.imread( self.args["image"] )
			output_image, boxes = self.run_interference( image )

			print( "Time took: {:.1f} sec".format(time.time() - start) )

			cv2.imshow( self.NN.cv_window_name, output_image )
			cv2.waitKey( 0 )

		#Close device and with it the thread
		self.graph.destroy()
		self.fifoIn.destroy()
		self.fifoOut.destroy()
		self.device.close()


	def run_interference( self, image ):
		resize_image, inputs = self.NN.preproces_image( image )
		self.graph.queue_inference_with_fifo_elem( self.fifoIn, self.fifoOut, inputs, 'user object' )

		prediction, _ = self.fifoOut.read_elem()
		return self.NN.postprocess( prediction, resize_image, 0.3, 0.3 )


#Run script
threads = []
delay = 0
for i in range(args["num"]):
	threads.append( feed_forward_thread( devices[i], args, NN, graph, delay=delay, video=video_mode) ) 
	delay += (170/(args["num"]*(i+1)))

#run thread
for thread in threads:
	thread.start()

#wait until threads are done
for thread in threads:
	thread.join()

#Done!!
print('Finished')
