import numpy as np
import tensorflow as tf
import cv2
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Net:

	classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
	colors = {
		"aeroplane": (254.0, 254.0, 254), 
		"bicycle": (239.9, 211.7, 127), 
		"bird": (225.8, 169.3, 0), 
		"boat": (211.7, 127.0, 254), 
		"bottle": (197.6, 84.7, 127), 
		"bus": (183.4, 42.3, 0),
		"car": (255, 0, 0), 
		"cat": (155.2, -42.3, 127), 
		"chair": (141.1, -84.7, 0), 
		"cow": (127.0, 254.0, 254), 
		"diningtable": (112.9, 211.7, 127), 
		"dog": (98.8, 169.3, 0), 
		"horse": (84.7, 127.0, 254), 
		"motorbike": (70.6, 84.7, 127), 
		"person": (56.4, 42.3, 0), 
		"pottedplant": (42.3, 0, 254), 
		"sheep": (28.2, -42.3, 127), 
		"sofa": (14.1, -84.7, 0), 
		"train": (0, 254, 254), 
		"tvmonitor": (-14.1, 211.7, 127)
	}

	image = tf.placeholder(tf.float32, shape=[1, 416, 416, 3], name="Input")

	def leaky_relu(self, x, alpha=0.1):
		return tf.nn.leaky_relu(x, alpha=alpha)

	def sigmoid(self, x):
		return 1. / (1. + np.exp(-x))

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		out = e_x / e_x.sum()
		return out

	def max_pool_layer(self, input_tensor, kernel=2, stride=2, padding='VALID'):
		pooling_result = tf.nn.max_pool(input_tensor, ksize=[1,kernel, kernel, 1], strides=[1, stride, stride, 1], padding=padding)
		return pooling_result

	def load_conv_layer_bn(self, name, loaded_weights, shape, offset):
		# Conv layer with Batch norm

		n_kernel_weights = shape[0]*shape[1]*shape[2]*shape[3]
		n_output_channels = shape[-1]
		n_bn_mean = n_output_channels
		n_bn_var = n_output_channels
		n_biases = n_output_channels
		n_bn_gamma = n_output_channels

		n_weights_conv_bn = (n_kernel_weights + n_output_channels * 4)
		biases = loaded_weights[offset:offset+n_biases]
		offset = offset + n_biases
		gammas = loaded_weights[offset:offset+n_bn_gamma]
		offset = offset + n_bn_gamma
		means = loaded_weights[offset:offset+n_bn_mean]
		offset = offset + n_bn_mean
		var = loaded_weights[offset:offset+n_bn_var]
		offset = offset + n_bn_var
		kernel_weights = loaded_weights[offset:offset+n_kernel_weights]
		offset = offset + n_kernel_weights

		# IMPORTANT: DarkNet conv_weights are serialized Caffe-style: (out_dim, in_dim, height, width)
		kernel_weights = np.reshape(kernel_weights,(shape[3],shape[2],shape[0],shape[1]),order='C')

		# IMPORTANT: Denormalize the weights with the Batch Normalization parameters
		for i in range(n_output_channels):
			scale = gammas[i] / np.sqrt(var[i] + self.bn_epsilon)
			kernel_weights[i,:,:,:] = kernel_weights[i,:,:,:] * scale
			biases[i] = biases[i] - means[i] * scale

		# IMPORTANT: Set weights to Tensorflow order: (height, width, in_dim, out_dim)
		kernel_weights = np.transpose(kernel_weights,[2,3,1,0])

		return biases,kernel_weights,offset

	def load_conv_layer(self, name, loaded_weights, shape, offset):
		# Conv layer without Batch norm

		n_kernel_weights = shape[0]*shape[1]*shape[2]*shape[3]
		n_output_channels = shape[-1]
		n_biases = n_output_channels

		n_weights_conv = (n_kernel_weights + n_output_channels)
		biases = loaded_weights[offset:offset+n_biases]
		offset = offset + n_biases
		kernel_weights = loaded_weights[offset:offset+n_kernel_weights]
		offset = offset + n_kernel_weights

		# IMPORTANT: DarkNet conv_weights are serialized Caffe-style: (out_dim, in_dim, height, width)
		# IMPORTANT: We would like to set these to Tensorflow order: (height, width, in_dim, out_dim)
		kernel_weights = np.reshape(kernel_weights,(shape[3],shape[2],shape[0],shape[1]),order='C')
		kernel_weights = np.transpose(kernel_weights,[2,3,1,0])

		return biases,kernel_weights,offset

	def preproces_image(self, input_img_path):
		input_image = cv2.imread(input_img_path)
		resized_image = cv2.resize(input_image,(416, 416), interpolation = cv2.INTER_CUBIC)
		image_data = np.array(resized_image, dtype='f')
		image_data /= 255.
		image_array = np.expand_dims(image_data, 0)

		return image_array


	def iou(self, boxA, boxB):
		# boxA = boxB = [x1,y1,x2,y2]

		# Determine the coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
 
		# Compute the area of intersection
		intersection_area = (xB - xA + 1) * (yB - yA + 1)
 
		# Compute the area of both rectangles
		boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
		# Compute the IOU
		iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

		return iou



	def non_maximal_suppression(self, thresholded_predictions, iou_threshold):

		nms_predictions = []

		# Add the best B-Box because it will never be deleted
		nms_predictions.append(thresholded_predictions[0])

		# For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
		# thresholded_predictions[i][0] = [x1,y1,x2,y2]
		i = 1
		while i < len(thresholded_predictions):
			n_boxes_to_check = len(nms_predictions)
			to_delete = False

			j = 0
			while j < n_boxes_to_check:
				curr_iou = self.iou(thresholded_predictions[i][0],nms_predictions[j][0])
				if(curr_iou > iou_threshold ):
					to_delete = True

				j = j+1

			if to_delete == False:
				nms_predictions.append(thresholded_predictions[i])
			i = i+1

		return nms_predictions

	def postprocess(self, predictions, input_img_path, score_threshold, iou_threshold, input_height=416, input_width=416):

		input_image = cv2.imread(input_img_path)
		input_image = cv2.resize(input_image,(input_height, input_width), interpolation = cv2.INTER_CUBIC)

		anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
		thresholded_predictions = []
		predictions = np.reshape(predictions, (13,13,5,25))

		# IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
		for row in range(13):
			for col in range(13):
				for b in range(5):
					tx, ty, tw, th, tc = predictions[row, col, b, :5]
					center_x = (float(col) + self.sigmoid(tx)) * 32.0
					center_y = (float(row) + self.sigmoid(ty)) * 32.0

					roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
					roi_h = np.exp(th) * anchors[2*b + 1] * 32.0

					final_confidence = self.sigmoid(tc)

					# Find best class
					class_predictions = predictions[row, col, b, 5:]
					class_predictions = self.softmax(class_predictions)
					class_predictions = tuple(class_predictions)
					best_class = class_predictions.index(max(class_predictions))
					best_class_score = class_predictions[best_class]

					# Compute the final coordinates on both axes
					left   = int(center_x - (roi_w/2.))
					right  = int(center_x + (roi_w/2.))
					top    = int(center_y - (roi_h/2.))
					bottom = int(center_y + (roi_h/2.))
		
					if( (final_confidence * best_class_score) > score_threshold):
						thresholded_predictions.append([ [left,top,right,bottom], final_confidence * best_class_score, self.classes[best_class] ])

		thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)
		nms_predictions = self.non_maximal_suppression(thresholded_predictions, iou_threshold)

		# Draw boxes with texts
		for i in range(len(nms_predictions)):
			color = self.colors[nms_predictions[i][2]]
			best_class_name = nms_predictions[i][2]
			textWidth = cv2.getTextSize(best_class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0] + 10
			
			input_image = cv2.rectangle(input_image, (nms_predictions[i][0][0], nms_predictions[i][0][1]), (nms_predictions[i][0][2],nms_predictions[i][0][3]), color)
			input_image = cv2.rectangle(input_image, (nms_predictions[i][0][0], nms_predictions[i][0][1]-20), (nms_predictions[i][0][0]+textWidth, nms_predictions[i][0][1]), color, -1)
			input_image = cv2.putText(input_image, best_class_name, (nms_predictions[i][0][0]+5, nms_predictions[i][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 4)
	  
		return input_image

	def __init__(self, Debugging=False):
		self.bn_epsilon = 1e-3

		if Debugging:
			offset = 0
			loaded_weights = []
	
			loaded_weights = np.fromfile("./weights/yolov2-tiny-voc.weights", dtype='f')
			# Delete the first 4 that are not real params...
			loaded_weights = loaded_weights[4:]

			# Conv1 , 3x3, 3->16
			self.biases1, self.weights1, offset = self.load_conv_layer_bn('conv1', loaded_weights, [3,3,3,16], offset)
			self.biases1 = tf.Variable(self.biases1, dtype=tf.float32)
			self.weights1 = tf.Variable(self.weights1, dtype=tf.float32)

			# Conv2 , 3x3, 16->32
			self.biases2, self.weights2, offset = self.load_conv_layer_bn('conv2', loaded_weights, [3,3,16,32], offset)
			self.biases2 = tf.Variable(self.biases2, dtype=tf.float32)
			self.weights2 = tf.Variable(self.weights2, dtype=tf.float32)

			# Conv3 , 3x3, 32->64
			self.biases3, self.weights3, offset = self.load_conv_layer_bn('conv3', loaded_weights, [3,3,32,64], offset)
			self.biases3 = tf.Variable(self.biases3, dtype=tf.float32)
			self.weights3 = tf.Variable(self.weights3, dtype=tf.float32)

			# Conv4 , 3x3, 64->128
			self.biases4, self.weights4, offset = self.load_conv_layer_bn('conv4', loaded_weights, [3,3,64,128], offset)
			self.biases4 = tf.Variable(self.biases4, dtype=tf.float32)
			self.weights4 = tf.Variable(self.weights4, dtype=tf.float32)

			# Conv5 , 3x3, 128->256
			self.biases5, self.weights5, offset = self.load_conv_layer_bn('conv5', loaded_weights, [3,3,128,256], offset)
			self.biases5 = tf.Variable(self.biases5, dtype=tf.float32)
			self.weights5 = tf.Variable(self.weights5, dtype=tf.float32)

			# Conv6 , 3x3, 256->512
			self.biases6, self.weights6, offset = self.load_conv_layer_bn('conv6', loaded_weights, [3,3,256,512], offset)
			self.biases6 = tf.Variable(self.biases6, dtype=tf.float32)
			self.weights6 = tf.Variable(self.weights6, dtype=tf.float32)

			# Conv7 , 3x3, 512->1024
			self.biases7, self.weights7, offset = self.load_conv_layer_bn('conv7', loaded_weights, [3,3,512,1024], offset)
			self.biases7 = tf.Variable(self.biases7, dtype=tf.float32)
			self.weights7 = tf.Variable(self.weights7, dtype=tf.float32)

			# Conv8 , 3x3, 1024->1024
			self.biases8, self.weights8, offset = self.load_conv_layer_bn('conv8', loaded_weights, [3,3,1024,1024], offset)
			self.biases8 = tf.Variable(self.biases8, dtype=tf.float32)
			self.weights8 = tf.Variable(self.weights8, dtype=tf.float32)

			# Conv9 , 1x1, 1024->125
			self.biases9, self.weights9, offset = self.load_conv_layer('conv9', loaded_weights, [1,1,1024,125], offset)
			self.biases9 = tf.Variable(self.biases9, dtype=tf.float32)
			self.weights9 = tf.Variable(self.weights9, dtype=tf.float32)

			print("Biases and weights are loaded!!")

	def predict(self):
		self.timer = time.time()
		#1 conv1     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16
		conv1 = tf.add(tf.nn.conv2d(self.image, self.weights1, strides=[1, 1, 1, 1], padding='SAME'), self.biases1)
		conv1 = self.leaky_relu( conv1 )

		#2 max1          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
		max1 = self.max_pool_layer( conv1 )

		#3 conv2     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32
		conv2 = tf.add(tf.nn.conv2d(max1, self.weights2, strides=[1, 1, 1, 1], padding='SAME'), self.biases2)
		conv2 = self.leaky_relu( conv2 )

		#4 max2          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
		max2 = self.max_pool_layer( conv2 )

		#5 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64
		conv3 = tf.add(tf.nn.conv2d(max2, self.weights3, strides=[1, 1, 1, 1], padding='SAME'), self.biases3)
		conv3 = self.leaky_relu( conv3 )

		#6 max3          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
		max3 = self.max_pool_layer( conv3 )

		#7 conv4    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128
		conv4 = tf.add(tf.nn.conv2d(max3, self.weights4, strides=[1, 1, 1, 1], padding='SAME'), self.biases4)
		conv4 = self.leaky_relu( conv4 )

		#8 max4          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
		max4 = self.max_pool_layer( conv4 )

		#9 conv5    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256
		conv5 = tf.add(tf.nn.conv2d(max4, self.weights5, strides=[1, 1, 1, 1], padding='SAME'), self.biases5)
		conv5 = self.leaky_relu( conv5 )

		#10 max5          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
		max5 = self.max_pool_layer( conv5 )

		#11 conv6   512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 512
		conv6 = tf.add(tf.nn.conv2d(max5, self.weights6, strides=[1, 1, 1, 1], padding='SAME'), self.biases6)
		conv6 = self.leaky_relu( conv6 )

		#12 max6          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
		max6 = self.max_pool_layer( conv6, stride=1, padding='SAME' )

		#13 conv7    1024  1 x 1 / 1    13 x  13 x512   ->    13 x  13 x 1024
		conv7 = tf.add(tf.nn.conv2d(max6, self.weights7, strides=[1, 1, 1, 1], padding='SAME'), self.biases7)
		conv7 = self.leaky_relu( conv7 )

		#14 conv8   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
		conv8 = tf.add(tf.nn.conv2d(conv7, self.weights8, strides=[1, 1, 1, 1], padding='SAME'), self.biases8)
		conv8 = self.leaky_relu( conv8 )

		#15 conv9   125  1 x 1 / 1    13 x  13 x 1024   ->    13 x  13 x125
		conv9 = tf.add(tf.nn.conv2d(conv8, self.weights9, strides=[1, 1, 1, 1], padding='SAME'), self.biases9, name="Output")

		return conv9




