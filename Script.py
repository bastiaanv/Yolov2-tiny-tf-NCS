import tensorflow as tf
import NeuralNetwork
import os
import cv2
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_name = "./images/Traffic.jpg"

NN = NeuralNetwork.Net(Debugging=True)
image = NN.image
picture = NN.preproces_image(image_name)



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	prediction = sess.run(NN.predict(), feed_dict={image:picture})
	print("Time took to compute: " + str((time.time()-NN.timer)*1000) + "ms")
	
	#Save Network
	saver = tf.train.Saver()
	saver.save(sess, "./model/NN.ckpt")
	tf.train.write_graph( sess.graph_def, "./model/", "NN.pb", as_text=False )	

	output_image = NN.postprocess(prediction, image_name, 0.3, 0.3)
	cv2.imwrite("test.jpg", output_image)
	cv2.imshow('Result', output_image)
	cv2.waitKey(0)
	print("Done!!")
