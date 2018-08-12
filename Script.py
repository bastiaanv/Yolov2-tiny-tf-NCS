import tensorflow as tf
import NeuralNetwork
import os
import cv2
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cv2_image = cv2.imread("./images/Traffic.jpg")

NN = NeuralNetwork.Net(Debugging=True)
image = NN.image
picture = NN.preproces_image(cv2_image)



with tf.Session() as sess:
	start = time.time()

	sess.run(tf.global_variables_initializer())
	prediction = sess.run(NN.predict(), feed_dict={image:picture})
	print("Time took to compute: " + str((time.time()-start)*1000) + "ms")
	
	#Save Network
	saver = tf.train.Saver()
	saver.save(sess, "./model/NN.ckpt")
	tf.train.write_graph( sess.graph_def, "./model/", "NN.pb", as_text=False )	

	output_image, boxes = NN.postprocess(prediction, cv2_image, 0.3, 0.3)
	cv2.imshow("Result", output_image)
	cv2.imwrite("./images/test.jpg", output_image)
	cv2.waitKey(0)

	print("Done!!")
