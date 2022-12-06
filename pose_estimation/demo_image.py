# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Roman Tsyganok (iskullbreakeri@gmail.com)
# ------------------------------------------------------------------------------

import cv2
import argparse
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Show demo')
  
    parser.add_argument('--img',
                        help='picture filename',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()


    parser.add_argument('--model',
                        help='model state file',
                        required=True,
                        type=str)
    parser.add_argument('--type',
                        help='ONNX/OpenVINO',
                        required=False,
                        type=str)

    parser.add_argument('--xml',
                        help='OpenVINO XML config',
                        required=False,
                        type=str)
    parser.add_argument('--backend',
                        help='OpenCV DNN Backend',
                        required=False,
                        type=str)
    parser.add_argument('--width',
                        help='input network width',
                        required=True,
                        type=int)
    parser.add_argument('--height',
                        help='input network height',
                        required=True,
                        type=int)

    args = parser.parse_args()

    return args
	
	
	
	
	
	
	
	
	
def main():
	args = parse_args()
	network = None
	
	scale = 1.0/255
	
	#linking rule
	points_links = [ [0,1],[0,2],[1,3],[2,4],[0,5],[0,6],[6,8],[8,10],[5,7],[7,9],[5,11],[6,12],[11,13],[13,15],[12,14],[14,16],[12, 11], [6, 5] ]
	points_number = 17
	#body part points description
	body_parts = {
	0:'nose',
	1:'left_eye',
	2:'right_eye',
	3:'left_ear',
	4:'right_ear',
	5:'left_shoulder',
	6:'right_shoulder',
	7:'left_elbow',
	8:'right_elbow',
	9:'left_palm',
	10:'right_palm',
	11:'left_hip',
	12:'right_hip',
	13:'left_knee',
	14:'right_knee',
	15:'left_feet',
	16:'right_feet'
	
	}
	#minimal probability
	threshold = 0.1
	
	testpath, filename = os.path.split(args.img)
	
	original = cv2.imread(args.img)
	image = np.copy(original)

	image_width = image.shape[1]
	image_height = image.shape[0]

	if args.type == 'ONNX':
		network = cv2.dnn.readNetFromONNX(args.model)
		
	elif args.type == 'OpenVINO':
		network = cv2.dnn.readNetFromModelOptimizer(args.xml, args.model)
	
	# default backend if wasn`t specified
	if not args.backend:
		network.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
	
	#in case you are going to use CUDA backend in OpenCV, make sure that opencv built with CUDA support
	elif args.backend == 'CUDA':
		network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	
	#in case you are going to use OpenVINO model, make sure that inference engine already installed and opencv built with IE support
	elif args.backend == 'INFERENCE':
		network.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
		network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	

	input = cv2.dnn.blobFromImage(image, scale, (args.width, args.height), (0, 0, 0), False);
	
	network.setInput(input)

	output = network.forward()

	H = output.shape[2]
	W = output.shape[3]
	
	points = []
	
	for i in range(points_number):
		# confidence map of corresponding body's part.
		probMap = output[0, i, :, :]

		# Find global maxima of the probMap.
		minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
		# Scale the point to fit on the original image
		x = (image_width * point[0]) / W
		y = (image_height * point[1]) / H

		if prob > threshold : 
			cv2.circle(image, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
			cv2.putText(image, "{}".format(body_parts[i]), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1, lineType=cv2.LINE_AA)

			# Add the point to the list if the probability is greater than the threshold
			points.append((int(x), int(y)))
		else :
			points.append(None)

	
	
	
	
	
	# Draw Skeleton
	for pair in points_links:
		partA = pair[0]
		partB = pair[1]
		
		try:
			if points[partA] and points[partB]:
				cv2.line(image, points[partA], points[partB], (0, 255, 144), 2)
		except:
			print("error")
	
	cv2.imshow('Original image', original)
	cv2.imshow('Skeleton', image)
	cv2.imwrite(testpath +'/result/'  + filename[:-4]+'_result' + '.jpg', image)
	cv2.waitKey(0)
	print(args)
	

	
if __name__ == '__main__':
    main()
