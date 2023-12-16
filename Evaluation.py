import cv2
import numpy as np
import os
import time
import unittest
import cython
from glob import iglob
import csv
from shapely.geometry import Polygon
from ast import literal_eval
import matplotlib.pyplot as plt



import sys



from CustomisedPlot import CustomisedPlot

sample_dir = 'full_path'


def mkdir(path):
    if (os.path.exists(path)):
        pass
    else:
        os.makedirs(path)


output_dir = os.path.join(sample_dir, 'output')
path_plot = os.path.join(output_dir, 'metrics')
mkdir(output_dir)
mkdir(path_plot)

CP = CustomisedPlot()

def evaluate_batch( path_root, conf_thres, iou_thres):

	# locate val dataset
	path_images_val = os.path.join(path_root)
	path_labels_val = os.path.join(path_root)
	# dest
	path_images_val_tmp = os.path.join(path_root)
	path_labels_val_tmp = os.path.join(path_root)



	model_version = 'DCM_Seated'
	label = '0'
	iou_thres = iou_thres
	conf_thres =conf_thres
	range_iou_thres = [iou_thres]
	range_conf_thres = [conf_thres]
	#range_conf_thres = [conf_thres]

	# plot
	title = 'DCM_Seated'
	x_label = 'recall'
	y_label = 'precision'
	x_lim = [0, 1.05]
	y_lim = [0, 1.05]
	fig, ax = CP.init_plot_scatter_2d( x_label, y_label, title, x_lim, y_lim)

	list_mF1_all = []
	list_iou_thres_all = []
	list_conf_thres_all = []

	list_mAP_all = []
	list_mAR_all = []

	list_mAP = []
	list_mAR = []

	total_true_positives = 0
	total_false_positives = 0
	total_false_negatives = 0
	total_number_of_images = 0
	total_no_ground_true =0



	for conf_thres in range_conf_thres:
		#print(conf_thres)

		inline_annotation_list = []

		# counter_tmp = 0
		for iou_thres in range_iou_thres:
			# generate save path
			path_images_tmp = os.path.join(path_images_val_tmp)
			path_labels_tmp = os.path.join(path_labels_val_tmp)

			list_num_relavant_item = []
			list_num_ground_true = []
			list_num_prediction = []
			list_precision_per_image = []
			list_recall_per_image = []
			list_F1_score_per_image = []

			jpgs_ids = []




			for path_image in sorted(iglob(os.path.join(path_images_val, '*.jpg'))) :


				image_raw = cv2.imread(path_image)
				basename_image = os.path.basename(path_image)
				id_image = os.path.splitext(basename_image)[0]


				jpgs_ids.append(id_image)
				total_number_of_images  += 1

				# original label
				path_label = os.path.join(path_labels_val, id_image + '.txt')
				# path_image

				# inferenced label
				path_label_tmp = os.path.join(path_labels_tmp, 'output', id_image + '.txt')



				# evaluated image
				path_image_tmp = os.path.join(path_images_tmp, 'output', id_image + '_out.jpg')

				file = open(path_label, "r")
				img = cv2.imread(path_image)
				Y = img.shape[0]
				X = img.shape[1]

				list_coor_eval = [] #list with all the person_detections from the model
				file_eval = open(path_label_tmp, "r")

				for row_eval in file_eval:  # reads the .txt from the model
					item_eval = row_eval.split()
					if item_eval[5] == label:
						# print(item_eval)
						x1_eval = int(float(item_eval[0]))
						y1_eval = int(float(item_eval[1]))
						x2_eval = int(float(item_eval[2]))
						y2_eval = int(float(item_eval[3]))
						coor_eval = [x1_eval, y1_eval, x2_eval, y2_eval]
						list_coor_eval.append(coor_eval)
				file_eval.close()


				list_coor = [] #list of ground truth persons
				for row in file:
					item = row.split()
					# <x> <y> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
					# for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
					# print(float(item[2]))
					if item[0] == label:
						x1 = int(float(item[1]) * X) - int(float(item[3]) * X / 2)
						y1 = int(float(item[2]) * Y) - int(float(item[4]) * Y / 2)
						x2 = int(float(item[1]) * X) + int(float(item[3]) * X / 2)
						y2 = int(float(item[2]) * Y) + int(float(item[4]) * Y / 2)

						# img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
						# cv2.imshow('image', img)
						# cv2.waitKey()

						coor = [x1, y1, x2, y2]
						list_coor.append(coor)
					else:
						print('problem')
				file.close()


				num_ground_true = len(list_coor)
				total_no_ground_true += num_ground_true



				num_prediction = len(list_coor_eval)







				num_relavant_item = 0
				false_positives =0

				false_negatives =0
				# for each eval coor, find the largest iou
				list_idx_best = []

				for coor in list_coor:
					iou_prev = 0

					temp_iou = []
					counter0 =0
					counter =0
					idx_best = -1

					for idx, coor_eval in enumerate(list_coor_eval): #read all model detections one by one and calculate iou for every groud grouth
						# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
						xA = max(coor[0], coor_eval[0])
						yA = max(coor[1], coor_eval[1])
						xB = min(coor[2], coor_eval[2])
						yB = min(coor[3], coor_eval[3])
						# compute the area of intersection rectangle
						interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
						# compute the area of both the prediction and ground-truth
						# rectangles
						boxAArea = (coor[2] - coor[0] + 1) * (coor[3] - coor[1] + 1)
						boxBArea = (coor_eval[2] - coor_eval[0] + 1) * (coor_eval[3] - coor_eval[1] + 1)
						# compute the intersection over union by taking the intersection
						# area and dividing it by the sum of prediction + ground-truth
						# areas - the interesection area
						iou = interArea / float(boxAArea + boxBArea - interArea)
						iou_prev = max(iou, iou_prev)
						temp_iou.append(iou)
						counter0 += 1
						if iou_prev == iou:
							coor_eval_best = coor_eval
							idx_best = idx
							counter += 1
					iou_final = max(temp_iou)
					# visualise
					img = cv2.rectangle(img, (coor[0], coor[1]), (coor[2], coor[3]), (0, 255, 0), 1)
					cv2.putText(img, 'iou ' + str(iou_prev)[0:4], (coor[0], coor[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
								(0, 255, 0), 1, cv2.LINE_AA)




					if iou_final > iou_thres or iou_final == iou_thres:

							list_idx_best.append(idx_best)
							num_relavant_item += 1
							#prediction matches the real annotation
							total_true_positives +=1
							img = cv2.rectangle(img, (coor_eval_best[0], coor_eval_best[1]),
												(coor_eval_best[2], coor_eval_best[3]), (0, 0, 255), 1)
						# print('passed iou thres', 'num_relavant_item', num_relavant_item, 'idx_best', idx_best, 'iou_prev', iou_prev)

					else:

						pass
						# print('failed iou thres', 'idx_best', idx_best, 'iou_prev', iou_prev)

				assert num_relavant_item <= num_prediction



				# calculate precision
				# print('num_relavant_item', num_relavant_item)
				if num_relavant_item  < num_ground_true:
					false_negatives += num_ground_true - num_relavant_item
					total_false_negatives += num_ground_true - num_relavant_item

				elif num_relavant_item > num_ground_true:
					false_positives += num_prediction - num_relavant_item
					total_false_positives += num_prediction - num_relavant_item


				try:
					precision_per_image = num_relavant_item / num_prediction
				except ZeroDivisionError:
					precision_per_image = 0

				# calculate recall
				try:
					recall_per_image = num_relavant_item / num_ground_true
				except ZeroDivisionError:
					recall_per_image = 0
				# F1 score
				try:
					F1_score = 2 * (precision_per_image * recall_per_image) / (
							precision_per_image + recall_per_image)
				except ZeroDivisionError:
					F1_score = 0

				#print('id', id_image, 'precision', precision_per_image, 'recall', recall_per_image, 'F1', F1_score,
					  #'num_prediction', num_prediction, 'num_ground_true', num_ground_true)

				# store image in output folder
				cv2.rectangle(img, (0, 0), (310, 200), (0, 0, 0), -1)
				cv2.putText(img, 'precision of this image ' + str(precision_per_image)[0:4], (20, 20),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(img, 'recall of this image ' + str(recall_per_image)[0:4], (20, 40),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(img, 'F1_score of this image ' + str(F1_score)[0:4], (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
							0.6, (255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(img, 'green box: dcm label ', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
							(255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(img, 'red box: predictions ', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
							1, cv2.LINE_AA)
				cv2.putText(img, 'model version: ' + model_version, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
							(255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(img, 'no of detections==person ' + str(num_relavant_item), (20, 140),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(img, 'total no of detections ' + str(num_prediction), (20, 160),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(img, 'real no of pedestrians ' + str(num_ground_true),
							(20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
				cv2.imwrite(path_image_tmp, img)



				# cuumulate detection history
				list_num_relavant_item.append(num_relavant_item)
				list_num_ground_true.append(num_ground_true)
				list_num_prediction.append(num_prediction)
				list_precision_per_image.append(precision_per_image)
				list_recall_per_image.append(precision_per_image)
				list_F1_score_per_image.append(F1_score)

			# write
			# print('conf_thres %f iou_thres %f path_label %s' % (conf_thres, iou_thres, path_label))

			# mean precision
				mAP = sum(list_num_relavant_item) / sum(list_num_prediction)
			# print('conf_thres', conf_thres, 'mAP@', iou_thres, mAP)
			# mean recall
				mAR = sum(list_num_relavant_item) / sum(list_num_ground_true)
			# print('conf_thres', conf_thres, 'mAR@', iou_thres, mAR)
				mF1_score = 2 * (mAP * mAR) / (mAP + mAR)
				#print('mF1score', mF1_score)
			# print('conf_thres', conf_thres, 'F1@', iou_thres, mF1_score)

				list_mAP.append(mAP)
				list_mAR.append(mAR)

				list_mAP_all.append(mAP)
				list_mAR_all.append(mAR)

				list_mF1_all.append(mF1_score)
				list_iou_thres_all.append(iou_thres)
				list_conf_thres_all.append(conf_thres)
			# inline_annotation_list.append('c' + str(conf_thres) +'i' + str(iou_thres) + 'f' + str(mF1_score)[0:5])
				inline_annotation_list.append('i' + str(iou_thres) + 'f' + str(mF1_score)[0:5])



	best_mF1 = max(list_mF1_all)
	best_mAP = max(list_mAP_all)
	best_mAR = max(list_mAR_all)
	# print('list_mF1', list_mF1_all)
	# get index of the best mf1 to find the iou and thres
	best_idx = list_mAP_all.index(max(list_mAP_all))
	id_best_image = jpgs_ids[best_idx]
	print('best_idx', best_idx)
	# find the best iou and conf thres
	#best_iou_thres = list_iou_thres_all[best_idx]
	#best_conf_thres = list_conf_thres_all[best_idx]

	#print('total_false_positives', total_false_positives)
	#print('total_false_negatives', total_false_negatives)

	median_mF1 = np.median(list_mF1_all)
	median_mAP = np.median(list_mAP_all)
	median_mAR = np.median(list_mAR_all)
	# print('list_mF1', list_mF1_all)
	# get index of the best mf1 to find the iou and thres


	print('best_mF1', best_mF1, 'best_mAP', best_mAP, 'best_mAR', best_mAR)
	print('median_mF1', median_mF1, 'best_mAP', median_mAP, 'best_mAR', median_mAR)

	# print('list_mAR', list_mAR, 'list_mAP', list_mAP)

	fig0, ax0 =  CP.plot_scatter_2d_without_save(fig, ax, list_mAR_all, list_mAP_all, best_mAP, best_mAR,median_mAP, median_mAR)

	fig1, ax1 = CP.plot_f1_score(list_mF1_all)

	# save figure with best values
	CP.save_fig(fig0, ax0, path_plot, title + '_Precision_vs_Accuracy')

	CP.save_fig(fig1, ax1, path_plot, title + '_F1_score')

	return best_mF1, best_mAP, best_mAR, median_mF1, median_mAP, median_mAR, path_plot, total_false_positives,total_false_negatives, id_best_image, total_number_of_images, total_no_ground_true


best_mF1, best_mAP, best_mAR, median_mF1, median_mAP, median_mAR, metrics_path , total_false_positives , total_false_negatives, id_best_image,total_number_of_images,total_no_ground_true= evaluate_batch(sample_dir, 0.2, 0.7)
