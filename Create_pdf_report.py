import cv2
import numpy as np
import os
import fnmatch
import time
import unittest
import cython
from glob import iglob
import csv
from shapely.geometry import Polygon
from ast import literal_eval
import matplotlib.pyplot as plt
import cv2
from CustomisedPlot import CustomisedPlot

from fpdf import FPDF
from pdf2image import convert_from_path
from Evaluation import best_mF1, best_mAP, best_mAR, median_mF1, median_mAP, median_mAR, metrics_path , total_false_positives , total_false_negatives, id_best_image,total_number_of_images,total_no_ground_true



home_dir = os.getcwd()




class myPDF(FPDF):
    def __init__(self, model_version: str, operator_name: str, validation_set: str, format='A4', unit='in'):
        """

        @rtype: object
        """
        super(myPDF, self).__init__()
        self.WIDTH = 210
        self.HEIGHT = 297


        logo_path = os.path.join(home_dir ,  'logo', 'dcm-logo-transparent.png')
        self.logo = logo_path






        self.model_version = model_version
        self.operator_name =operator_name
        self.validation_set = validation_set
        self.pdf_output_path = os.path.join(home_dir, 'report-' + str(self.model_version) + '.pdf')


    def read_images(self, metrics_path):

        self.output_best_image  = os.path.join(os.path.dirname(metrics_path) + '/', id_best_image + '_out'+ '.jpg')

        for filename in os.listdir(metrics_path):

            if 'Accuracy' in filename:
                self.accuracy_precision = os.path.join(metrics_path + '/' + filename)
                print(self.accuracy_precision)
            elif 'F1' in filename:
                self.F1 = os.path.join(metrics_path + '/' + filename)
                print(self.F1)

        return self.output_best_image, self.accuracy_precision, self.F1



    def header(self):
        #header
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(128)
        self.set_fill_color(173, 216, 230)
        self.cell(self.WIDTH - 50)
        self.cell(0, 5, txt = 'DCM_Internal', border=0, align = 'R', fill=  1)
        self.image(self.logo,x =5, y =0 , w=20, h=20)
        self.ln(7)


    def footer(self):
        #footer

        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def page_body(self):
        # title

        self.set_font(family='Helvetica', style='b', size=30)
        self.set_text_color(128)
        self.set_fill_color(173, 216, 230)

        #self.multi_cell(x=40, y=40, txt='Model_Performance_Report')
        #self.text(20, 40, 'Model_Performance_Report' )
        self.cell(0, 10, txt = 'Model_Performance_Report', border=0, align = 'C', fill=  1)
        self.ln(10)





        #CEll-1


        pdf.set_font(family='Helvetica', style='b', size=16)
        pdf.multi_cell(200, 10, txt='Model_version: {}'.format(self.model_version))
        pdf.multi_cell(200, 10, txt='Operator: {}'.format(self.operator_name))
        pdf.multi_cell(200, 10, txt='Validation_set_ID: {}'.format(self.validation_set))

        self.ln(5)


        pdf.multi_cell(200, 7, txt='Best_Performing_Image',  align = 'L')
        pdf.image(self.output_best_image,  w=100, h=100)
        self.ln(3)

        pdf.multi_cell(200, 10, txt='Model Metrics')
        pdf.multi_cell(200, 10, txt='Total_images: {}, Total_no_ground_truth: {}'.format(total_number_of_images, total_no_ground_true))
        pdf.multi_cell(350, 10,border = 0, txt='best_mF1: {}, best_mAP: {}, best_mAR: {}'.format(np.round(best_mF1,3), np.round(best_mAP, 4), np.round(best_mAR, 4)))
        pdf.multi_cell(350, 10,border = 0, txt='median_mF1: {}, median_mAP: {}, median_mAR: {}'.format(np.round(median_mF1,3), np.round(median_mAP, 4), np.round(median_mAR, )))
        pdf.multi_cell(350, 10,border = 0, txt='total_false_positives: {}, total_false_negatives: {}'.format(total_false_positives, total_false_negatives))

        self.ln(5)

        pdf.set_auto_page_break(auto=1)
        pdf.add_page()
        pdf.multi_cell(200, 7, txt='Precision_vs_Recall',  align = 'C')
        pdf.image(self.accuracy_precision,  w=200, h=200)
        self.ln(2)

        pdf.set_auto_page_break(auto=1)
        pdf.add_page()
        pdf.multi_cell(200, 7, txt='F1_score_Distribution', align='C')
        pdf.image(self.F1, w=200, h=200)
        self.ln(2)






        pdf.output('DCM_Model_Validation_.pdf', 'F')





if __name__ == '__main__':



    pdf = myPDF('DCM_pedestrian_detector', 'MariaK', 'ScotRail_test_set')
    pdf.read_images(metrics_path)
    pdf.add_page()
    pdf.page_body()




