#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' layout module '

__author__ = 'Xi Wang'

import cv2
import numpy as np

class NewLayout:

    def __init__(self, warped, margin, person, pp):
        """
        :param warped: a numpy array warped.shape=[h,w,c]
        :param margin: the video margin
        :param person: include person or not, keyword para "Yes" or "No"
        :param pp: person position, "left" or "right" or "none"
        """
        self.margin = margin
        self.warped = warped
        self.person = person
        self.pp = pp
        self.screenwidth = self.warped.shape[1]
        self.screenheight = self.warped.shape[0]

    def setbg(self):
        """
        Confirm the new video background size
        :return: background, screenroi, personroi
        """
        scale = round(self.screenheight / self.screenwidth, 2)
        if self.person == "Yes":
            if scale > 0.65:
                bg = np.zeros((self.screenheight + self.margin * 2, int(self.screenwidth * 4 / 3) + self.margin * 3, 3),
                              np.uint8)
                print('The new video aspect ratio: 4:3')
                screenroi = (self.screenwidth, self.screenheight)
                personroi = (int(self.screenwidth * 1 / 3), int(self.screenwidth * 1 / 3))
            else:
                bg = np.zeros((self.screenheight + self.margin * 2, int(self.screenwidth * 5 / 4) + self.margin * 3, 3),
                              np.uint8)
                print('The new video aspect ratio: 16:9')
                screenroi = (self.screenwidth, self.screenheight)
                personroi = (int(self.screenwidth * 1 / 4), int(self.screenwidth * 1 / 4))
        if self.person == "No":
            if scale > 0.65:
                bg = np.zeros((self.screenheight + self.margin * 2, self.screenwidth + self.margin * 2, 3), np.uint8)
                print('The new video aspect ratio: 4:3')
                screenroi = (self.screenwidth, self.screenheight)
                personroi = None
            else:
                bg = np.zeros((self.screenheight + self.margin * 2, self.screenwidth + self.margin * 2, 3), np.uint8)
                print('The new video aspect ratio: 16:9')
                screenroi = (self.screenwidth, self.screenheight)
                personroi = None

        print('The new video resolution: {}'.format(bg.shape[:2]))
        return bg, screenroi, personroi

    def format(self, frame, bg, person_bbox, warped, screenroi, personroi):
        """
        :param frame: video frame
        :param bg: the background size
        :param person_bbox: the person coordinate person_bbox[x,y,w,h] w: the width of bbox  h: the height of bbox
        :param warped: the correct screen
        :param screenroi: screen roi size in bg (h,w)
        :param personroi: person roi size in bg (h,w)
        :return: bg
        """
        warped_resize = cv2.resize(warped, screenroi)  # input [w,h]  output [h,w]
        p_bbox_x, p_bbox_y, p_bbox_w, p_bbox_h = person_bbox[0], person_bbox[1], person_bbox[2], person_bbox[3]
        person_crop = frame[p_bbox_y:p_bbox_y + p_bbox_w, p_bbox_x:p_bbox_x + p_bbox_w]

        if self.pp == "right":
            person_resize = cv2.resize(person_crop, personroi)  # input [w,h]  output [h,w]
            # set screen position
            bg[self.margin:self.margin + self.screenheight, self.margin:self.margin + self.screenwidth] = warped_resize
            # set person position
            bg[self.margin + int(0.5 * (self.screenheight - person_resize.shape[0])):self.margin + int(
                0.5 * (self.screenheight + person_resize.shape[0])),
            2 * self.margin + self.screenwidth:2 * self.margin + self.screenwidth + person_resize.shape[
                1]] = person_resize

        if self.pp == "left":
            person_resize = cv2.resize(person_crop, personroi)  # input [w,h]  output [h,w]
            # set screen position
            bg[self.margin:self.margin + self.screenheight,
            2 * self.margin + person_resize.shape[1]:2 * self.margin + person_resize.shape[
                1] + self.screenwidth] = warped_resize
            # set person position
            bg[self.margin + int(0.5 * (self.screenheight - person_resize.shape[0])):self.margin + int(
                0.5 * (self.screenheight + person_resize.shape[0])),
            self.margin:self.margin + person_resize.shape[1]] = person_resize

        if self.pp == "none":
            bg[self.margin:self.margin + self.screenheight, self.margin:self.margin + self.screenwidth] = warped_resize

        return bg





