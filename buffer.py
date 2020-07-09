from skimage.measure import compare_ssim
import numpy as np
import random
import colorsys
import cv2

class Buffer:
    def __init__(self, N=50):
        self.color_sets = []
        self.color_index = 0
        self.random_colors_rgb(N)
        self.instructor_buffer = {}
        self.screen_buffer = {}
        self.list_inst_bbox = []
        self.list_screen_bbox = []
        
        
    def compute_iou(self, boxA, boxB):
        """
        input: boxA numpy []
        """
        if not isinstance(boxA,(tuple)):
            boxA = boxA.tolist()
            boxB = boxB.tolist()

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def crop(self, image, bbox):
        y1, x1, y2, x2 = bbox
        image = image[y1:y2, x1:x2]
        return image


    def update_instructor(self, frame, rois, class_ids, theta=0.9):
        """
        input:
             frame: raw fram
             theta: hyperparameter
        """
        
        self.get_instructor_bbox(rois, class_ids)
        
        if not self.instructor_buffer:
            for j in range(len(self.list_inst_bbox)):
                self.instructor_buffer[str(j)] = [1, self.crop(frame, self.list_inst_bbox[j]), self.color_sets[self.color_index], self.list_inst_bbox[j]]
                self.color_index += 1
        else:
            for box in self.list_inst_bbox:
                iou = 0
                for key, item in self.instructor_buffer.items():
                    iou = self.compute_iou(item[3], box)
                    if iou != 0:
                        if iou < theta:
                            item[3] = box
                            item[0] += 1
                        else:
                            item[0] += 1
                        break
                if iou == 0:
                    self.instructor_buffer[str(len(self.instructor_buffer))] = [1, self.crop(frame, box), self.color_sets[self.color_index], box]
                    self.color_index += 1


    def update_screen(self, frame, rois, class_ids, beta=0.85):
        """
        input:
             frame: raw fram
             beta: hyperparameter
        """
        
        self.get_screen_bbox(rois, class_ids)
        
        if not self.screen_buffer:
            for j in range(len(self.list_screen_bbox)):
                self.screen_buffer["screen"+str(j)] = [1, self.crop(frame, self.list_screen_bbox[j]), self.color_sets[self.color_index], self.list_screen_bbox[j]]
                self.color_index += 1
        else:
            for box in self.list_screen_bbox:
                iou = 0
                for key, item in self.screen_buffer.items():
                    iou = self.compute_iou(item[3], box)
                    if iou != 0:
                        if iou < beta:
                            item[3] = box
                            item[0] += 1
                        else:
                            item[0] += 1
                if iou == 0:
                    self.screen_buffer["screen" + str(len(self.screen_buffer))] = [1, self.crop(frame, box), self.color_sets[self.color_index], box]
                    self.color_index += 1


    def get_instructor_bbox(self, boxes, class_ids):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        output:[x,y,w,h]
        """
        # clear cache before running
        self.list_inst_bbox = []
        
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instructor to display *** \n")

        for i in range(N):
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]      

            # find instructor
            if class_ids[i] == 1:
                for j in range(len(class_ids)):
                    if class_ids[j] == 3:
                        value = self.compute_iou(boxes[i], boxes[j])
                        if value != 0:
                            self.list_inst_bbox.append((y1, x1, y2, x2))


    def get_screen_bbox(self, boxes, class_ids):
        """
        output:[x,y,w,h]
        """
        
        # clear cache before running
        self.list_screen_bbox = []
        
        N = boxes.shape[0]
        if not N:
            print("\n*** No screen to display *** \n")

        for i in range(N):
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if class_ids[i] == 2:
                self.list_screen_bbox.append((y1, x1, y2, x2))


    def random_colors_rgb(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        self.color_sets = [tuple(round(j * 255) for j in colorsys.hsv_to_rgb(i / N, 1, brightness)) for i in range(N)]
        random.shuffle(self.color_sets)


    def compute_SSIM(self, pre_img, cur_img):
        """
        input:
            pre_img: gray image
            cur_img: gray image
        output: 
            score
        """
        pre_image = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
        cur_image = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(pre_image, cur_image, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))


    def top_screen(self, n=1):
        """
        input: {"ID",[time, value, color, bbox]}
        output: identity, color, bbox
        """
        if not self.screen_buffer:
            print("\n*** No instances to display *** \n")
            return None
        
        if n > len(self.list_screen_bbox):
            n = len(self.list_screen_bbox)
        
        result = []
        ranking = sorted(self.screen_buffer.items(), key=lambda x: -x[1][0])
        for i in range(n):
            if i < len(self.screen_buffer):
                result.append([ranking[i][0], ranking[i][1][2], ranking[i][1][3]])
            
        return result
    
    def top_instructor(self, n=1):
        """
        input: {"ID",[time, value, color, bbox]}
        output: identity, color, bbox
        """
        if not self.instructor_buffer:
            print("\n*** No instances to display *** \n")
            return None
        
        if n > len(self.list_inst_bbox):
            n = len(self.list_inst_bbox)
        
        result = []
        ranking = sorted(self.instructor_buffer.items(), key=lambda x: -x[1][0])
        for i in range(n):
            if i < len(self.instructor_buffer):
                result.append([ranking[i][0], ranking[i][1][2], ranking[i][1][3]])
            
        return result