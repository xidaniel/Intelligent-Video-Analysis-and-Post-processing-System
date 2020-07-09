import cv2


class Tracker:
    def __init__(self,):
        self.tracker = cv2.MultiTracker_create()
        self.init_once = False
        # tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = cv2.TrackerMOSSE_create()

    def track(self, frame, bbox):
        if not self.init_once:
            ret = self.tracker.add(cv2.TrackerMOSSE_create(), frame, bbox)
            self.init_once = True

        ret, boxes = self.tracker.update(frame)
        for newbox in boxes:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

        bbox_person = list(map(int, boxes[0]))
        return bbox_person
