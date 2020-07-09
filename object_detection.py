import os
import mrcnn.model as modellib
from mrcnn import visualize
import lecture


class Detection:
    def __init__(self):
        # Root directory of the project
        ROOT_DIR = os.getcwd()

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        LECTURE_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_video.h5")

        class InferenceConfig(lecture.LectureConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        #config.display()
        print("Loading detection model......")
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        # Load weights trained on MS-COCO
        self.model.load_weights(LECTURE_MODEL_PATH, by_name=True)
        self.class_names = ['BG', 'person', 'screen','face']

    def detect(self, image):
        results = self.model.detect([image], verbose=0)
        #print("Detection complete")
        r = results[0]
        image = visualize.show_image(image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])
        return image
    
    def detect_v1(self, image):
        results = self.model.detect([image], verbose=0)
        #print("Detection complete")
        r = results[0]
        return r

    def get_bboxs(self, image):
        results = self.model.detect([image], verbose=0)
        print("Detection complete")
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])
        dic = dict(zip(r['class_ids'], r['rois']))
        person, screen = dic[1], dic[2]

        return person, screen
