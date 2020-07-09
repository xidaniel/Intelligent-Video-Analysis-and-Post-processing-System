import cv2


class VideoDecoder:
    def __init__(self, input_path):
        self.input_path = input_path
        self.capture = cv2.VideoCapture(self.input_path)
        self.total_frame = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = int(round(self.capture.get(cv2.CAP_PROP_FPS), 0))
        print("The video has {} frames, FPS is {}.".format(self.total_frame, self.FPS))

    def decode_frame(self):
        ret, frame = self.capture.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_total_frames(self):
        return self.total_frame

    def get_FPS(self):
        return self.FPS

    def close(self):
        self.capture.release()


class VideoEncoder:
    def __init__(self, output_path, background, FPS):
        self.output_path = output_path
        self.background = background
        self.FPS = FPS
        self.videoWrite = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.FPS, (self.background.shape[1], self.background.shape[0]))

    def write(self):
        self.videoWrite.write(self.background)

    def write_real_time(self, image):
        self.videoWrite.write(image)

    def close(self):
        self.videoWrite.release()
