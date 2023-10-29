import os

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from groundingdino.util.inference import Model


class CardDetector:
    def __init__(
        self,
        prompt,
        model_config_path,
        model_weights_path,
        text_threshold=0.3,
        box_threshold=0.5,
        device="cuda",
    ):
        self.prompt = prompt
        self.model = Model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_weights_path,
            device=device,
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

    def detect_image_single(self, image):
        detections, _ = self.model.predict_with_caption(
            image,
            self.prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        return detections

    def detect_images(self, images, return_annotated_images=False):
        detections_batch = self.model.predict_with_caption_batch(
            images,
            self.prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        if return_annotated_images:
            annotated_images = self.annotate_images(images, detections_batch)
            return detections_batch, annotated_images
        else:
            return detections_batch

    # def detect_video(self, video_path, fps, frames_per_batch):
    #     video_reader = video_utils.VideoReader(video_path, extraction_fps=fps, frames_per_batch=frames_per_batch)
    #     for frame_batch in tqdm.tqdm(video_reader.read_2(), total=video_reader.n_batches):
    #         detections_batch = self.detect_images(frame_batch)
    #         yield detections_batch

    #     video_reader.close()

    @staticmethod
    def annotate_image(image, detections, extra_labels):
        box_annotator = sv.BoxAnnotator()
        # annotated_images = []
        labels = [
            f""
            for (_, _, confidence, class_id, _), extra_label in zip(
                detections, extra_labels
            )
        ]
        annotated_frame = box_annotator.annotate(
            scene=image.copy(), detections=detections, labels=labels
        )
        # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        # annotated_frame_pil = Image.fromarray(annotated_frame)
        # annotated_images.append(annotated_frame_pil)

        return annotated_frame

    def crop_detections(self, image, img_detection):
        if len(img_detection) == 0:
            return []
        # image = np.array(image)
        
        image.shape
        detections_xyxy = img_detection.xyxy.astype(int)
        cropped_images = []
        for det in detections_xyxy:
            cropped_img = image[det[1] : det[3], det[0] : det[2]]
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            cropped_img_pil = Image.fromarray(cropped_img)
            cropped_images.append(cropped_img_pil)

        return cropped_images

    def crop_detections_batch_frames(self, images, detections):
        cropped_images_batch_frames = []
        for image, img_detection in zip(images, detections):
            cropped_images = self.crop_detections(image, img_detection)
            cropped_images_batch_frames.append(cropped_images)

        return cropped_images_batch_frames
