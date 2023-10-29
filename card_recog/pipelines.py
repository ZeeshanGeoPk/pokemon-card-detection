import os

import warnings
import pandas as pd
import supervision as sv
import tqdm
import utils.video as video_utils
from card_recog.classifier import CardEmbeddor
from card_recog.detector import CardDetector
from card_recog.embeddings import PokemonIndex


warnings.filterwarnings("ignore")


class PokemonScannerPipeline:
    def __init__(self, video_path, config, result_save_path=None):
        self.config = config
        
        self.video_reader = video_utils.VideoReader(
            video_path,
            extraction_fps=config["extraction_fps"],
            frames_per_batch=config["frames_per_batch"],
        )

        gd_config = os.path.join(
            config["weights_dir"], config["grounding_dino_config_file"]
        )
        gd_weights = os.path.join(
            config["weights_dir"], config["grounding_dino_weights_file"]
        )

        self.card_detector = CardDetector(
            config["detection_prompt"],
            gd_config,
            gd_weights,
            config["detection_text_threshold"],
            config["detection_box_threshold"],
            config["device"],
        )

        self.card_embeddor = CardEmbeddor(config["embedding_model"], config["device"])
        self.pokemon_index = PokemonIndex(self.card_embeddor, self.card_detector, config["weights_dir"])

        if result_save_path is not None:
            self.video_writer = video_utils.VideoWriter(
                result_save_path, config["extraction_fps"], self.video_reader.frame_size
            )
        else:
            self.video_writer = None

    def process(self):
        i = 0
        detections_df = pd.DataFrame(columns=["timestamp", "pokemon"])
        for frames_batch, frames_timestamps in tqdm.tqdm(self.video_reader.read_2()):
            detections_batch = self.card_detector.detect_images(frames_batch)
            detections_frames_cropped = self.card_detector.crop_detections_batch_frames(
                frames_batch, detections_batch
            )

            for frame, frame_timestamp, detections_frame_cropped, detection in zip(
                frames_batch, frames_timestamps, detections_frames_cropped, detections_batch
            ):
                # for dfc in detections_frame_cropped:
                #     dfc.save(f"debug/{i:04d}.png")
                #     i += 1
                
                if len(detections_frame_cropped) == 0:
                    if self.video_writer is not None:
                        self.video_writer.add_frame(frame)
                    continue
                
                img_nearest_scores, img_names_nearest = self.pokemon_index.search(
                    detections_frame_cropped
                )
                
                img_names_nearest_sel = [img_name for img_name, score in zip(img_names_nearest, img_nearest_scores) 
                    if (score > 0.61) and (
                        img_name not in detections_df.loc[detections_df["timestamp"].between(frame_timestamp - 10_000, frame_timestamp), "pokemon"].values
                    )
                ]
                img_names_nearest_sel_df = pd.DataFrame({"timestamp": [frame_timestamp], "pokemon": [img_names_nearest_sel]})
                img_names_nearest_sel_df = img_names_nearest_sel_df.explode("pokemon")
                detections_df = pd.concat([detections_df, img_names_nearest_sel_df], ignore_index=True)
                
                # labels = [f"{name} | ({score:.2f})" for name, score in zip(img_names_nearest, img_nearest_scores)]
                labels = [f"" for name, score in zip(img_names_nearest, img_nearest_scores)]
                
                annotated_frame = self.card_detector.annotate_image(frame, detection, labels)
                if self.video_writer is not None:
                    self.video_writer.add_frame(annotated_frame)
                    
                if img_nearest_scores[0] > 0.6:
                    print("Sc", img_nearest_scores, img_names_nearest)
                    
        if self.video_writer is not None:
            self.video_writer.close()
            
        return detections_df["pokemon"].dropna().values.tolist()
            
    def build_index(self):
        self.pokemon_index.build_index(self.config["image_dir"])
