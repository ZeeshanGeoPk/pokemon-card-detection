import os

import cv2
import faiss
import numpy as np
import pandas as pd

import utils.image as image_utils
from card_recog.classifier import CardEmbeddor


class PokemonIndex:
    def __init__(self, card_embeddor, card_detector, index_directory, d=384):
        self.d = d
        self.index_path = os.path.join(index_directory, "index.faiss")
        self.metadata_file_path = os.path.join(index_directory, "metadata.csv")
        self.card_embeddor = card_embeddor
        self.card_detector = card_detector

        if os.path.isfile(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.metadata_df = pd.read_csv(self.metadata_file_path)
        else:
            self.index = None

    def build_index(self, images_folder):
        if self.index:
            self.index.reset()
        else:
            self.index = faiss.IndexFlatIP(self.d)

        img_names_all = []
        idxs_all = []
        for idxs, img_names, imgs_batch in image_utils.load_images_from_folder(
            images_folder, batch_size=8
        ):
            idxs_array = np.array(idxs)#.reshape(-1, 1)
            imgs_batch_array = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in imgs_batch]
            
            cropped_imgs_batch = []
            for img, img_name in zip(imgs_batch_array, img_names):
                detection = self.card_detector.detect_image_single(img)
                cropped_imgs = self.card_detector.crop_detections(img, detection)
                # print("aaa", img_name)
                cropped_img_sel = cropped_imgs[0]
                cropped_imgs_batch.append(cropped_img_sel)
            # cropped_imgs = self.card_detector.crop_detections_batch_frames(imgs_batch_array, detections)
            # flatten the list
            # cropped_imgs = [img for img_list in cropped_imgs for img in img_list]
            
            embeddings = self.card_embeddor.embed_images(cropped_imgs_batch)
            embeddings = embeddings.cpu().numpy()
            faiss.normalize_L2(embeddings)
            # self.index.add_with_ids(embeddings.cpu().numpy(), idxs_array)
            self.index.add(embeddings)

            img_names_all += img_names
            idxs_all += idxs

        metadata_df = pd.DataFrame({"idx": idxs_all, "img_name": img_names_all})
        metadata_df.to_csv(self.metadata_file_path, index=False)
        self.metadata_df = metadata_df

        faiss.write_index(self.index, self.index_path)
        print(f"Updated index with {len(idxs_all)} images")

    def search(self, query_imgs, k=1):
        if self.index is None:
            raise Exception("Index is not built")
        
        query_embeddings = self.card_embeddor.embed_images(query_imgs)
        query_embeddings = query_embeddings.cpu().numpy()
        faiss.normalize_L2(query_embeddings)
        
        scores, idxs_nearest = self.index.search(query_embeddings, k)
        scores, idxs_nearest = scores[:, 0], idxs_nearest[:, 0]

        img_names_nearest = self.metadata_df.loc[
            self.metadata_df["idx"].isin(idxs_nearest), "img_name"
        ].values.tolist()
        return scores, img_names_nearest
