import json
import os

import pandas as pd
import gradio as gr
from card_recog.pipelines import PokemonScannerPipeline


def main(video_path, result_save_path="result.mp4"):
    with open("config.json") as f:
        config = json.load(f)
        
    pok_scanner = PokemonScannerPipeline(video_path, config, result_save_path)
    detected_pokemons = pok_scanner.process()
    detected_pokemons = {"Detected Card IDs from video": detected_pokemons}
    return detected_pokemons#, "./result.mp4"

def update_index():
    with open("config.json") as f:
        config = json.load(f)
    pok_scanner = PokemonScannerPipeline("snap.mp4", config)
    pok_scanner.build_index()
    scanned_files_df = pok_scanner.pokemon_index.metadata_df
    scanned_file_names = scanned_files_df["img_name"].values.tolist()
    
    scanned_files_display = scanned_files_df.iloc[:20]["img_name"].values.tolist()
    scanned_files_display = [(os.path.join(config["image_dir"], f), f) for f in scanned_files_display]
    
    updated_ids_dict = {"Updated AI model with Card IDs": scanned_file_names}
    return updated_ids_dict, scanned_files_display


def get_indexed_images():
    with open("config.json") as f:
        config = json.load(f)
        
    indexed_files_path = os.path.join(config["weights_dir"], "metadata.csv")
    indexed_files_df = pd.read_csv(indexed_files_path)
    indexed_files_names = indexed_files_df["img_name"].values.tolist()
    indexed_files_display = indexed_files_df.iloc[:20]["img_name"].values.tolist()
    indexed_files_display = [(os.path.join(config["image_dir"], f), f) for f in indexed_files_display]
    return indexed_files_display
    
    
if __name__ == "__main__":

    # video_path = "snap.mp4"
    save_path = "result.mp4"
    
    # detected_pokemons = main(video_path, save_path)
    with gr.Blocks() as demo:
        gr.Markdown("# Pokemon Card Recognition")
        gr.Markdown("Upload a video of Pokemon cards to detect the Pokemon in the video")
        with gr.Row():
            with gr.Column():
                input_video = gr.Video()
                with gr.Row():
                    run_button = gr.Button("Detect Cards from Video")
                    update_button = gr.Button("Scan + Retrain AI Model on New Cards")
            with gr.Column():
                # output_video = gr.Video()
                detections = gr.JSON(label="Output", show_label=True)
                
        indexed_images_display = gr.Gallery(get_indexed_images, label="Indexed cards", show_label=True)
        indexed_images_display.style(columns=5, padding="3px", border="1px solid black", object_fit="contain")
            
        run_button.click(main, inputs=[input_video], outputs=[detections])
        update_button.click(update_index, inputs=[], outputs=[detections, indexed_images_display])
        
    demo.launch()
        
    