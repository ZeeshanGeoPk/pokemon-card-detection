import json

from card_recog.pipelines import PokemonScannerPipeline


def main(video_path, result_save_path=None):
    with open("config.json") as f:
        config = json.load(f)
        
    pok_scanner = PokemonScannerPipeline(video_path, config, result_save_path)
    pok_scanner.build_index()
    pok_scanner.process()
    
    
if __name__ == "__main__":
    video_path = "snap.mp4"
    save_path = "result.mp4"
    
    main(video_path, save_path)