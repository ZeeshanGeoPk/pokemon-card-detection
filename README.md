# Pokemon Card Detection App

This is a project that aims to detect and identify pokemon cards from videos of streamers opening pokemon packs. The project uses **OpenCV** and **Pytorch** to perform image processing and deep learning tasks.

## Installation

To install the required dependencies, you need to have **conda** installed on your system. Then, you can create a conda environment using the following command:

```bash
conda env create --file environment.yml -n py
```

This will create an environment named `py` with all the necessary packages. To activate the environment, use:

```bash
conda activate py
```

## Usage

To run the application, use:

```bash
python app.py
```

This will launch a web interface where you can upload a video file. The application will then detect and identify the pokemon cards in the video and output the numbers of cards pulled to an excel file.

##Demo
![demo](https://github.com/ZeeshanGeoPk/pokemon-card-detection/assets/108798674/d78cae27-6926-4d20-9f1c-17a818d9cb79)
