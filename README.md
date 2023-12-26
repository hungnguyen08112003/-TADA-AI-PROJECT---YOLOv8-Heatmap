---
comments: true
description: Human Detection with YOLOv8 and Heatmap Generation
keywords: Ultralytics, YOLOv8, Advanced Data Visualization, Heatmap Technology, Object Detection and Tracking, Jupyter Notebook, Python SDK, Command Line Interface
---

# TADA Project: Human Detection with YOLOv8 and Heatmap Generation 

## Introduction to Heatmaps
  Unlocking the power of data visualization, TADA's custom heatmaps, generated seamlessly with Ultralytics YOLOv8, translate intricate data into a vivid, personalized experience. This visual tool employs a captivating spectrum of colors, with warmer hues indicating heightened intensities and cooler tones indicating lower values. Tailored for TADA's unique insights, these heatmaps excel in unveiling intricate patterns, correlations, and anomalies within our UNIQLO retail environment. Providing an immersive and tailored approach, TADA's heatmaps revolutionize the interpretation of data, creating a visual narrative that speaks uniquely to our retail insights and customer dynamics.
## Why Choose YOLOv8 for this Project?
- **Speed and Efficiency:** YOLOv8 is renowned for its exceptional speed and efficiency in real-time object detection. This ensures that our system can promptly identify and analyze individuals within the retail environment, contributing to quick decision-making processes.
- **Accuracy:** YOLOv8 is designed to deliver high accuracy in object detection tasks. This is paramount for our project as we aim to precisely identify and analyze individuals, allowing for reliable heatmap generation and detailed insights into customer behaviors.
- **Ease of Use:** YOLOv8 is known for its user-friendly design, making it accessible for our team to implement and maintain. This ease of use streamlines the development and deployment processes, allowing us to focus on delivering valuable insights rather than grappling with complex technicalities.
- **Versatility:** YOLOv8 is a versatile model that excels in various scenarios, including object detection and tracking. This adaptability is crucial for our retail-focused project, where we need a robust and flexible solution to capture diverse customer interactions within UNIQLO stores.
- **Community Support:** YOLOv8 benefits from a robust community and continuous development. This ensures that our implementation remains up-to-date with the latest advancements, improvements, and security patches, providing a reliable foundation for our project's success.
## Real World Applications

| Detect                                                                                                 | Heatmap                                                                                                |
|:------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
| ![Ultralytics YOLOv8 Transportation Heatmap](https://github.com/hungnguyen08112003/AI-PJ/assets/134583410/4183bb95-e01b-4b68-a5e6-396c76872623){:width="640px" height="360px"} | ![Ultralytics YOLOv8 Retail Heatmap](https://github.com/hungnguyen08112003/AI-PJ/assets/134583410/445fefc3-76b2-45ff-980a-bc54209a68be){:width="640px" height="360px"} |
| Ultralytics YOLOv8 Detect Human                                                                       | Ultralytics YOLOv8 Retail Heatmap                                                                      |
## System Overview

### Application Parameters

| Parameter          | Type                | Default                                          | Description                                              |
|--------------------|---------------------|--------------------------------------------------|----------------------------------------------------------|
| frame_width        | `int`               | 1280                                             | Width of the video frame                                  |
| frame_height       | `int`               | 720                                              | Height of the video frame                                 |
| cell_size          | `int`               | 40                                               | Size of grid cells for heatmap generation                |
| n_cols             | `int`               | Calculated based on frame_width and cell_size     | Number of columns in the grid                             |
| n_rows             | `int`               | Calculated based on frame_height and cell_size    | Number of rows in the grid                                |
| model              | `YOLO`              | Initialized with "best300epochs.pt" weights      | YOLO object detection model                              |
| heat_matrix        | `numpy.ndarray`     | -                                                | Matrix for accumulating heatmap data                      |
| device             | `torch.device`      | GPU if available, else CPU                        | Device for model execution                                |
| out                | `cv2.VideoWriter`   | -                                                | Video writer object for output video file                 |
| w                  | `int`               | Alias for frame_width                            | -                                                          |
| h                  | `int`               | Alias for frame_height                           | -                                                          |

### Video Stream Parameters
| Parameter             | Type              | Description                                       |
|-----------------------|-------------------|---------------------------------------------------|
| count                 | `int`             | Counter for frames processed                      |
| current_frame         | `int`             | Current frame index                               |
| resized_image_heat    | `int`             | Placeholder for resized heat image                |
| accumulated_heatmap   | `numpy.ndarray`   | Matrix for accumulating heatmap over time         |

### GUI Parameters
- **VideoState:** Class to store video playback state, including pause status and accumulated heatmap.
### Functions
| Function                            | Description                                                                                                      |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------|
| get_row_col(x, y)                   | Function to calculate the row and column indices based on coordinates.                                           |
| draw_grid(image)                    | Function to draw a grid on the input image.                                                                       |
| get_output_layers(net)              | Function to get the output layers of the YOLO model.                                                              |
| video_stream(video, video_state)    | Main function for processing video frames, performing object detection, and generating heatmaps.                 |
| fast_forward(video)                 | Function to skip forward in the video.                                                                            |
| rewind(video)                       | Function to rewind the video.                                                                                     |
| toggle_play_pause()                 | Function to toggle play/pause status.                                                                            |
| show_accumulated_heatmap()          | Function to display the accumulated heatmap.                                                                     |
| change_video_source()               | Function to change the video source.                                                                             |

### GUI Components
| GUI Component     | Description                                            |
|-------------------|--------------------------------------------------------|
| main_window       | Tkinter main window for the user interface.            |
| compare_entries() | Function to compare login credentials.                 |
| on_return(event)   | Function to handle the return key press event.         |

### Usage
- 1  Install dependencies using pip install -r requirements.txt.
- 2 Run the script, and a GUI will prompt for a username and password.
- 3 Upon successful login, choose a video file, control playback, and visualize the accumulated heatmap.
