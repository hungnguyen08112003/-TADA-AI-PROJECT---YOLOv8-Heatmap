import cv2
import numpy as np
import tkinter as tk
import threading
from skimage.transform import resize
from ultralytics import YOLO
from tkinter import filedialog
from PIL import Image, ImageTk
import math
import torch 

# Parameters
frame_width = 1280
frame_height = 720
cell_size = 40
n_cols = frame_width // cell_size
n_rows = frame_height // cell_size
model = YOLO("best300epochs.pt")
heat_matrix = np.zeros((n_rows, n_cols))

#Use GPU 
device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),10 , (frame_width, frame_height))
w= frame_width
print("Frame Width", w)
h= frame_height
print("Frame Height", h)

# Function to get row and column from coordinates
def get_row_col(x, y):
    row = y // cell_size
    col = x // cell_size
    return row, col

# Function to draw grid 
def draw_grid(image):
    for i in range(n_rows):
        start_point = (0, (i + 1) * cell_size)
        end_point = (frame_width, (i + 1) * cell_size)
        color = (255, 255, 255)
        thickness = 1
        image = cv2.line(image, start_point, end_point, color, thickness)

    for i in range(n_cols):
        start_point = ((i + 1) * cell_size, 0)
        end_point = ((i + 1) * cell_size, frame_height)
        color = (255, 255, 255)
        thickness = 1
        image = cv2.line(image, start_point, end_point, color, thickness)

    return image

# Funtion return the output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

class VideoState:
    def __init__(self):
        self.paused = False
        self.accumulated_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

# Main video stream function
def video_stream(video, video_state):
    global heat_matrix
    count =0
    current_frame =0
    resized_image_heat =0  
    accumulated_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
   
    while True:
        if not video_state.paused:
            for _ in range(7):
                ret, frame = video.read()
                if not ret:
                    break
        count +=1
        result = list(model.predict(frame, conf=0.6, classes=0))[0]
        bbox_xyxys= result.boxes.xyxy.tolist()
        confidences = result.boxes.conf
        labels = list(result.names.values()) 
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 =bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name= 'person'
                conf=math.ceil((confidence*100))/100
                label=f'{class_name}{conf}'
                print("frame N", count, "", x1, x2, y1, y2)
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2) [0]
                c2 = x1 + t_size[0], y1 - t_size[1]-3
                r, c = get_row_col( (x2 + x1)//2, (y2 + y1)//2)
                heat_matrix [r, c]+=1
                cv2.rectangle(frame, (x1, y1),(x2, y2), (0,255, 255), 3)
                cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1-2), 0,1, [255, 255, 255], thickness=1, lineType= cv2.LINE_AA)

        temp_heat_matrix = heat_matrix.copy()
        temp_heat_matrix = resize(temp_heat_matrix, (frame_height, frame_width))  # Resized
        temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
        temp_heat_matrix = np.uint8(temp_heat_matrix * 255)
        accumulated_heatmap += temp_heat_matrix
        video_state.accumulated_heatmap += temp_heat_matrix

        image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)

        if not isinstance(heat_matrix[current_frame], np.ndarray):
            heat_matrix[current_frame] = np.array(heat_matrix[current_frame])

        if not isinstance(resized_image_heat, np.ndarray):
            resized_image_heat = np.array(resized_image_heat)

        if heat_matrix[current_frame].shape == resized_image_heat.shape:
                resized_image_heat = cv2.resize(resized_image_heat, (heat_matrix[current_frame].shape[1], heat_matrix[current_frame].shape[0]))
                heat_matrix[current_frame] = resized_image_heat  # Assign the data has resized
        
        frame = draw_grid(frame)
        cv2.addWeighted(image_heat, 0.5, frame, 0.5, 0, frame)
        cv2.imshow("Video", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            video_state.paused = not video_state.paused
        elif key == ord('f'):
            for _ in range(30):
                video.read()
        elif key == ord('r'):
            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            video.set(cv2.CAP_PROP_POS_FRAMES, max(current_frame - 30, 0))

        elif key == ord('h'):
        # Calculator and show the accumulated heatmap
            final_heatmap = accumulated_heatmap / np.max(accumulated_heatmap)
            final_heatmap = np.uint8(final_heatmap * 255)
            final_heatmap = cv2.applyColorMap(final_heatmap, cv2.COLORMAP_JET)

            final_heatmap_with_grid = draw_grid(final_heatmap)
            cv2.imshow("Accumulated Heatmap with Grid", final_heatmap_with_grid)
    video.release()
    cv2.destroyAllWindows()

video_state = VideoState()
# Main function to initialize the program
def main():
    global video_state
    video_state.accumulated_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    def fast_forward(video):
        for _ in range(30):
            video.read()

    def rewind(video):
        global heat_matrix
        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        video.set(cv2.CAP_PROP_POS_FRAMES, max(current_frame - 30, 0))
       
    def toggle_play_pause():
        video_state.paused = not video_state.paused

    def show_accumulated_heatmap():
        final_heatmap = video_state.accumulated_heatmap / np.max(video_state.accumulated_heatmap)
        final_heatmap = np.uint8(final_heatmap * 255)
        final_heatmap = cv2.applyColorMap(final_heatmap, cv2.COLORMAP_JET)

        final_heatmap_with_grid = draw_grid(final_heatmap)
        cv2.imshow("Accumulated Heatmap with Grid", final_heatmap_with_grid)

    def change_video_source():
        new_source = get_video_file()
        if new_source:
            global video
            global heat_matrix
            video = cv2.VideoCapture(new_source)
            video_thread = threading.Thread(target=video_stream , args=(video, video_state))
            video_thread.daemon = True
            video_thread.start()

            label3.pack_forget()
            entry3.pack_forget()
            button2.pack_forget()

            fast_forward_button.pack()
            play_button.pack()
            rewind_button.pack()
            heatmap_button.pack()


    def get_video_file():
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        return file_path
        
    main_window = tk.Tk()
    main_window.title("TADA7 PROJECT")
    main_window.geometry("720x720") 
    
    def compare_entries():
        input1 = entry1.get()
        input2 = entry2.get()


        if input1 == "admin" and input2 == "tada7":
            label1.pack_forget()
            entry1.pack_forget()

            label2.pack_forget()
            entry2.pack_forget()

            button1.pack_forget()

            label3.pack()
            entry3.pack()
            button2.pack()  

        else:
            print("Từ chối")

    def on_return(event):
        compare_entries()

    canvas = tk.Canvas(main_window, width=300, height=300)
    canvas.pack()

    # Add BackGround
    image = Image.open("Back Ground.jpg")  
    photo = ImageTk.PhotoImage(image)

    background_label = tk.Label(main_window, image=photo)
    background_label.place(relwidth=1, relheight=1)

    label1 = tk.Label(main_window, text="User Name:")
    label1.pack()

    entry1 = tk.Entry(main_window)
    entry1.pack()

    label2 = tk.Label(main_window, text="Password:")
    label2.pack()

    entry2 = tk.Entry(main_window)
    entry2.pack()

    label3 = tk.Label(main_window, text="Video Path:")
    entry3 = tk.Entry(main_window)

    button1 = tk.Button(main_window, text="Login", command=compare_entries)
    button1.pack()

    button2 = tk.Button(main_window, text="Choose Video", command=change_video_source)  # Create button choose Video
    # Click Button 2
    button2.pack_forget()

    # Add funtion to interactive with the interface
    fast_forward_button = tk.Button(main_window, text="Fast Forward", command=lambda: fast_forward(video))
    fast_forward_button.pack_forget()

    rewind_button = tk.Button(main_window, text="Rewind", command=lambda: rewind(video))
    rewind_button.pack_forget()

    play_button = tk.Button(main_window, text="Play/Pause", command= toggle_play_pause)
    play_button.pack_forget()

    heatmap_button = tk.Button(main_window, text="Show Heatmap", command=show_accumulated_heatmap)
    heatmap_button.pack_forget()

    # Link the content that being add to compare_entries()
    main_window.bind('<Return>', on_return)
    main_window.mainloop()
    

if __name__ == "__main__":
    main()
    