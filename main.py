import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageTk  # Still needed for displaying images
import torch
import mss  # For screenshots
import threading

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
                             QFileDialog, QMessageBox, QHBoxLayout, QSlider, QSpinBox,
                             QCheckBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDir

# Check if ultralytics is installed
try:
    import ultralytics
except ImportError:
    print("Ultralytics (YOLOv8) is not installed. Installing...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("Ultralytics installed successfully.")
        import ultralytics  # Import again after installation
    except subprocess.CalledProcessError as e:
        print(f"Error installing ultralytics: {e}")
        QMessageBox.critical(None, "Error", "Failed to install ultralytics. Please install it manually (pip install ultralytics).")
        sys.exit(1)


class DetectionWorker(QThread):
    """
    A worker thread to perform object detection in the background,
    preventing the GUI from freezing during long processing times.
    """
    finished = pyqtSignal(np.ndarray)  # Signal to emit the processed image
    error = pyqtSignal(str) # Signal to emit error messages

    def __init__(self, model, image):
        super().__init__()
        self.model = model
        self.image = image

    def run(self):
        try:
            results = self.model.predict(source=self.image, verbose=False)

            # Overlay the detections on the image
            image_with_boxes = self.image.copy()

            for result in results:
                boxes = result.boxes
                names = result.names

                for box in boxes:
                    xyxy = box.xyxy[0].tolist()  # Bounding box coordinates (x1, y1, x2, y2)
                    cls = int(box.cls[0])        # Class ID
                    confidence = box.conf[0].item() # Confidence score

                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f"{names[cls]} {confidence:.2f}"

                    # Draw rectangle and label
                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                    cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.finished.emit(image_with_boxes) # Emit signal with the processed image
        except Exception as e:
            self.error.emit(str(e))


class YoloApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Object Detection")

        self.model = None
        self.image_paths = []  # List of image file paths in the folder
        self.current_image_index = -1  # Index of the currently displayed image
        self.image = None  # Store the OpenCV image (numpy array)
        self.results = None  # Store detection results
        self.folder_path = None
        self.auto_detect = False  # Flag for automatic detection

        # UI elements
        self.load_model_button = QPushButton("Load YOLOv8 Model")
        self.load_folder_button = QPushButton("Load Folder")
        self.capture_screen_button = QPushButton("Capture Screen")
        self.run_detection_button = QPushButton("Run Detection")
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.auto_detect_checkbox = QCheckBox("Auto Detect") #Checkbox for Auto Detection

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)  # Center the image

        self.status_label = QLabel("Status: Ready")
        self.image_count_label = QLabel("Image: None")

        # Disable buttons initially
        self.load_folder_button.setEnabled(False)
        self.capture_screen_button.setEnabled(False)
        self.run_detection_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_model_button)
        button_layout.addWidget(self.load_folder_button)
        button_layout.addWidget(self.capture_screen_button)
        button_layout.addWidget(self.run_detection_button)

        navigation_layout = QHBoxLayout()
        navigation_layout.addWidget(self.prev_button)
        navigation_layout.addWidget(self.next_button)
        navigation_layout.addWidget(self.auto_detect_checkbox) #Added Checkbox to navigation layout

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(navigation_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.image_count_label)

        self.setLayout(main_layout)

        # Connections
        self.load_model_button.clicked.connect(self.load_model)
        self.load_folder_button.clicked.connect(self.load_folder)
        self.capture_screen_button.clicked.connect(self.capture_screen)
        self.run_detection_button.clicked.connect(self.run_detection)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.auto_detect_checkbox.stateChanged.connect(self.toggle_auto_detect)

        # Detection worker thread
        self.detection_worker = None

    def load_model(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select YOLOv8 Model", "", "YOLOv8 Model (*.pt)")

        if file_path:
            try:
                self.status_label.setText("Status: Loading model...")
                QApplication.processEvents()  # Update the UI

                # Check if CUDA is available and use it
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = ultralytics.YOLO(file_path)
                self.model.to(device)  # Move model to the selected device

                self.status_label.setText(f"Status: Model loaded successfully (Device: {device})")
                self.load_folder_button.setEnabled(True)
                self.capture_screen_button.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
                self.status_label.setText("Status: Error loading model")

    def load_folder(self):
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(self, "Select Image Folder")

        if folder_path:
            self.folder_path = folder_path
            self.image_paths = self.load_image_paths(folder_path)
            if self.image_paths:
                self.current_image_index = 0
                self.load_image_from_path()
                self.update_image_count_label()
                self.run_detection_button.setEnabled(True)
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Warning", "No images found in the selected folder.")
                self.image_count_label.setText("Image: None")
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)

    def load_image_from_path(self):
        if self.image_paths and self.current_image_index >= 0:
            image_path = self.image_paths[self.current_image_index]
            try:
                self.image = cv2.imread(image_path) #Load with OpenCV
                self.load_and_display_image() #Display the image.
                if self.auto_detect:
                    self.run_detection()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image from path: {e}")


    def load_image_paths(self, folder_path):
        image_paths = []
        supported_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
        dir = QDir(folder_path)
        dir.setFilter(QDir.Files | QDir.NoDotAndDotDot | QDir.NoSymLinks) # List only files, no directories

        file_infos = dir.entryInfoList()
        for file_info in file_infos:
            file_path = file_info.absoluteFilePath()
            ext = os.path.splitext(file_path)[1].lower() #Get file extension
            if ext in supported_extensions:
                image_paths.append(file_path)
        return image_paths


    def capture_screen(self):
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Use monitor 1 (primary)
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                img_np = np.array(img)
                self.image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
                self.image_paths = [] # Clear image paths as we are now displaying the screen
                self.current_image_index = -1
                self.load_and_display_image()  # Display the image from screen grab
                self.run_detection_button.setEnabled(True)
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
                if self.auto_detect:
                    self.run_detection()


        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to capture screen: {e}")


    def load_and_display_image(self):
        try:
            if self.image is None:
                return #If no image loaded from file or screenshot, exit

            # Convert to RGB for PIL
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # Convert to QImage
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Display in QLabel
            pixmap = QPixmap.fromImage(q_image)

            # Resize while preserving aspect ratio
            max_width = 600
            max_height = 400
            pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio)

            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter) # Center the image
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load and display image: {e}")

    def run_detection(self):
        if not self.model:
            QMessageBox.critical(self, "Error", "No model loaded. Please load a YOLOv8 model first.")
            return

        if self.image is None:
            QMessageBox.critical(self, "Error", "No image loaded. Please load an image or capture the screen first.")
            return

        self.status_label.setText("Status: Running detection...")
        self.run_detection_button.setEnabled(False) # Disable during processing
        QApplication.processEvents()  # Update the UI

        # Create and start the detection worker thread
        self.detection_worker = DetectionWorker(self.model, self.image)
        self.detection_worker.finished.connect(self.display_detection_results)
        self.detection_worker.error.connect(self.show_detection_error)
        self.detection_worker.start()


    def display_detection_results(self, image_with_boxes):
        """
        Displays the image with detection overlays in the GUI.
        Called when the DetectionWorker finishes successfully.
        """
        self.image = image_with_boxes  # Update the image with the detections
        self.load_and_display_image() # Load and display the new image
        self.status_label.setText("Status: Detection complete.")
        self.run_detection_button.setEnabled(True) # Re-enable the button


    def show_detection_error(self, message):
        """
        Displays an error message to the user.
        Called when the DetectionWorker encounters an error.
        """
        QMessageBox.critical(self, "Error", f"Error during detection: {message}")
        self.status_label.setText("Status: Error during detection.")
        self.run_detection_button.setEnabled(True)  #Re-enable button even after error

    def show_previous_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.load_image_from_path()
            self.update_image_count_label()

    def show_next_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.load_image_from_path()
            self.update_image_count_label()


    def update_image_count_label(self):
        if self.image_paths:
             self.image_count_label.setText(f"Image: {self.current_image_index + 1} / {len(self.image_paths)}")
        else:
            self.image_count_label.setText("Image: None")

    def toggle_auto_detect(self, state):
        self.auto_detect = (state == Qt.Checked)
        if self.auto_detect:
            self.run_detection() # Run detection if auto_detect is turned on

if __name__ == '__main__':
    app = QApplication(sys.argv)
    yolo_app = YoloApp()
    yolo_app.show()
    sys.exit(app.exec_())