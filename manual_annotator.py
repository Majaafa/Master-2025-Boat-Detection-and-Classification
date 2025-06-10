import sys
import os
import cv2
import numpy as np
import shutil

from PyQt5.QtWidgets import QApplication, QLabel, QFileDialog, QMainWindow, QProgressBar
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QPoint, QRect

class ManualAnnotator(QMainWindow):
    def __init__(self, image_folder, label_folder):
        super().__init__()
        self.image_folder = image_folder
        self.label_folder = label_folder
        
        # Create bad images and labels folders
        self.bad_images_folder = os.path.join(os.path.dirname(image_folder), "bad_images")
        os.makedirs(self.bad_images_folder, exist_ok=True)
        self.bad_labels_folder = os.path.join(os.path.dirname(label_folder), "bad_labels")
        os.makedirs(self.bad_labels_folder, exist_ok=True)
        
        # Initializing
        self.image_files = []
        self.load_image_files(image_folder)
        self.bounding_boxes = []
        self.selected_box_index = -1
        self.image = None
        self.current_image_id = 0
        self.original_image_width = 0
        self.original_image_height = 0
        self.skipped = 0
        self.bad_images = 0
        
        self.resize = False
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.drawing = False
        self.setMouseTracking(True)
        self.start_point = QPoint()
        self.end_point = QPoint()
        
        self.selected_label = ""
        self.label_map = {'sailboat': 0, 'paddleboat':1, 'buoy':2, 'hobbyboat':3, 'bigboat':4, 'mediumboat':5}
        self.keybindings_text = "Keybindings:\n 's' to label sailboat\n 'c' to label kayak&canoe\n 'b' to label buoy\n 'h' to label hobbyboat\n 'x' to label bigship\n 'm' to label mediumboat\n '1' to enter delete mode\n '2' to enter change class mode\n '3' to flag as bad image\n 'left' to go back\n 'right' or 'space' to go forward\n 'esc' to exit\n 'backspace' remove last label"
        
        self.delete_mode = False
        self.change_class_mode = False
        self.bad_image_confirm = False
        self.bad_image_warning_text = "WARNING: Press '3' again to confirm moving this image to bad_images folder"
        self.delete_text = "DELETE MODE: Click on a bounding box to delete it. Press '1' again to exit delete mode or e.g. 'c'."
        self.change_class_text = "CHANGE CLASS MODE: Click on a bounding box, then press a class key (s,c,b,h,x,m) to change its class. Press '2' again to exit."
        self.bad_image_text = "BAD IMAGE: Image will be moved to bad_images folder"

        self.window_height = QApplication.desktop().screenGeometry().height()
        self.window_width = QApplication.desktop().screenGeometry().width()

        self.initGUI()
    
    # Loads all files in the given folder and subfolders (add image extensions as needed)
    def load_image_files(self, img_folder):
        for root, _, files in os.walk(img_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
        self.image_files.sort()
    
    # Initializes the GUI components
    def initGUI(self):
        self.setWindowTitle("Let's Label Some Images!")
    
        self.setGeometry(0, 0, self.window_width, self.window_height-100)

        self.label_instructions = QLabel(self)
        self.label_instructions.setGeometry(self.window_width-250, 70, 700, 300)
        self.label_instructions.setStyleSheet("color: blue")
        self.label_instructions.setText(self.keybindings_text)

        self.progress = QProgressBar(self)
        self.progress.setGeometry(50, 20, self.window_width-300, 20)
        self.progress.setMaximum(len(self.image_files))

        self.file_name_text = QLabel(self)
        self.file_name_text.setGeometry(50, self.window_height-140, 700, 20)

        self.label_image_counter_text = QLabel(self)
        self.label_image_counter_text.setGeometry(550, self.window_height-140, 700, 20)
        
        # Mode indicator
        self.label_mode = QLabel(self)
        self.label_mode.setGeometry(50, 45, self.window_width-300, 20)
        self.label_mode.setStyleSheet("color: red; font-weight: bold")
        self.label_mode.setVisible(False)
        
        # Load last image if exists
        self.current_image_id = self.load_last_image_index()
        if self.current_image_id >= len(self.image_files):
            self.current_image_id = 0
 
        self.load_image()
        self.show()

    # Prepares the image for display and loads it
    def load_image(self):
        if self.current_image_id < 0 or self.current_image_id >= len(self.image_files):
            return
        
        image_path = self.image_files[self.current_image_id]
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            print(f"Error loading image: {image_path}")
            if self.current_image_id < len(self.image_files) - 1:
                self.current_image_id += 1
                self.load_image()
            return
            
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.original_image_height, self.original_image_width = self.image.shape[:2]
        
        # Scale image to fit screen if needed
        max_display_width = self.window_width - 100
        max_display_height = self.window_height - 200
        
        if self.original_image_width > max_display_width or self.original_image_height > max_display_height:
            # Calculate scale factors
            width_ratio = max_display_width / self.original_image_width
            height_ratio = max_display_height / self.original_image_height
            scale_factor = min(width_ratio, height_ratio)
            
            # Scale image
            new_width = int(self.original_image_width * scale_factor)
            new_height = int(self.original_image_height * scale_factor)
            self.image = cv2.resize(self.image, (new_width, new_height))
            
            # Store scale factors for coordinate conversion
            self.scale_x = scale_factor
            self.scale_y = scale_factor
        else:
            self.scale_x = 1.0
            self.scale_y = 1.0
        
        # Update progress and image info
        self.progress.setValue(self.current_image_id + 1)
        self.file_name_text.setText(f"Current Image: {os.path.basename(image_path)}")
        self.label_image_counter_text.setText(f"Image {self.current_image_id + 1} of {len(self.image_files)}")
        
        self.bounding_boxes = []
        
        # Load existing annotations if available
        # Create label path with same structure as image path but under label folder
        rel_path = os.path.relpath(image_path, self.image_folder)
        label_dir = os.path.join(self.label_folder, os.path.dirname(rel_path))
        os.makedirs(label_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path)
        file_name, _ = os.path.splitext(base_name)
        label_path = os.path.join(label_dir, file_name + ".txt")
        
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                boundings = file.readlines()
                print("Loaded existing annotations from:", label_path)
                print(boundings)
            
            for param in boundings:
                parts = param.strip().split()
                if len(parts) >= 5:
                    if parts[0] == "None":
                        label = "None"
                    else:
                        try:
                            label_idx = int(parts[0])
                            label = list(self.label_map.keys())[list(self.label_map.values()).index(label_idx)]
                        except (ValueError, IndexError):
                            continue
                    
                    # Convert YOLO format back to screen coordinates
                    x_center = float(parts[1]) * self.original_image_width
                    y_center = float(parts[2]) * self.original_image_height
                    width = float(parts[3]) * self.original_image_width
                    height = float(parts[4]) * self.original_image_height
                    
                    # Calculate corners
                    x1 = int((x_center - width/2) * self.scale_x) + 50
                    y1 = int((y_center - height/2) * self.scale_y) + 70
                    x2 = int((x_center + width/2) * self.scale_x) + 50
                    y2 = int((y_center + height/2) * self.scale_y) + 70
                    
                    # Add to bounding boxes
                    self.bounding_boxes.append((QPoint(x1, y1), QPoint(x2, y2), label))
        
        # Reset modes when loading new image
        self.exit_special_modes()
        self.update()

    # Helper function for exiting special modes and reset states
    def exit_special_modes(self):
        self.delete_mode = False
        self.change_class_mode = False
        self.selected_box_index = -1
        self.label_mode.setVisible(False)
        self.bad_image_confirm = False
        self.update()

    # Find bounding box under cursor for deletion or class change
    def find_box_under_cursor(self, click_pos):
        for i, (start, end, _) in enumerate(self.bounding_boxes):
            rect = QRect(start, end)
            if rect.contains(click_pos):
                return i
        return -1
    
    # Mouse press event handler
    def mousePressEvent(self, event):
        
        if self.image is None:
            return
        
        # Optional if you want the side button on the mouse to navigate
        #------
        # X1Button: back mouse button
        if event.button() == Qt.XButton1:
            self.bad_image_confirm = False
            if self.current_image_id > 0:
                self.save_annotations()
                self.current_image_id -= 1
                self.load_image()
                
                # Update skipped status if needed
                if os.path.exists("skipped_images.txt"):
                    with open("skipped_images.txt", "r") as file:
                        file_content = file.read().splitlines()
                    
                    filename = self.image_files[self.current_image_id]
                    if filename in file_content:
                        self.skipped -= 1
                        file_content.remove(filename)
                        # Rewrite file without this entry
                        with open("skipped_images.txt", "w") as file:
                            file.write("\n".join(file_content) + ("\n" if file_content else ""))
            return 
        
        # X2Button: forward mouse button
        elif event.button() == Qt.XButton2:
            self.save_annotations()
            self.save_last_image_index()
            if self.current_image_id == len(self.image_files) - 1:
                self.close()
            else:
                self.current_image_id += 1
                self.load_image()
            return 
        #----------

        # If delete mode and mouse clicked
        if self.delete_mode and event.button() == Qt.LeftButton:
            box_index = self.find_box_under_cursor(event.pos())
            if box_index >= 0:
                self.bounding_boxes.pop(box_index)
                self.update()
            return
            
        # If change class mode and mouse clicked
        if self.change_class_mode and event.button() == Qt.LeftButton:
            self.selected_box_index = self.find_box_under_cursor(event.pos())
            if self.selected_box_index >= 0:
                self.update()
            return
         
        img_width = self.image.shape[1]
        img_height = self.image.shape[0]

        # Adjust click position to account for image offset
        click_x = event.x() - 50
        click_y = event.y() - 70
        
        # Check if click is within image bounds
        if 0 <= click_x < img_width and 0 <= click_y < img_height:

            # If not in delete or change class mode, start drawing a new bounding box
            if event.button() == Qt.LeftButton and not self.drawing:
                self.drawing = True
                self.start_point = event.pos()
                self.end_point = event.pos()
                self.update()

            # If already drawing, finalize the bounding box
            elif event.button() == Qt.LeftButton and self.drawing:
                self.drawing = False
                self.end_point = event.pos()
                if self.selected_label:  # Only add if a label is selected
                    self.bounding_boxes.append((self.start_point, self.end_point, self.selected_label))
                self.update()

    # Updates the mouse position while drawing
    def mouseMoveEvent(self, event):
        self.end_point = event.pos()
        self.update()
    
    # Displays the image and bounding boxes on the widget
    def paintEvent(self, event):

        if self.image is None:
            return

        # Create a painter to draw on the widget
        painter = QPainter(self)
        painter.drawPixmap(50, 70, QPixmap.fromImage(self.get_qimage()))

        # Set rectangle color and thickness
        pen = QPen(Qt.magenta, 1.5, Qt.SolidLine)  
        painter.setPen(pen)

        # Draw all saved bounding boxes
        for i, (start, end, label) in enumerate(self.bounding_boxes):

            # Different colors for current modes
            if self.change_class_mode and i == self.selected_box_index:
                painter.setPen(QPen(Qt.green, 3.0, Qt.SolidLine))  # Thicker green line for selected box
            else:
                painter.setPen(QPen(Qt.magenta, 1.5, Qt.SolidLine))  # Reset to default
            
            # Draw the bounding box
            rect = QRect(start, end)
            painter.drawRect(rect)
            font = QFont("Arial", 18)  # Font: Arial, Size: 18
            painter.setFont(font)
            painter.drawText(start, label)

        # Draw live bounding box if currently drawing
        if self.drawing:
            painter.setPen(QPen(Qt.magenta, 1.5, Qt.SolidLine))
            live_rect = QRect(self.start_point, self.end_point)
            painter.drawRect(live_rect)
        
        # If not drawing and end_point is set, draw guide lines to help with alignment
        if not self.drawing and self.end_point is not None:
            help_pen = QPen(Qt.gray, 1, Qt.DashLine)
            painter.setPen(help_pen)

            # Draw vertical and horizontal guide lines centered on the mouse
            painter.drawLine(self.end_point.x(), 70, self.end_point.x(), self.height()-50)  # Vertical line
            painter.drawLine(50, self.end_point.y(), self.width()-50, self.end_point.y())  # Horizontal line

    # Converts the OpenCV image to QImage for display
    def get_qimage(self):

        if self.image is None:
            return QImage()
            
        temp_image = self.image.copy()
        h, w, ch = temp_image.shape
        return QImage(temp_image.data, w, h, ch * w, QImage.Format_RGB888)

    # Handles key press events for various functionalities
    def keyPressEvent(self, event):
        key = event.key()
        
        # Handle class key presses
        if key in [Qt.Key_S, Qt.Key_C, Qt.Key_B, Qt.Key_H, Qt.Key_X, Qt.Key_M]:

            if key == Qt.Key_S:
                label = 'sailboat'
            elif key == Qt.Key_C:
                label = 'paddleboat'
            elif key == Qt.Key_B:
                label = 'buoy'
            elif key == Qt.Key_H:
                label = 'hobbyboat'
            elif key == Qt.Key_X:
                label = 'bigboat'
            elif key == Qt.Key_M:
                label = 'mediumboat'
            
            # If in change class mode and a box is selected, change its class
            if self.change_class_mode and self.selected_box_index >= 0:
                start, end, _ = self.bounding_boxes[self.selected_box_index]
                self.bounding_boxes[self.selected_box_index] = (start, end, label)
                self.selected_box_index = -1
            else:
                # Otherwise, set selected label for new boxes
                self.selected_label = label
                self.exit_special_modes()
        
        # Special mode keys

        # Toggle delete mode
        elif key == Qt.Key_1:
            self.delete_mode = not self.delete_mode
            self.change_class_mode = False
            self.selected_box_index = -1
            self.label_mode.setVisible(self.delete_mode)
            if self.delete_mode:
                self.label_mode.setText(self.delete_text)
                self.drawing = False

        # Toggle change class mode
        elif key == Qt.Key_2:
            self.change_class_mode = not self.change_class_mode
            self.delete_mode = False
            self.selected_box_index = -1
            self.label_mode.setVisible(self.change_class_mode)
            if self.change_class_mode:
                self.label_mode.setText(self.change_class_text)
                self.drawing = False
        
        # Put image in bad_images folder
        elif key == Qt.Key_3:

            # Flag current image as bad and move it to bad_images folder
            if not self.bad_image_confirm:
                self.bad_image_confirm = True
                self.label_mode.setText(self.bad_image_warning_text)
                self.label_mode.setVisible(True)

            # Second press: Flag current image as bad and move it
            else:
                self.flag_bad_image()
                self.bad_image_confirm = False
                    
        # Navigation keys
        # left for previous image
        elif key == Qt.Key_Left:
            self.bad_image_confirm = False
            if self.current_image_id > 0:
                self.save_annotations()
                self.current_image_id -= 1
                self.load_image()
                
                # Update skipped status if needed
                if os.path.exists("skipped_images.txt"):
                    with open("skipped_images.txt", "r") as file:
                        file_content = file.read().splitlines()
                    
                    filename = self.image_files[self.current_image_id]
                    if filename in file_content:
                        self.skipped -= 1
                        file_content.remove(filename)
                        # Rewrite file without this entry
                        with open("skipped_images.txt", "w") as file:
                            file.write("\n".join(file_content) + ("\n" if file_content else ""))

        # right arrow for next image or space
        elif key == Qt.Key_Right or key == Qt.Key_Space:
            self.bad_image_confirm = False
            self.save_annotations()
            self.save_last_image_index()
            if self.current_image_id == len(self.image_files) - 1:
                self.close()
            else:
                self.current_image_id += 1
                self.load_image()
                
        # Editing keys
        # Backspace to remove last bounding box
        elif key == Qt.Key_Backspace:
            self.bad_image_confirm = False
            if self.bounding_boxes:
                self.bounding_boxes.pop()
                self.update()
        
        # Escape to exit or save and close
        elif key == Qt.Key_Escape:
            self.bad_image_confirm = False
            # If in any special mode, exit that mode
            if self.delete_mode or self.change_class_mode:
                self.exit_special_modes()
            # Otherwise, save and exit
            else:
                self.save_annotations()
                self.close()
        self.update()

    # Flags the current image as bad and moves it to the bad_images folder
    def flag_bad_image(self):
        
        if self.image is None:
            return
        
        image_path = self.image_files[self.current_image_id]
        
        # Get relative path to maintain folder structure in bad_images folder
        rel_path = os.path.relpath(image_path, self.image_folder)
        bad_image_dir = os.path.join(self.bad_images_folder, os.path.dirname(rel_path))
        os.makedirs(bad_image_dir, exist_ok=True)
        
        # Move image file
        bad_image_path = os.path.join(bad_image_dir, os.path.basename(image_path))
        try:
            shutil.move(image_path, bad_image_path)
            
            # Move corresponding label file if exists
            base_name = os.path.basename(image_path)
            file_name, _ = os.path.splitext(base_name)
            
            # Original label path
            label_dir = os.path.join(self.label_folder, os.path.dirname(rel_path))
            label_path = os.path.join(label_dir, file_name + ".txt")
            
            # Bad label path
            bad_label_dir = os.path.join(self.bad_labels_folder, os.path.dirname(rel_path))
            os.makedirs(bad_label_dir, exist_ok=True)
            bad_label_path = os.path.join(bad_label_dir, file_name + ".txt")
            
            if os.path.exists(label_path):
                shutil.move(label_path, bad_label_path)
            
            # Track the bad image in a file
            self.save_bad_image_names()
            self.bad_images += 1
            
            # Temporarily show message
            self.label_mode.setText(self.bad_image_text)
            self.label_mode.setVisible(True)
            
            # Continue to the next image
            if self.current_image_id < len(self.image_files) - 1:
                # Remove the flagged image from the list
                self.image_files.pop(self.current_image_id)
                # Load the next image (which is now at the same index)
                self.load_image()
            else:
                # If we're at the last image, go back one or just close
                if len(self.image_files) > 1:
                    self.image_files.pop(self.current_image_id)
                    self.current_image_id = max(0, self.current_image_id - 1)
                    self.load_image()
                else:
                    self.close()
                
        except Exception as e:
            print(f"Error moving bad image: {e}")

    # Saves the annotations to the label folder
    def save_annotations(self):
        if self.image is None:
            return
            
        # Create label path with same structure as image path but under label folder
        image_path = self.image_files[self.current_image_id]
        rel_path = os.path.relpath(image_path, self.image_folder)
        label_dir = os.path.join(self.label_folder, os.path.dirname(rel_path))
        os.makedirs(label_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path)
        file_name, _ = os.path.splitext(base_name)
        label_path = os.path.join(label_dir, file_name + ".txt")
        
        with open(label_path, "w") as file:
            if not self.bounding_boxes:
                self.save_skipped_image_names()
                self.skipped += 1
            else:
                for (start, end, label) in self.bounding_boxes:

                    # Calculate coordinates in image space
                    x1 = (start.x() - 50) / self.scale_x
                    y1 = (start.y() - 70) / self.scale_y
                    x2 = (end.x() - 50) / self.scale_x
                    y2 = (end.y() - 70) / self.scale_y

                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, self.original_image_width))
                    y1 = max(0, min(y1, self.original_image_height))
                    x2 = max(0, min(x2, self.original_image_width))
                    y2 = max(0, min(y2, self.original_image_height))

                    # Compute YOLO format
                    x_center = ((x1 + x2) / 2) / self.original_image_width
                    y_center = ((y1 + y2) / 2) / self.original_image_height
                    width = abs(x2 - x1) / self.original_image_width
                    height = abs(y2 - y1) / self.original_image_height
                    
                    # Ensure values are within 0-1 range
                    x_center = max(0, min(x_center, 1))
                    y_center = max(0, min(y_center, 1))
                    width = max(0, min(width, 1))
                    height = max(0, min(height, 1))
                    
                    file.write(f"{self.label_map.get(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def save_last_image_index(self):
        with open("last_image.txt", "w") as file:
            file.write(str(self.current_image_id))

    def save_skipped_image_names(self):
        # Read existing skipped images
        skipped_images = []
        if os.path.exists("skipped_images.txt"):
            with open("skipped_images.txt", "r") as file:
                skipped_images = file.read().splitlines()
        
        filename = self.image_files[self.current_image_id]
        if filename not in skipped_images:
            skipped_images.append(filename)
            
            # Write updated list
            with open("skipped_images.txt", "w") as file:
                file.write("\n".join(skipped_images) + "\n")

    def save_bad_image_names(self):
        # Read existing bad images
        bad_images = []
        if os.path.exists("bad_images.txt"):
            with open("bad_images.txt", "r") as file:
                bad_images = file.read().splitlines()
        
        filename = self.image_files[self.current_image_id]
        if filename not in bad_images:
            bad_images.append(filename)
            
            # Write updated list
            with open("bad_images.txt", "w") as file:
                file.write("\n".join(bad_images) + "\n")

    def load_last_image_index(self):
        if os.path.exists("last_image.txt"):
            with open("last_image.txt", "r") as file:
                index = file.read().strip()
                if index.isdigit():
                    return int(index)
        return 0  


if __name__ == "__main__":
    app = QApplication(sys.argv)
    image_folder = QFileDialog.getExistingDirectory(None, "Select Image Folder")
    label_folder = QFileDialog.getExistingDirectory(None, "Select Label Folder")
    
    if image_folder:
        window = ManualAnnotator(image_folder,label_folder)
        sys.exit(app.exec_())