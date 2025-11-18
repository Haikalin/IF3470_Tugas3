import sys
from matplotlib.pyplot import stem
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QFrame, QScrollArea, QToolTip
)
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QPalette, QColor
from PyQt5.QtCore import Qt
from pathlib import Path
from colorsys import hsv_to_rgb
from pathlib import Path




# ============================================================
# MODEL CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 183
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hasil_training.pth"

CLASS_NAMES = [
    'unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
    'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
    'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'hair brush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush',
    'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
    'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff',
    'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone',
    'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit',
    'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house',
    'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain',
    'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant-other',
    'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river',
    'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
    'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone',
    'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel',
    'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other',
    'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
    'waterdrops', 'window-blind', 'window-other', 'wood'
]

def generate_distinct_colors(num_classes):
    base_maps = [
        matplotlib.colormaps['tab20'].colors,
        matplotlib.colormaps['tab20b'].colors,
        matplotlib.colormaps['tab20c'].colors
    ]
    base_colors = np.vstack(base_maps)
    colors = []

    for i in range(min(num_classes, len(base_colors))):
        colors.append(base_colors[i])

    rng = np.random.default_rng(123)

    def color_distance(c1, c2):
        return np.sqrt(np.sum((c1 - c2) ** 2))

    while len(colors) < num_classes:
        candidates = []
        for _ in range(50):
            h = rng.random()
            s = 0.65 + rng.random() * 0.35
            v = 0.65 + rng.random() * 0.35
            candidates.append(np.array(hsv_to_rgb(h, s, v)))

        best_candidate = None
        best_min_dist = -1

        for cand in candidates:
            distances = [color_distance(cand, c) for c in colors]
            min_dist = np.min(distances)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = cand

        colors.append(best_candidate)

    return np.array(colors)


def load_model(checkpoint_path):
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded: {checkpoint_path}")
    return model


def infer(model, img: Image.Image):
    original_size = img.size

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img_tensor)["out"]
        pred = out.argmax(1).squeeze().cpu().numpy()

    pred = Image.fromarray(pred.astype(np.uint8))
    pred = pred.resize(original_size, Image.NEAREST)
    pred = np.array(pred)

    return pred


# ============================================================
# CUSTOM WIDGETS
# ============================================================

class DropZone(QLabel):
    def __init__(self, title, parent=None, allow_interaction=True):
        super().__init__(parent)
        self.title = title
        self.allow_interaction = allow_interaction
        
        # Enable hover tracking for both input/output
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover)

        if allow_interaction:
            self.setAcceptDrops(True)
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setMouseTracking(True)

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(280)
        self.setup_style()

        
    def setup_style(self):
        if self.allow_interaction:
            self.setText(f"""
                <div style='text-align: center; padding: 40px;'>
                    <p style='font-size: 48px; margin: 0;'>📤</p>
                    <p style='font-size: 16px; font-weight: bold; margin: 10px 0;'>Drop your image here</p>
                    <p style='font-size: 12px; color: #999; margin: 5px 0;'>or click to browse from your computer</p>
                    <p style='font-size: 11px; color: #bbb; margin: 20px 0 0 0;'>Supported formats: PNG, JPG, WebP</p>
                </div>
            """)
            self.setStyleSheet("""
                QLabel {
                    background-color: #f8f9fa;
                    border: 3px dashed #d0d0d0;
                    border-radius: 15px;
                    color: #666;
                }
                QLabel:hover {
                    background-color: #f0f0f5;
                    border-color: #b580ff;
                }
            """)
        else:
            self.setText(f"""
                <div style='text-align: center; padding: 40px;'>
                    <p style='font-size: 48px; margin: 0;'>🖼️</p>
                    <p style='font-size: 16px; font-weight: bold; margin: 10px 0;'>Segmentation result will appear here</p>
                    <p style='font-size: 12px; color: #999; margin: 5px 0;'>Click submit to process your image</p>
                </div>
            """)
            self.setStyleSheet("""
                QLabel {
                    background-color: #f8f9fa;
                    border: 3px dashed #d0d0d0;
                    border-radius: 15px;
                    color: #666;
                }
            """)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if self.allow_interaction and event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    background-color: #f0e6ff;
                    border: 3px dashed #b580ff;
                    border-radius: 15px;
                }
            """)
            
    def dragLeaveEvent(self, event):
        if self.allow_interaction:
            self.setup_style()
        
    def dropEvent(self, event: QDropEvent):
        if self.allow_interaction:
            files = [u.toLocalFile() for u in event.mimeData().urls()]
            if files:
                self.parent().load_image_from_path(files[0])
            self.setup_style()
        
    def mousePressEvent(self, event):
        if self.allow_interaction:
            self.parent().browse_image()
    
    def mouseMoveEvent(self, event):
        """Forward mouse move events to parent for hover functionality"""
        if not self.allow_interaction:
            self.parent().on_hover_output(event)


class LegendItem(QFrame):
    def __init__(self, color, text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Color box
        color_box = QLabel()
        color_box.setFixedSize(30, 30)
        color_box.setStyleSheet(f"""
            background-color: {color};
            border-radius: 5px;
            border: 1px solid #ddd;
        """)
        
        # Text label
        text_label = QLabel(text)
        text_label.setStyleSheet("font-size: 13px; color: #333;")
        
        layout.addWidget(color_box)
        layout.addWidget(text_label)
        layout.addStretch()
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 5px;
                margin: 2px;
            }
            QFrame:hover {
                background-color: #f5f5f5;
            }
        """)


# ============================================================
# MAIN APPLICATION
# ============================================================

class SegApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN SEGMENTATION")
        self.setGeometry(100, 50, 1500, 700)
        
        # Set window background
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(250, 250, 252))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.model = load_model(MODEL_PATH)
        self.colors = generate_distinct_colors(NUM_CLASSES)
        self.input_image_path = None
        self.input_image = None
        self.pred_mask = None
        self.input_pixmap = None
        self.legend_layout = None  # Store reference to legend layout

        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(20)
        
        # Header
        header = QLabel("CNN <span style='color: #b580ff;'>SEGMENTATION</span>")
        header.setStyleSheet("""
            font-size: 42px;
            font-weight: bold;
            color: #1a1a1a;
            margin-bottom: 5px;
        """)
        header.setAlignment(Qt.AlignCenter)
        
        main_layout.addWidget(header)        
        # Content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left side - Input and Output
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        
        # Input section
        input_header = QLabel("Input Image")
        input_header.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #b580ff, stop:1 #ff80bf);
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 12px;
            border-radius: 10px;
        """)
        left_layout.addWidget(input_header)
        
        self.input_zone = DropZone("Input Image", self, allow_interaction=True)
        self.input_zone.setMinimumHeight(280)
        left_layout.addWidget(self.input_zone)
        
        # Output section
        output_header = QLabel("Segmented Output")
        output_header.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #b580ff, stop:1 #ff80bf);
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 12px;
            border-radius: 10px;
        """)
        left_layout.addWidget(output_header)
        
        self.output_zone = DropZone("Segmented Output", self, allow_interaction=False)
        self.output_zone.setMinimumHeight(280)
        self.output_zone.setMouseTracking(True)
        left_layout.addWidget(self.output_zone)
        
        content_layout.addLayout(left_layout, 60)
        
        # Right side - Legend and Button
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        
        # Legend section
        legend_header = QLabel("Classification Legend")
        legend_header.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #b580ff, stop:1 #ff80bf);
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 12px;
            border-radius: 10px;
        """)
        right_layout.addWidget(legend_header)
        
        # Scrollable legend
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #b580ff;
                border-radius: 5px;
            }
        """)
        
        legend_widget = QWidget()
        self.legend_layout = QVBoxLayout()
        self.legend_layout.setSpacing(2)
        self.legend_layout.setContentsMargins(10, 10, 10, 10)
        
        # Initially show placeholder text
        placeholder = QLabel("Load and segment an image to see detected classes")
        placeholder.setStyleSheet("""
            color: #999;
            font-size: 13px;
            padding: 20px;
        """)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setWordWrap(True)
        self.legend_layout.addWidget(placeholder)
        
        self.legend_layout.addStretch()
        legend_widget.setLayout(self.legend_layout)
        scroll.setWidget(legend_widget)
        
        right_layout.addWidget(scroll)
        
        # Submit button
        self.btn_submit = QPushButton("⚡ Submit Segmentation")
        self.btn_submit.setMinimumHeight(60)
        self.btn_submit.setCursor(Qt.PointingHandCursor)
        self.btn_submit.setStyleSheet("""
            QPushButton {
                background-color: #cc6600;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 12px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #e67300;
            }
            QPushButton:pressed {
                background-color: #b35900;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.btn_submit.clicked.connect(self.run_segmentation)
        self.btn_submit.setEnabled(False)
        
        right_layout.addWidget(self.btn_submit)
        
        content_layout.addLayout(right_layout, 40)
        
        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self.load_image_from_path(path)
    
    def load_image_from_path(self, path):
        try:
            self.input_image = Image.open(path).convert("RGB")
            self.input_pixmap = QPixmap(path)
            self.input_image_path = path
            
            scaled_pixmap = self.input_pixmap.scaled(
                500, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.input_zone.setPixmap(scaled_pixmap)
            self.input_zone.setStyleSheet("""
                QLabel {
                    background-color: white;
                    border: 2px solid #b580ff;
                    border-radius: 15px;
                }
            """)
            
            self.btn_submit.setEnabled(True)
            
        except Exception as e:
            print(f"Error loading image: {e}")

    def run_segmentation(self):
        if self.input_image is None:
            return

        self.btn_submit.setEnabled(False)
        self.btn_submit.setText("⏳ Processing...")
        QApplication.processEvents()

        try:
            # Run inference
            self.pred_mask = infer(self.model, self.input_image)

            # Create colored visualization
            color_result = self.colors[self.pred_mask]
            color_img = Image.fromarray((color_result * 255).astype(np.uint8))

            # --- Simpan ke folder output_image ---
            stem = Path(self.input_image_path).stem
            base_dir = Path(__file__).resolve().parent
            output_dir = base_dir / "output_image"
            output_dir.mkdir(exist_ok=True)

            temp_path = output_dir / f"output_{stem}.png"
            color_img.save(temp_path)
            print(f"Saved segmented output to: {temp_path}")

            # Tampilkan di GUI
            output_pixmap = QPixmap(str(temp_path))
            scaled_output = output_pixmap.scaled(
                500, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.output_zone.setPixmap(scaled_output)
            self.output_zone.setStyleSheet("""
                QLabel {
                    background-color: white;
                    border: 2px solid #b580ff;
                    border-radius: 15px;
                }
            """)

            # Update legend
            self.update_legend()

        except Exception as e:
            print(f"Error during segmentation: {e}")

        finally:            
            self.btn_submit.setEnabled(True)
            self.btn_submit.setText("⚡ Submit Segmentation")

    
    def update_legend(self):
        """Update legend with classes found in the segmentation"""
        if self.pred_mask is None:
            return
        
        # Clear existing legend items
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get unique classes in the mask (excluding 0/unlabeled)
        unique_classes = np.unique(self.pred_mask)
        unique_classes = unique_classes[unique_classes > 0]  # Remove unlabeled
        
        # Count pixels for each class
        class_counts = []
        for cls in unique_classes:
            count = np.sum(self.pred_mask == cls)
            class_counts.append((cls, count))
        
        # Sort by count (most common first)
        class_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Add legend items for detected classes (limit to top 15)
        for cls, count in class_counts[:15]:
            color_rgb = self.colors[cls]
            color_hex = f"#{int(color_rgb[0]*255):02x}{int(color_rgb[1]*255):02x}{int(color_rgb[2]*255):02x}"
            class_name = CLASS_NAMES[cls]
            
            # Calculate percentage
            total_pixels = self.pred_mask.size
            percentage = (count / total_pixels) * 100
            
            label_text = f"{class_name.title()} ({percentage:.1f}%)"
            self.legend_layout.addWidget(LegendItem(color_hex, label_text, self))
        
        self.legend_layout.addStretch()
    
    def on_hover_output(self, event):
        if self.pred_mask is None:
            QToolTip.hideText()
            return

        if not self.output_zone.pixmap():
            QToolTip.hideText()
            return

        label_width = self.output_zone.width()
        label_height = self.output_zone.height()
        pixmap_width = self.output_zone.pixmap().width()
        pixmap_height = self.output_zone.pixmap().height()

        x_offset = (label_width - pixmap_width) // 2
        y_offset = (label_height - pixmap_height) // 2

        mouse_x = event.x() - x_offset
        mouse_y = event.y() - y_offset

        if mouse_x < 0 or mouse_y < 0 or mouse_x >= pixmap_width or mouse_y >= pixmap_height:
            QToolTip.hideText()
            return

        scale_x = self.pred_mask.shape[1] / pixmap_width
        scale_y = self.pred_mask.shape[0] / pixmap_height

        orig_x = int(mouse_x * scale_x)
        orig_y = int(mouse_y * scale_y)

        if 0 <= orig_x < self.pred_mask.shape[1] and 0 <= orig_y < self.pred_mask.shape[0]:
            cls = self.pred_mask[orig_y, orig_x]

            if 0 < cls < len(CLASS_NAMES):
                class_name = CLASS_NAMES[cls]
                tooltip = f"Class {cls}: {class_name.title()}"
                pos = self.output_zone.mapToGlobal(event.pos())
                QToolTip.showText(pos, tooltip, self.output_zone)
            else:
                QToolTip.hideText()
        else:
            QToolTip.hideText()



# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SegApp()
    window.show()
    sys.exit(app.exec_())