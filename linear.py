#!/usr/bin/env python3
"""
🔄 Matrix Transformation Studio - Complete Application
Image Processing dengan Matrix Transformations menggunakan Python dan Streamlit

Features:
- 5 Types of Transformations: Translation, Scaling, Rotation, Shearing, Reflection
- Real-time Preview and Matrix Visualization
- Preset Transformations
- Image Download
- Demo Mode
- CLI Interface

Author: Your Name
License: MIT
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import sys
import os
import argparse
import tempfile
import shutil
from typing import Tuple, Optional, Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    APP_TITLE = "🔄 Matrix Transformation Studio"
    APP_DESCRIPTION = "Image Processing dengan Matrix Transformations"
    VERSION = "1.0.0"
    
    # Image settings
    MAX_IMAGE_SIZE = 2000
    SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    
    # Transformation limits
    TRANSLATION_RANGE = (-200, 200)
    SCALING_RANGE = (0.1, 3.0)
    ROTATION_RANGE = (-180, 180)
    SHEARING_RANGE = (-1.0, 1.0)
    
    # UI Settings
    SIDEBAR_WIDTH = "300px"
    IMAGE_WIDTH = 400

# ============================================================================
# CORE TRANSFORMATION ENGINE
# ============================================================================

class MatrixTransformer:
    """
    Core class untuk melakukan transformasi matriks pada gambar
    """
    
    def __init__(self):
        self.image = None
        self.transformed_image = None
        self.original_shape = None
    
    def load_image(self, image_source) -> bool:
        """
        Load image dari berbagai sumber (file upload, file path, atau numpy array)
        """
        try:
            if hasattr(image_source, 'read'):  # UploadedFile
                file_bytes = np.asarray(bytearray(image_source.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            elif isinstance(image_source, str):  # File path
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"Image file not found: {image_source}")
                image = cv2.imread(image_source)
            elif isinstance(image_source, np.ndarray):  # Numpy array
                image = image_source.copy()
            else:
                raise ValueError("Unsupported image source type")
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Convert BGR ke RGB untuk matplotlib
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Store original shape
            self.original_shape = image.shape
            
            # Check image size
            if image.shape[0] > Config.MAX_IMAGE_SIZE or image.shape[1] > Config.MAX_IMAGE_SIZE:
                # Resize image if too large
                scale = min(Config.MAX_IMAGE_SIZE / image.shape[0], 
                           Config.MAX_IMAGE_SIZE / image.shape[1])
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            self.image = image
            return True
            
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return False
    
    def create_transformation_matrix(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Membuat matriks transformasi 3x3 berdasarkan parameter
        """
        # Start dengan matriks identitas
        transform_matrix = np.eye(3)
        
        # Ekstrak parameter dengan default values
        tx = params.get('translation_x', 0)
        ty = params.get('translation_y', 0)
        sx = params.get('scaling_x', 1)
        sy = params.get('scaling_y', 1)
        rotation = params.get('rotation', 0)
        shear_x = params.get('shearing_x', 0)
        shear_y = params.get('shearing_y', 0)
        reflect_h = params.get('reflection_horizontal', False)
        reflect_v = params.get('reflection_vertical', False)
        
        # Convert rotasi ke radian
        angle_rad = np.radians(rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Build transformation matrix step by step
        # 1. Reflection
        if reflect_h:
            reflect_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
            transform_matrix = reflect_matrix @ transform_matrix
        
        if reflect_v:
            reflect_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
            transform_matrix = reflect_matrix @ transform_matrix
        
        # 2. Scaling
        scale_matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        transform_matrix = scale_matrix @ transform_matrix
        
        # 3. Rotation
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        transform_matrix = rotation_matrix @ transform_matrix
        
        # 4. Shearing
        shear_matrix = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]])
        transform_matrix = shear_matrix @ transform_matrix
        
        # 5. Translation
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        transform_matrix = translation_matrix @ transform_matrix
        
        return transform_matrix
    
    def apply_transformation(self, matrix: np.ndarray) -> np.ndarray:
        """
        Aplikasikan transformasi ke gambar
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        height, width = self.image.shape[:2]
        
        # Dapatkan koordinat sudut gambar
        corners = np.array([
            [0, 0, 1],
            [width, 0, 1],
            [width, height, 1],
            [0, height, 1]
        ]).T
        
        # Transformasi koordinat
        transformed_corners = matrix @ corners
        transformed_corners = transformed_corners[:2] / transformed_corners[2]
        
        # Hitung bounding box baru
        min_x, min_y = np.min(transformed_corners, axis=1)
        max_x, max_y = np.max(transformed_corners, axis=1)
        
        # Ukuran canvas baru dengan padding
        padding = 50
        new_width = int(max_x - min_x + 2 * padding)
        new_height = int(max_y - min_y + 2 * padding)
        
        # Matriks transformasi untuk OpenCV
        # Convert dari 3x3 ke 2x3
        cv_matrix = matrix[:2, :]
        
        # Adjust translation untuk centering
        cv_matrix[0, 2] -= min_x - padding
        cv_matrix[1, 2] -= min_y - padding
        
        # Aplikasikan transformasi
        self.transformed_image = cv2.warpAffine(
            self.image, cv_matrix, (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return self.transformed_image
    
    def get_preset_transformations(self) -> Dict[str, Dict[str, Any]]:
        """
        Mendapatkan preset transformasi yang umum
        """
        return {
            "Flip Horizontal": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': True, 'reflection_vertical': False
            },
            "Flip Vertical": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': True
            },
            "Rotasi 90°": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 90,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            },
            "Rotasi 180°": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 180,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            },
            "Scale 2x": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 2, 'scaling_y': 2,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            },
            "Scale 0.5x": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 0.5, 'scaling_y': 0.5,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            },
            "Skew Kanan": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 0,
                'shearing_x': 0.3, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            },
            "Skew Atas": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': -0.3,
                'reflection_horizontal': False, 'reflection_vertical': False
            }
        }

# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def display_matrix(matrix: np.ndarray, title: str = "Matriks Transformasi"):
        """
        Menampilkan matriks dengan format yang bagus
        """
        st.subheader(title)
        
        # Format matriks untuk display
        matrix_str = ""
        for i in range(matrix.shape[0]):
            row_str = " | ".join([f"{matrix[i,j]:8.3f}" for j in range(matrix.shape[1])])
            matrix_str += f"[ {row_str} ]\n"
        
        st.code(matrix_str, language='text')
        
        # Penjelasan komponen matriks
        with st.expander("Penjelasan Matriks"):
            st.markdown("""
            **Komponen Matriks 3x3:**
            - **[0,0], [0,1], [1,0], [1,1]**: Transformasi linear (rotasi, skala, shear)
            - **[0,2], [1,2]**: Translasi (X, Y)
            - **[2,0], [2,1]**: Perspektif (tidak digunakan dalam implementasi ini)
            - **[2,2]**: Koordinat homogen
            """)
    
    @staticmethod
    def create_sample_image(width=400, height=300):
        """Create a sample image for demo"""
        # Create gradient background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient
        for i in range(height):
            image[i, :, :] = [int(255 * i / height), 100, 255 - int(255 * i / height)]
        
        # Add some shapes
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(image, (300, 150), 50, (0, 255, 0), -1)
        cv2.line(image, (0, 0), (width, height), (255, 0, 0), 3)
        
        # Add text
        cv2.putText(image, "DEMO", (width//2 - 50, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        return image
    
    @staticmethod
    def display_image_comparison(original, transformed, title_original="Original", title_transformed="Transformed"):
        """Display side-by-side image comparison"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📷 {title_original}")
            if original is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(original)
                ax.set_title(title_original)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No image to display")
        
        with col2:
            st.subheader(f"✨ {title_transformed}")
            if transformed is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(transformed)
                ax.set_title(title_transformed)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                
                # Download button
                if st.button("💾 Download Transformed Image"):
                    UIComponents.download_image(transformed)
            else:
                st.info("No transformed image to display")
    
    @staticmethod
    def download_image(image):
        """Download image as PNG"""
        pil_image = Image.fromarray(image)
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.download_button(
            label="📥 Download PNG",
            data=img_buffer,
            file_name="transformed_image.png",
            mime="image/png"
        )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class MatrixTransformationApp:
    """Main application class"""
    
    def __init__(self):
        self.transformer = MatrixTransformer()
        self.ui = UIComponents()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=Config.APP_TITLE,
            page_icon="🔄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_header(self):
        """Render application header"""
        st.title(Config.APP_TITLE)
        st.markdown(Config.APP_DESCRIPTION)
        st.markdown(f"Version {Config.VERSION}")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("🎛️ Kontrol Transformasi")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Gambar",
            type=Config.SUPPORTED_FORMATS,
            help="Upload gambar untuk di-transformasi"
        )
        
        # Demo mode toggle
        demo_mode = st.sidebar.checkbox("🎨 Demo Mode", help="Use sample image for demo")
        
        if demo_mode:
            # Create sample image
            sample_image = self.ui.create_sample_image()
            self.transformer.load_image(sample_image)
            st.sidebar.success("✅ Demo image loaded!")
        elif uploaded_file is not None:
            if self.transformer.load_image(uploaded_file):
                st.sidebar.success("✅ Gambar berhasil dimuat!")
        
        # Only show transformation controls if image is loaded
        if self.transformer.image is not None:
            self.render_transformation_controls()
    
    def render_transformation_controls(self):
        """Render transformation parameter controls"""
        st.sidebar.subheader("📐 Parameter Transformasi")
        
        # Translation
        st.sidebar.markdown("**🔄 Translation (Translasi)**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            tx = st.slider("X (pixels)", *Config.TRANSLATION_RANGE, 0, key="tx")
        with col2:
            ty = st.slider("Y (pixels)", *Config.TRANSLATION_RANGE, 0, key="ty")
        
        # Scaling
        st.sidebar.markdown("**📏 Scaling (Skala)**")
        col3, col4 = st.sidebar.columns(2)
        with col3:
            sx = st.slider("X", *Config.SCALING_RANGE, 1.0, 0.1, key="sx")
        with col4:
            sy = st.slider("Y", *Config.SCALING_RANGE, 1.0, 0.1, key="sy")
        
        # Rotation
        st.sidebar.markdown("**🔄 Rotation (Rotasi)**")
        rotation = st.slider("Sudut (derajat)", *Config.ROTATION_RANGE, 0, key="rotation")
        
        # Shearing
        st.sidebar.markdown("**🔀 Shearing (Skew)**")
        col5, col6 = st.sidebar.columns(2)
        with col5:
            shear_x = st.slider("X", *Config.SHEARING_RANGE, 0.0, 0.1, key="shear_x")
        with col6:
            shear_y = st.slider("Y", *Config.SHEARING_RANGE, 0.0, 0.1, key="shear_y")
        
        # Reflection
        st.sidebar.markdown("**🔁 Reflection (Refleksi)**")
        col7, col8 = st.sidebar.columns(2)
        with col7:
            reflect_h = st.checkbox("Horizontal", key="reflect_h")
        with col8:
            reflect_v = st.checkbox("Vertical", key="reflect_v")
        
        # Preset transformations
        st.sidebar.subheader("⚡ Preset Transformations")
        presets = self.transformer.get_preset_transformations()
        
        selected_preset = st.sidebar.selectbox(
            "Pilih Preset:",
            ["None"] + list(presets.keys())
        )
        
        if selected_preset != "None":
            preset_params = presets[selected_preset]
            # Update session state dengan preset values
            for key, value in preset_params.items():
                st.session_state[key.replace('translation_', '').replace('scaling_', '').replace('shearing_', '').replace('reflection_', '')] = value
            st.rerun()
        
        # Reset button
        if st.sidebar.button("🔄 Reset All", type="secondary"):
            for key in list(st.session_state.keys()):
                if key.startswith(('tx', 'ty', 'sx', 'sy', 'rotation', 'shear_', 'reflect_')):
                    del st.session_state[key]
            st.rerun()
        
        # Create transformation parameters
        params = {
            'translation_x': tx,
            'translation_y': ty,
            'scaling_x': sx,
            'scaling_y': sy,
            'rotation': rotation,
            'shearing_x': shear_x,
            'shearing_y': shear_y,
            'reflection_horizontal': reflect_h,
            'reflection_vertical': reflect_v
        }
        
        return params
    
    def render_main_content(self, params):
        """Render main content area"""
        if self.transformer.image is None:
            self.render_welcome_screen()
            return
        
        # Create transformation matrix
        transform_matrix = self.transformer.create_transformation_matrix(params)
        
        # Apply transformation
        try:
            transformed_image = self.transformer.apply_transformation(transform_matrix)
            
            # Display image comparison
            self.ui.display_image_comparison(
                self.transformer.image, 
                transformed_image
            )
            
            # Display transformation matrix
            self.ui.display_matrix(transform_matrix)
            
            # Active transformations indicator
            self.show_active_transformations(params)
            
        except Exception as e:
            st.error(f"Error applying transformation: {str(e)}")
    
    def render_welcome_screen(self):
        """Render welcome screen when no image is loaded"""
        st.info("👆 Silakan upload gambar atau aktifkan Demo Mode untuk memulai transformasi")
        
        # Display transformation info
        st.markdown("---")
        st.subheader("📚 Jenis Transformasi yang Tersedia:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔄 Translation**
            - Memindahkan objek
            - Parameter: X, Y (pixels)
            """)
        
        with col2:
            st.markdown("""
            **📏 Scaling**
            - Mengubah ukuran objek
            - Parameter: X, Y (scale factor)
            """)
        
        with col3:
            st.markdown("""
            **🔄 Rotation**
            - Memutar objek
            - Parameter: Sudut (derajat)
            """)
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("""
            **🔀 Shearing**
            - Mengubah bentuk objek
            - Parameter: X, Y (shear factor)
            """)
        
        with col5:
            st.markdown("""
            **🔁 Reflection**
            - Mencerminkan objek
            - Parameter: Horizontal, Vertical
            """)
    
    def show_active_transformations(self, params):
        """Show which transformations are active"""
        active_transforms = []
        
        if params.get('translation_x', 0) != 0 or params.get('translation_y', 0) != 0:
            active_transforms.append("Translation")
        
        if params.get('scaling_x', 1) != 1 or params.get('scaling_y', 1) != 1:
            active_transforms.append("Scaling")
        
        if params.get('rotation', 0) != 0:
            active_transforms.append("Rotation")
        
        if params.get('shearing_x', 0) != 0 or params.get('shearing_y', 0) != 0:
            active_transforms.append("Shearing")
        
        if params.get('reflection_horizontal', False) or params.get('reflection_vertical', False):
            active_transforms.append("Reflection")
        
        if active_transforms:
            st.info(f"🔧 Active Transformations: {', '.join(active_transforms)}")
        else:
            st.info("ℹ️ No active transformations")
    
    def run(self):
        """Run the main application"""
        self.setup_page_config()
        self.render_header()
        
        # Render sidebar and get parameters
        params = self.render_sidebar()
        
        # Render main content
        self.render_main_content(params)

# ============================================================================
# DEMO MODE
# ============================================================================

def run_demo():
    """Run demo mode with all transformations"""
    print("🔄 Matrix Transformation Studio - Demo Mode")
    print("=" * 50)
    
    transformer = MatrixTransformer()
    ui = UIComponents()
    
    # Create sample image
    print("📷 Creating sample image...")
    sample_image = ui.create_sample_image()
    transformer.load_image(sample_image)
    
    # Define demo transformations
    demo_transformations = [
        {
            'name': 'Translation (100, 50)',
            'params': {
                'translation_x': 100, 'translation_y': 50,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            }
        },
        {
            'name': 'Scale (1.5x, 0.8x)',
            'params': {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1.5, 'scaling_y': 0.8,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            }
        },
        {
            'name': 'Rotation 45°',
            'params': {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 45,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            }
        },
        {
            'name': 'Shear X=0.3',
            'params': {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 0,
                'shearing_x': 0.3, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            }
        },
        {
            'name': 'Reflection Horizontal',
            'params': {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': True, 'reflection_vertical': False
            }
        },
        {
            'name': 'Complex Transform',
            'params': {
                'translation_x': 50, 'translation_y': 30,
                'scaling_x': 1.2, 'scaling_y': 0.9,
                'rotation': 30,
                'shearing_x': 0.2, 'shearing_y': 0.1,
                'reflection_horizontal': False, 'reflection_vertical': False
            }
        }
    ]
    
    # Create output directory
    output_dir = "demo_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Apply each transformation
    for i, transform in enumerate(demo_transformations):
        print(f"🔧 Applying: {transform['name']}")
        
        # Create transformation matrix
        matrix = transformer.create_transformation_matrix(transform['params'])
        
        print(f"   Matrix:")
        for row in matrix:
            print(f"   [{row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f}]")
        
        # Apply transformation
        transformed_image = transformer.apply_transformation(matrix)
        
        # Save comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(transformer.image)
        ax1.set_title("Original")
        ax1.axis('off')
        
        ax2.imshow(transformed_image)
        ax2.set_title(f"Transformed: {transform['name']}")
        ax2.axis('off')
        
        filename = f"{output_dir}/transform_{i+1}_{transform['name'].replace(' ', '_').replace('°', 'deg')}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {filename}")
        print()
    
    print("🎉 Demo completed! Check the 'demo_output' folder for results.")
    print("📊 To run the full app: streamlit run matrix_transformation_app.py")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Matrix Transformation Studio')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--version', action='version', version=f'Matrix Transformation Studio {Config.VERSION}')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        # Run Streamlit app
        app = MatrixTransformationApp()
        app.run()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
