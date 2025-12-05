#!/usr/bin/env python3
"""
🔄 Matrix Transformation Studio - Simple Version
Image Processing dengan Matrix Transformations
"""

try:
    import streamlit as st
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import io
    import base64
    import sys
    import os
    import argparse
    from typing import Dict, Any
    
    # Fallback untuk OpenCV
    try:
        import cv2
        OPENCV_AVAILABLE = True
    except ImportError:
        OPENCV_AVAILABLE = False
        print("⚠️ OpenCV not available, using PIL fallback")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Please install: pip install streamlit numpy matplotlib pillow")
    sys.exit(1)

class SimpleMatrixTransformer:
    """Simplified version without OpenCV dependency"""
    
    def __init__(self):
        self.image = None
        self.transformed_image = None
    
    def load_image(self, image_source) -> bool:
        """Load image using PIL"""
        try:
            if hasattr(image_source, 'read'):  # UploadedFile
                self.image = Image.open(image_source).convert('RGB')
            elif isinstance(image_source, str):  # File path
                self.image = Image.open(image_source).convert('RGB')
            elif isinstance(image_source, Image.Image):  # PIL Image
                self.image = image_source.convert('RGB')
            else:
                raise ValueError("Unsupported image source type")
            
            return True
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return False
    
    def create_transformation_matrix(self, params: Dict[str, Any]) -> np.ndarray:
        """Create 3x3 transformation matrix"""
        tx = params.get('translation_x', 0)
        ty = params.get('translation_y', 0)
        sx = params.get('scaling_x', 1)
        sy = params.get('scaling_y', 1)
        rotation = params.get('rotation', 0)
        shear_x = params.get('shearing_x', 0)
        shear_y = params.get('shearing_y', 0)
        reflect_h = params.get('reflection_horizontal', False)
        reflect_v = params.get('reflection_vertical', False)
        
        # Convert rotation to radians
        angle_rad = np.radians(rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Build transformation matrix
        matrix = np.eye(3)
        
        # Reflection
        if reflect_h:
            matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ matrix
        if reflect_v:
            matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ matrix
        
        # Scaling
        matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]]) @ matrix
        
        # Rotation
        matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]) @ matrix
        
        # Shearing
        matrix = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]]) @ matrix
        
        # Translation
        matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]) @ matrix
        
        return matrix
    
    def apply_transformation_pil(self, matrix: np.ndarray) -> Image.Image:
        """Apply transformation using PIL"""
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Get image size
        width, height = self.image.size
        
        # Create a larger canvas
        canvas_size = max(width, height) * 3
        canvas = Image.new('RGB', (canvas_size, canvas_size), 'white')
        
        # Calculate center offset
        center_x = canvas_size // 2
        center_y = canvas_size // 2
        
        # Paste original image in center
        paste_x = center_x - width // 2
        paste_y = center_y - height // 2
        canvas.paste(self.image, (paste_x, paste_y))
        
        # Apply transformations step by step
        transformed = canvas
        
        # Extract parameters from matrix
        tx = matrix[0, 2]
        ty = matrix[1, 2]
        
        # Apply translation
        if tx != 0 or ty != 0:
            new_size = (canvas_size + abs(int(tx)) * 2, canvas_size + abs(int(ty)) * 2)
            new_canvas = Image.new('RGB', new_size, 'white')
            new_x = new_size[0] // 2 - canvas_size // 2 + int(tx)
            new_y = new_size[1] // 2 - canvas_size // 2 + int(ty)
            new_canvas.paste(transformed, (new_x, new_y))
            transformed = new_canvas
        
        # Apply rotation
        if matrix[0, 1] != 0 or matrix[1, 0] != 0:  # Has rotation
            angle = np.arctan2(matrix[1, 0], matrix[0, 0]) * 180 / np.pi
            transformed = transformed.rotate(-angle, expand=True, fillcolor='white')
        
        # Apply scaling
        if abs(matrix[0, 0]) != 1 or abs(matrix[1, 1]) != 1:
            scale_x = abs(matrix[0, 0])
            scale_y = abs(matrix[1, 1])
            new_width = int(transformed.width * scale_x)
            new_height = int(transformed.height * scale_y)
            transformed = transformed.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crop to content
        bbox = transformed.getbbox()
        if bbox:
            transformed = transformed.crop(bbox)
        
        self.transformed_image = transformed
        return transformed
    
    def apply_transformation(self, matrix: np.ndarray) -> Image.Image:
        """Apply transformation using available backend"""
        if OPENCV_AVAILABLE:
            return self.apply_transformation_opencv(matrix)
        else:
            return self.apply_transformation_pil(matrix)
    
    def apply_transformation_opencv(self, matrix: np.ndarray) -> Image.Image:
        """Apply transformation using OpenCV (if available)"""
        # Convert PIL to OpenCV format
        img_array = np.array(self.image)
        
        height, width = img_array.shape[:2]
        
        # Get corners
        corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]).T
        
        # Transform corners
        transformed_corners = matrix @ corners
        transformed_corners = transformed_corners[:2] / transformed_corners[2]
        
        # Calculate new bounds
        min_x, min_y = np.min(transformed_corners, axis=1)
        max_x, max_y = np.max(transformed_corners, axis=1)
        
        # New canvas size
        padding = 50
        new_width = int(max_x - min_x + 2 * padding)
        new_height = int(max_y - min_y + 2 * padding)
        
        # Apply transformation
        cv_matrix = matrix[:2, :]
        cv_matrix[0, 2] -= min_x - padding
        cv_matrix[1, 2] -= min_y - padding
        
        transformed = cv2.warpAffine(img_array, cv_matrix, (new_width, new_height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
        
        # Convert back to PIL
        self.transformed_image = Image.fromarray(transformed)
        return self.transformed_image
    
    def get_preset_transformations(self) -> Dict[str, Dict[str, Any]]:
        """Get preset transformations"""
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
            "Scale 2x": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 2, 'scaling_y': 2,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            }
        }

def create_sample_image():
    """Create sample image for demo"""
    img = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw shapes
    draw.rectangle([50, 50, 150, 150], fill='red')
    draw.ellipse([250, 100, 350, 200], fill='green')
    draw.line([(0, 0), (400, 300)], fill='blue', width=3)
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((150, 250), "DEMO", fill='black', font=font, anchor='mm')
    
    return img

def display_matrix(matrix: np.ndarray, title: str = "Matriks Transformasi"):
    """Display transformation matrix"""
    st.subheader(title)
    
    matrix_str = ""
    for i in range(matrix.shape[0]):
        row_str = " | ".join([f"{matrix[i,j]:8.3f}" for j in range(matrix.shape[1])])
        matrix_str += f"[ {row_str} ]\n"
    
    st.code(matrix_str, language='text')

def main():
    """Main application"""
    st.set_page_config(
        page_title="Matrix Transformation Studio",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Matrix Transformation Studio")
    st.markdown("Image Processing dengan Matrix Transformations")
    
    # Initialize transformer
    transformer = SimpleMatrixTransformer()
    
    # Sidebar
    st.sidebar.header("🎛️ Kontrol Transformasi")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Gambar",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    # Demo mode
    demo_mode = st.sidebar.checkbox("🎨 Demo Mode")
    
    if demo_mode:
        sample_img = create_sample_image()
        transformer.load_image(sample_img)
        st.sidebar.success("✅ Demo image loaded!")
    elif uploaded_file is not None:
        if transformer.load_image(uploaded_file):
            st.sidebar.success("✅ Image loaded!")
    
    # Only show controls if image is loaded
    if transformer.image is not None:
        # Transformation controls
        st.sidebar.subheader("📐 Parameter Transformasi")
        
        # Translation
        st.sidebar.markdown("**🔄 Translation**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            tx = st.slider("X", -200, 200, 0)
        with col2:
            ty = st.slider("Y", -200, 200, 0)
        
        # Scaling
        st.sidebar.markdown("**📏 Scaling**")
        col3, col4 = st.sidebar.columns(2)
        with col3:
            sx = st.slider("X", 0.1, 3.0, 1.0, 0.1)
        with col4:
            sy = st.slider("Y", 0.1, 3.0, 1.0, 0.1)
        
        # Rotation
        st.sidebar.markdown("**🔄 Rotation**")
        rotation = st.slider("Sudut", -180, 180, 0)
        
        # Reflection
        st.sidebar.markdown("**🔁 Reflection**")
        col5, col6 = st.sidebar.columns(2)
        with col5:
            reflect_h = st.checkbox("Horizontal")
        with col6:
            reflect_v = st.checkbox("Vertical")
        
        # Presets
        st.sidebar.subheader("⚡ Presets")
        presets = transformer.get_preset_transformations()
        selected_preset = st.sidebar.selectbox("Pilih Preset", ["None"] + list(presets.keys()))
        
        if selected_preset != "None":
            preset = presets[selected_preset]
            tx = preset['translation_x']
            ty = preset['translation_y']
            sx = preset['scaling_x']
            sy = preset['scaling_y']
            rotation = preset['rotation']
            reflect_h = preset['reflection_horizontal']
            reflect_v = preset['reflection_vertical']
        
        # Create parameters
        params = {
            'translation_x': tx,
            'translation_y': ty,
            'scaling_x': sx,
            'scaling_y': sy,
            'rotation': rotation,
            'shearing_x': 0,
            'shearing_y': 0,
            'reflection_horizontal': reflect_h,
            'reflection_vertical': reflect_v
        }
        
        # Create and apply transformation
        matrix = transformer.create_transformation_matrix(params)
        transformed = transformer.apply_transformation(matrix)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original")
            st.image(transformer.image, use_column_width=True)
        
        with col2:
            st.subheader("✨ Transformed")
            st.image(transformed, use_column_width=True)
            
            # Download button
            if st.button("💾 Download"):
                img_buffer = io.BytesIO()
                transformed.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                st.download_button(
                    label="📥 Download PNG",
                    data=img_buffer,
                    file_name="transformed_image.png",
                    mime="image/png"
                )
        
        # Display matrix
        display_matrix(matrix)
        
        # Active transformations
        active = []
        if tx != 0 or ty != 0:
            active.append("Translation")
        if sx != 1 or sy != 1:
            active.append("Scaling")
        if rotation != 0:
            active.append("Rotation")
        if reflect_h or reflect_v:
            active.append("Reflection")
        
        if active:
            st.info(f"🔧 Active: {', '.join(active)}")
    
    else:
        st.info("👆 Upload image or enable Demo Mode")

if __name__ == "__main__":
    main()
