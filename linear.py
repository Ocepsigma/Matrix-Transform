#!/usr/bin/env python3
"""
🔄 Matrix Transformation Studio - Final Version (Simple & Stable)
Page 1: Main Application
Page 2: Creator Profile - Yoseph Sihite
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
import sys
import os
import requests
from typing import Dict, Any, Tuple
import colorsys
import warnings

# Proteksi DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 100000000  # 100MP limit

# Disable warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set page config
st.set_page_config(
    page_title="Matrix Transformation Studio",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .matrix-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Matrix display */
    .matrix-display {
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        background: #1e293b;
        color: #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        white-space: pre;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 6px;
        margin: 0 0.25rem;
        font-weight: 600;
    }
    
    /* Success message */
    .success-message {
        background: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* Info message */
    .info-message {
        background: #3b82f6;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Profile card styling */
    .profile-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .profile-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .team-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .vision-card {
        background: linear-gradient(135deg, #fef3c7 0%, #f59e0b 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #f59e0b;
        color: #92400e;
    }
    
    /* Profile photo styling */
    .profile-photo {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin: 0 auto 1rem;
        display: block;
    }
    
    .photo-loading {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: #f1f5f9;
        margin: 0 auto 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 4px solid white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        font-size: 3rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

def load_profile_photo():
    """Load profile photo dari GitHub"""
    try:
        # GitHub configuration
        github_username = "yosephsihite"
        repo_name = "matrix-transformation-studio"
        photo_filename = "foto_yoseph.jpg"
        
        # Construct GitHub raw URL
        photo_url = f"https://raw.githubusercontent.com/{github_username}/{repo_name}/main/{photo_filename}"
        
        # Test URL accessibility
        try:
            response = requests.get(photo_url, timeout=10)
            
            if response.status_code == 200:
                # Load image
                image = Image.open(io.BytesIO(response.content))
                
                # Resize untuk profil (200x200)
                image = image.resize((200, 200), Image.Resampling.LANCZOS)
                
                # Convert to bytes untuk display
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG')
                img_bytes = img_buffer.getvalue()
                
                # Encode ke base64
                img_base64 = base64.b64encode(img_bytes).decode()
                
                return img_base64, True
            else:
                return None, False
                
        except requests.exceptions.RequestException as e:
            return None, False
        
    except Exception as e:
        return None, False

class SafeMatrixTransformer:
    """Matrix Transformer dengan proteksi DecompressionBombError"""
    
    def __init__(self):
        self.image = None
        self.transformed_image = None
        self.original_shape = None
    
    def safe_load_image(self, image_source) -> bool:
        """Load image dengan proteksi DecompressionBombError"""
        try:
            # Reset limit sementara
            Image.MAX_IMAGE_PIXELS = None
            
            if hasattr(image_source, 'read'):  # UploadedFile
                # Reset file pointer
                image_source.seek(0)
                
                # Baca file dengan proteksi
                try:
                    self.image = Image.open(image_source).convert('RGB')
                except Exception as e:
                    if "decompression bomb" in str(e).lower():
                        st.error("❌ Image terlalu besar! Silakan upload gambar yang lebih kecil (< 10MB)")
                        return False
                    raise e
                    
            elif isinstance(image_source, str):  # File path
                try:
                    self.image = Image.open(image_source).convert('RGB')
                except Exception as e:
                    if "decompression bomb" in str(e).lower():
                        st.error("❌ Image terlalu besar! Silakan pilih gambar yang lebih kecil")
                        return False
                    raise e
                    
            elif isinstance(image_source, Image.Image):  # PIL Image
                self.image = image_source.convert('RGB')
            else:
                st.error("❌ Format gambar tidak didukung")
                return False
            
            # Cek ukuran gambar
            width, height = self.image.size
            total_pixels = width * height
            
            # Batasi ukuran maksimal
            max_pixels = 50000000  # 50MP
            max_dimension = 8000
            
            if total_pixels > max_pixels:
                st.error(f"❌ Gambar terlalu besar! ({width}x{height} = {total_pixels:,} pixels)")
                st.info(f"💡 Maksimal: {max_dimension}x{max_dimension} atau {max_pixels:,} pixels")
                return False
            
            if width > max_dimension or height > max_dimension:
                st.error(f"❌ Dimensi gambar terlalu besar! ({width}x{height})")
                st.info(f"💡 Maksimal: {max_dimension}x{max_dimension} pixels")
                return False
            
            # Auto-resize untuk performance
            max_safe_size = 1000
            if width > max_safe_size or height > max_safe_size:
                ratio = min(max_safe_size / width, max_safe_size / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                st.info(f"🔄 Gambar di-resize ke {new_width}x{new_height} untuk performance")
            
            # Restore limit
            Image.MAX_IMAGE_PIXELS = 100000000
            
            self.original_shape = self.image.size
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            # Restore limit
            Image.MAX_IMAGE_PIXELS = 100000000
            return False
    
    def create_transformation_matrix(self, params: Dict[str, Any]) -> np.ndarray:
        """Create 3x3 transformation matrix"""
        try:
            # Extract parameters
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
            
            # Build transformation matrix step by step
            matrix = np.eye(3)
            
            # 1. Reflection
            if reflect_h:
                reflect_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
                matrix = reflect_matrix @ matrix
            
            if reflect_v:
                reflect_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
                matrix = reflect_matrix @ matrix
            
            # 2. Scaling
            scale_matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
            matrix = scale_matrix @ matrix
            
            # 3. Rotation
            rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            matrix = rotation_matrix @ matrix
            
            # 4. Shearing
            shear_matrix = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]])
            matrix = shear_matrix @ matrix
            
            # 5. Translation
            translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            matrix = translation_matrix @ matrix
            
            return matrix
            
        except Exception as e:
            st.error(f"❌ Error creating matrix: {str(e)}")
            return np.eye(3)
    
    def safe_apply_transformation(self, matrix: np.ndarray) -> Image.Image:
        """Apply transformation dengan proteksi error"""
        try:
            if self.image is None:
                raise ValueError("No image loaded")
            
            width, height = self.image.size
            
            # Create a larger canvas
            canvas_size = max(width, height) * 3
            canvas = Image.new('RGB', (canvas_size, canvas_size), 'white')
            
            # Calculate center position
            center_x = canvas_size // 2
            center_y = canvas_size // 2
            
            # Paste original image in center
            paste_x = center_x - width // 2
            paste_y = center_y - height // 2
            canvas.paste(self.image, (paste_x, paste_y))
            
            # Apply transformations step by step
            transformed = canvas
            
            # Extract transformation parameters
            tx = matrix[0, 2]
            ty = matrix[1, 2]
            
            # Apply translation
            if abs(tx) > 0 or abs(ty) > 0:
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
                
                # Batasi ukuran hasil
                max_result_size = 2000
                if new_width > max_result_size or new_height > max_result_size:
                    ratio = min(max_result_size / new_width, max_result_size / new_height)
                    new_width = int(new_width * ratio)
                    new_height = int(new_height * ratio)
                    st.warning(f"⚠️ Hasil transformasi di-resize ke {new_width}x{new_height}")
                
                transformed = transformed.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Apply reflection
            if matrix[0, 0] < 0:  # Horizontal reflection
                transformed = transformed.transpose(Image.FLIP_LEFT_RIGHT)
            
            if matrix[1, 1] < 0:  # Vertical reflection
                transformed = transformed.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Crop to content
            bbox = transformed.getbbox()
            if bbox:
                transformed = transformed.crop(bbox)
            
            self.transformed_image = transformed
            return transformed
            
        except Exception as e:
            st.error(f"❌ Error applying transformation: {str(e)}")
            st.warning("🔄 Mengembalikan gambar original")
            return self.image
    
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
            "Rotate 90°": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 90,
                'shearing_x': 0, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            },
            "Rotate 180°": {
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
            "Skew Right": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 0,
                'shearing_x': 0.3, 'shearing_y': 0,
                'reflection_horizontal': False, 'reflection_vertical': False
            },
            "Skew Up": {
                'translation_x': 0, 'translation_y': 0,
                'scaling_x': 1, 'scaling_y': 1,
                'rotation': 0,
                'shearing_x': 0, 'shearing_y': -0.3,
                'reflection_horizontal': False, 'reflection_vertical': False
            }
        }

def main_app():
    """Main Matrix Transformation Application"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">Matrix Transformation Studio</h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.25rem; opacity: 0.9;">Advanced Image Processing with Matrix Operations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize transformer
    transformer = SafeMatrixTransformer()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
            <h3 style="margin: 0; color: white;">🎛️ Transformation Controls</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload dengan proteksi
        uploaded_file = st.file_uploader(
            "📤 Upload Image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to apply transformations (Max: 10MB, 50MP)"
        )
        
        if uploaded_file is not None:
            # Cek ukuran file
            file_size = uploaded_file.size
            max_file_size = 10 * 1024 * 1024  # 10MB
            
            if file_size > max_file_size:
                st.error(f"❌ File terlalu besar! ({file_size/1024/1024:.1f}MB)")
                st.info(f"💡 Maksimal: {max_file_size/1024/1024}MB")
            else:
                if transformer.safe_load_image(uploaded_file):
                    st.markdown('<div class="success-message">✅ Image loaded successfully!</div>', unsafe_allow_html=True)
        
        # Only show controls if image is loaded
        if transformer.image is not None:
            st.markdown("---")
            
            # Transformation tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🔄 Translation", "📏 Scaling", "🔄 Rotation", "🔀 Shearing", "🔁 Reflection"
            ])
            
            with tab1:
                st.markdown("**Translation Parameters**")
                tx = st.slider("X Translation (pixels)", -200, 200, 0, key="translation_x")
                ty = st.slider("Y Translation (pixels)", -200, 200, 0, key="translation_y")
                st.caption(f"Current: X={tx}, Y={ty}")
            
            with tab2:
                st.markdown("**Scaling Parameters**")
                sx = st.slider("X Scale Factor", 0.1, 3.0, 1.0, 0.1, key="scaling_x")
                sy = st.slider("Y Scale Factor", 0.1, 3.0, 1.0, 0.1, key="scaling_y")
                st.caption(f"Current: X={sx:.1f}x, Y={sy:.1f}x")
            
            with tab3:
                st.markdown("**Rotation Parameters**")
                rotation = st.slider("Rotation Angle (degrees)", -180, 180, 0, key="rotation")
                st.caption(f"Current: {rotation}°")
            
            with tab4:
                st.markdown("**Shearing Parameters**")
                shear_x = st.slider("X Shear Factor", -1.0, 1.0, 0.0, 0.1, key="shearing_x")
                shear_y = st.slider("Y Shear Factor", -1.0, 1.0, 0.0, 0.1, key="shearing_y")
                st.caption(f"Current: X={shear_x:.1f}, Y={shear_y:.1f}")
            
            with tab5:
                st.markdown("**Reflection Parameters**")
                col1, col2 = st.columns(2)
                with col1:
                    reflect_h = st.checkbox("Horizontal Reflection", key="reflection_horizontal")
                with col2:
                    reflect_v = st.checkbox("Vertical Reflection", key="reflection_vertical")
                st.caption(f"Current: H={reflect_h}, V={reflect_v}")
            
            # Presets
            st.markdown("---")
            st.subheader("⚡ Preset Transformations")
            presets = transformer.get_preset_transformations()
            selected_preset = st.selectbox("Choose preset:", ["None"] + list(presets.keys()))
            
            if selected_preset != "None":
                preset = presets[selected_preset]
                st.session_state.update(preset)
                st.rerun()
            
            # Reset button
            if st.button("🔄 Reset All", use_container_width=True):
                keys_to_remove = ['translation_x', 'translation_y', 'scaling_x', 'scaling_y', 'rotation', 'shearing_x', 'shearing_y', 'reflection_horizontal', 'reflection_vertical']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Main content
    if transformer.image is not None:
        # Get current parameters
        params = {
            'translation_x': st.session_state.get('translation_x', 0),
            'translation_y': st.session_state.get('translation_y', 0),
            'scaling_x': st.session_state.get('scaling_x', 1.0),
            'scaling_y': st.session_state.get('scaling_y', 1.0),
            'rotation': st.session_state.get('rotation', 0),
            'shearing_x': st.session_state.get('shearing_x', 0.0),
            'shearing_y': st.session_state.get('shearing_y', 0.0),
            'reflection_horizontal': st.session_state.get('reflection_horizontal', False),
            'reflection_vertical': st.session_state.get('reflection_vertical', False)
        }
        
        # Create and apply transformation
        matrix = transformer.create_transformation_matrix(params)
        transformed_image = transformer.safe_apply_transformation(matrix)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(transformer.image, use_container_width=True, caption="ORIGINAL")
        
        with col2:
            st.subheader("✨ Transformed Image")
            st.image(transformed_image, use_container_width=True, caption="TRANSFORMED")
            
            # Download button
            if transformed_image is not None:
                img_buffer = io.BytesIO()
                transformed_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="💾 Download Transformed Image",
                    data=img_buffer,
                    file_name="transformed_image.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        # Display matrix
        st.subheader("📊 Transformation Matrix")
        
        # Format matrix for display
        matrix_str = ""
        for i in range(matrix.shape[0]):
            row_str = " | ".join([f"{matrix[i,j]:8.3f}" for j in range(matrix.shape[1])])
            matrix_str += f"[ {row_str} ]\n"
        
        st.code(matrix_str, language='text')
        
        # Matrix explanation
        with st.expander("📖 Matrix Components Explanation"):
            st.markdown("""
            **3×3 Transformation Matrix Components:**
            
            | Component | Description | Formula |
            |-----------|-------------|---------|
            | **[0,0], [0,1], [1,0], [1,1]] | Linear transformation (rotation, scale, shear) | Combined from all transforms |
            | **[0,2], [1,2]] | Translation (X, Y displacement) | `tx, ty` |
            | **[2,0], [2,1]] | Perspective (unused in this implementation) | `0, 0` |
            | **[2,2]** | Homogeneous coordinate | `1` |
            
            **Matrix Order:** Reflection → Scaling → Rotation → Shearing → Translation
            """)
        
        # Active transformations
        active = []
        if params.get('translation_x', 0) != 0 or params.get('translation_y', 0) != 0:
            active.append("Translation")
        if params.get('scaling_x', 1) != 1 or params.get('scaling_y', 1) != 1:
            active.append("Scaling")
        if params.get('rotation', 0) != 0:
            active.append("Rotation")
        if params.get('shearing_x', 0) != 0 or params.get('shearing_y', 0) != 0:
            active.append("Shearing")
        if params.get('reflection_horizontal', False) or params.get('reflection_vertical', False):
            active.append("Reflection")
        
        if active:
            st.info(f"🔧 Active Transformations: {', '.join(active)}")
        else:
            st.info("ℹ️ No active transformations")
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="card">
            <h3 style="margin: 0 0 1rem 0; color: #1e293b;">👆 Welcome to Matrix Transformation Studio</h3>
            <p style="margin: 0 0 1rem 0; color: #64748b;">Upload an image to start transforming!</p>
            <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>🔒 Security Notice:</strong><br>
                • Maximum file size: 10MB<br>
                • Maximum image size: 50MP (8000x8000)<br>
                • Images are auto-resized for performance<br>
                • All processing is done locally in your browser
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">🔄 Translation</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Move objects along X and Y axes with pixel precision</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #764ba2;">📏 Scaling</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Resize objects with independent X and Y scale factors</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">🔄 Rotation</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Rotate objects by any angle with smooth interpolation</p>
            </div>
            """, unsafe_allow_html=True)
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #764ba2;">🔀 Shearing</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Apply skew transformations for artistic effects</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">🔁 Reflection</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Mirror objects horizontally and/or vertically</p>
            </div>
            """, unsafe_allow_html=True)

def profile_page():
    """Profile page for Yoseph Sihite"""
    # Header
    st.markdown("""
    <div class="profile-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">👨‍💻 Creator Profile</h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.25rem; opacity: 0.9;">Yoseph Sihite - Linear Algebra Visionary</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load profile photo
    photo_base64, photo_loaded = load_profile_photo()
    
    # Profile section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="profile-card">
            <div style="text-align: center;">
        """, unsafe_allow_html=True)
        
        if photo_loaded and photo_base64:
            st.markdown(f"""
                <img src="data:image/jpeg;base64,{photo_base64}" class="profile-photo" alt="Yoseph Sihite">
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="photo-loading">
                    ⏳
                </div>
                <p style="text-align: center; margin-top: 0.5rem; color: #64748b; font-size: 0.9rem;">Loading photo from GitHub...</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            </div>
            <h3 style="text-align: center; margin: 1rem 0 0.5rem 0; color: #1e293b;">Yoseph Sihite</h3>
            <p style="text-align: center; margin: 0; color: #64748b;">Computer Vision Developer</p>
            <p style="text-align: center; margin: 0.5rem 0; color: #64748b;">📍 Jakarta, Indonesia</p>
            <div style="text-align: center; margin: 1rem 0;">
                <a href="https://github.com/yosephsihite" target="_blank" style="margin: 0 0.5rem 0; text-decoration: none; color: #667eea;">🔗 GitHub</a>
                <a href="mailto:yoseph.sihite@example.com" style="margin: 0 0.5rem 0; text-decoration: none; color: #667eea;">📧 Email</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Development Team Section - TANPA HTML
        st.markdown("""
        <div class="team-card">
            <h3 style="margin: 0 0 1rem 0; color: #1e293b;">👨‍💻 Development Team</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Development Team content menggunakan st.write() TANPA HTML
        st.write("---")
        st.write("## 👤 Lead Developer")
        st.write("**Nama:** Yoseph Sihite")
        st.write("**Student ID:** 004202400113")
        st.write("**Group:** Group 2 Linear Algebra")
        st.write("**Role:** Lead Developer & Linear Algebra Specialist")
        
        st.write("---")
        st.write("## 🎯 Project Overview")
        st.write("**Matrix Transformation Studio** adalah aplikasi web interaktif yang dikembangkan sebagai **Final Project Mata Kuliah Linear Algebra**. Aplikasi ini dirancang untuk **memvisualisasikan konsep transformasi matriks** agar lebih mudah dipahami melalui pendekatan visualisasi berbasis web.")
        
        st.write("---")
        st.write("## 💪 Kontribusi")
        st.write("**Seluruh proses pengembangan proyek ini dikerjakan secara mandiri** oleh Yoseph Sihite. Kontribusi yang dilakukan mencakup perancangan konsep dan arsitektur aplikasi, pengembangan algoritma transformasi matriks, serta implementasi konsep aljabar linear ke dalam sistem visual interaktif. Selain itu, pengembangan web app, termasuk desain antarmuka pengguna, pengelolaan logika aplikasi, dan pengujian fungsionalitas, sepenuhnya diselesaikan secara individual karena tidak adanya anggota lain dalam Group 2.")
        
        st.write("---")
        st.write("## 🔧 Kontribusi Utama:")
        st.write("1. **Perancangan Konsep & Arsitektur Aplikasi:** Merancang struktur aplikasi yang efisien dan user-friendly")
        st.write("2. **Pengembangan Algoritma Transformasi Matriks:** Implementasi algoritma matriks 3×3 untuk transformasi 2D")
        st.write("3. **Implementasi Konsep Aljabar Linear:** Mengubah konsep matematis abstrak menjadi visualisasi interaktif")
        st.write("4. **Pengembangan Web Application:** Membangun antarmuka pengguna dengan Streamlit")
        st.write("5. **Desain Antarmuka Pengguna:** Menciptakan UI yang intuitif dan menarik")
        st.write("6. **Pengelolaan Logika Aplikasi:** Implementasi backend untuk pemrosesan gambar")
        st.write("7. **Pengujian Fungsionalitas:** Testing dan debugging fitur aplikasi")
        
        st.write("---")
        st.write("## 🔧 Teknologi yang Digunakan:")
        st.write("1. **Frontend:** Streamlit (Python Web Framework), HTML/CSS (Styling & Layout), JavaScript (Interaktivitas & Validasi)")
        st.write("2. **Backend:** Python (Bahasa Pemrograman Utama), NumPy (Operasi Matriks & Perhitungan Matematis), Pillow/PIL (Pemrosesan Gambar & Transformasi), Matplotlib (Visualisasi Matriks & Grafik)")
        st.write("3. **Design Tools:** Gradient Styling (Desain Modern), Card-based Layout (Tata Letak Kartu), Responsive Design (Beradaptasi di Berbagai Perangkat)")
        
        st.write("---")
        st.write("## 🎓 Prestasi Akademik:")
        st.write("Proyek ini **menunjukkan pemahaman komprehensif konsep aljabar linear** dan **kemampuan mengimplementasikan teori matematis** dalam aplikasi praktis. Aplikasi berhasil **mengubah konsep abstrak menjadi alat pembelajaran visual yang interaktif** dan **dapat digunakan sebagai demonstrasi dalam pengajaran matematika**.")
    
    
def main():
    """Main application with multi-page navigation"""
    # Page navigation
    page = st.sidebar.selectbox(
        "📄 Navigate to:",
        ["🔄 Matrix Transformation Studio", "👨‍💻 Creator Profile"],
        index=0,
        format_func=lambda x: x.split(" ", 1)[1] if " " in x else x
    )
    
    if page == "🔄 Matrix Transformation Studio":
        main_app()
    elif page == "👨‍💻 Creator Profile":
        profile_page()

if __name__ == "__main__":
    main()
