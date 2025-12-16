#!/usr/bin/env python3
"""
🔄 Matrix Transformation Studio - Fixed Version with Image Processing
Page 1: Main Application - Matrix Transformations & Image Processing
Page 2: Creator Profile - Yoseph Sihite
✅ Fixed AttributeError for processed_image
✅ Proper image initialization and handling
✅ All image processing features: Blur, Sharpen, Background Removal
✅ Bilingual support (Indonesian/English)
✅ Error handling and safety checks
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageEnhance, ImageFilter
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

# Language translations dictionary
TRANSLATIONS = {
    'id': {
        # Navigation
        'nav_matrix_studio': '🔄 Studio Transformasi Matriks',
        'nav_creator_profile': '👨‍💻 Profil Pembuat',
        'navigate_to': '📄 Navigasi ke:',
        
        # Main App
        'app_title': 'Studio Transformasi Matriks',
        'app_subtitle': 'Pemrosesan Gambar Tingkat Lanjut dengan Operasi Matriks',
        'welcome_title': '👆 Selamat Datang di Studio Transformasi Matriks',
        'welcome_subtitle': 'Unggah gambar untuk mulai mentransformasi!',
        'security_notice': '🔒 Pemberitahuan Keamanan:',
        'security_points': [
            'Ukuran file maksimal: 10MB',
            'Ukuran gambar maksimal: 50MP (8000x8000)',
            'Gambar di-resize otomatis untuk performa',
            'Semua pemrosesan dilakukan secara lokal di browser Anda'
        ],
        
        # Controls
        'transformation_controls': '🎛️ Kontrol Transformasi',
        'image_processing': '🎨 Pemrosesan Gambar',
        'upload_image': '📤 Unggah Gambar',
        'upload_help': 'Unggah gambar untuk menerapkan transformasi (Maks: 10MB, 50MP)',
        'file_too_large': '❌ File terlalu besar!',
        'max_size': '💡 Maksimal:',
        'image_loaded': '✅ Gambar berhasil dimuat!',
        
        # Transformation types
        'translation': '🔄 Translasi',
        'scaling': '📏 Skala',
        'rotation': '🔄 Rotasi',
        'shearing': '🔀 Geser',
        'reflection': '🔁 Refleksi',
        
        # Image Processing
        'blur': '🌫 Blur',
        'sharpen': '🔍 Tajamkan',
        'background_removal': '🎨 Hapus Latar Belakang',
        'blur_intensity': 'Intensitas Blur',
        'sharpen_intensity': 'Intensitas Tajamkan',
        'background_tolerance': 'Toleransi Latar Belakang',
        'apply_processing': 'Terapkan Pemrosesan',
        'reset_processing': 'Reset Pemrosesan',
        
        # Parameters
        'translation_params': 'Parameter Translasi',
        'x_translation': 'Translasi X (piksel)',
        'y_translation': 'Translasi Y (piksel)',
        'scaling_params': 'Parameter Skala',
        'x_scale_factor': 'Faktor Skala X',
        'y_scale_factor': 'Faktor Skala Y',
        'rotation_params': 'Parameter Rotasi',
        'rotation_angle': 'Sudut Rotasi (derajat)',
        'shearing_params': 'Parameter Geser',
        'x_shear_factor': 'Faktor Geser X',
        'y_shear_factor': 'Faktor Geser Y',
        'reflection_params': 'Parameter Refleksi',
        'horizontal_reflection': 'Refleksi Horizontal',
        'vertical_reflection': 'Refleksi Vertikal',
        
        # Current values
        'current': 'Saat ini:',
        
        # Presets
        'preset_transformations': '⚡ Transformasi Preset',
        'choose_preset': 'Pilih preset:',
        'reset_all': '🔄 Reset Semua',
        
        # Display
        'original_image': '📷 Gambar Asli',
        'original_label': 'ASLI',
        'transformed_image': '✨ Gambar Transformasi',
        'transformed_label': 'TRANSFORMASI',
        'processed_image': '🎨 Gambar Diproses',
        'processed_label': 'DIPROSES',
        'download_image': '💾 Unduh Gambar Transformasi',
        'download_processed': '💾 Unduh Gambar Diproses',
        'transformation_matrix': '📊 Matriks Transformasi',
        'matrix_explanation': '📖 Penjelasan Komponen Matriks',
        'active_transformations': '🔧 Transformasi Aktif:',
        'no_active_transformations': 'ℹ️ Tidak ada transformasi aktif',
        
        # Features
        'translation_desc': 'Pindahkan objek sepanjang sumbu X dan Y dengan presisi piksel',
        'scaling_desc': 'Ubah ukuran objek dengan faktor skala X dan Y yang independen',
        'rotation_desc': 'Putar objek dengan sudut apa pun dengan interpolasi yang halus',
        'shearing_desc': 'Terapkan transformasi skew untuk efek artistik',
        'reflection_desc': 'Cerminkan objek secara horizontal dan/atau vertikal',
        'blur_desc': 'Menghaluskan gambar dengan efek blur',
        'sharpen_desc': 'Meningkatkan ketajaman gambar',
        'background_removal_desc': 'Menghapus latar belakang gambar',
        
        # Profile Page
        'creator_profile': '👨‍💻 Profil Pembuat',
        'profile_subtitle': 'Yoseph Sihite - Aljabar Linear',
        'photo_loading': '⏳',
        'loading_photo': 'Memuat foto dari GitHub...',
        'upload_profile_photo': '### Unggah Foto Profil Manual',
        'choose_profile_photo': 'Pilih foto profil',
        'photo_uploaded': '✅ Foto berhasil diunggah dan diproses!',
        'photo_error': '⚠️ Foto profil tidak dapat dimuat dari GitHub. Pastikan file foto_yoseph.jpg ada di repository Ocepsigma/Matrix-Transform/main/',
        'photo_url': '🔗 URL: https://raw.githubusercontent.com/Ocepsigma/Matrix-Transform/main/foto_yoseph.jpg',
        
        # Development Team
        'lead_developer': '## 👤 Pengembang Utama',
        'name': 'Nama:',
        'student_id': 'ID Mahasiswa:',
        'group': 'Grup:',
        'role': 'Peran:',
        
        # Project sections
        'project_overview': '## 🎯 Ikhtisar Proyek',
        'project_description': '**Studio Transformasi Matriks** adalah aplikasi web interaktif yang dikembangkan sebagai **Proyek Akhir Mata Kuliah Aljabar Linear**. Aplikasi ini dirancang untuk **memvisualisasikan konsep transformasi matriks** agar lebih mudah dipahami melalui pendekatan visualisasi berbasis web.',
        'contributions': '## 💪 Kontribusi',
        'contributions_description': '**Seluruh proses pengembangan proyek ini dikerjakan secara mandiri** oleh Yoseph Sihite. Kontribusi yang dilakukan mencakup perancangan konsep dan arsitektur aplikasi, pengembangan algoritma transformasi matriks, serta implementasi konsep aljabar linear ke dalam sistem visual interaktif. Selain itu, pengembangan web app, termasuk desain antarmuka pengguna, pengelolaan logika aplikasi, dan pengujian fungsionalitas, sepenuhnya diselesaikan secara individual karena tidak adanya anggota lain dalam Group 2.',
        
        # Language selector
        'select_language': '🌐 Pilih Bahasa',
        'indonesian': 'Bahasa Indonesia',
        'english': 'English'
    },
    'en': {
        # Navigation
        'nav_matrix_studio': '🔄 Matrix Transformation Studio',
        'nav_creator_profile': '👨‍💻 Creator Profile',
        'navigate_to': '📄 Navigate to:',
        
        # Main App
        'app_title': 'Matrix Transformation Studio',
        'app_subtitle': 'Advanced Image Processing with Matrix Operations',
        'welcome_title': '👆 Welcome to Matrix Transformation Studio',
        'welcome_subtitle': 'Upload an image to start transforming!',
        'security_notice': '🔒 Security Notice:',
        'security_points': [
            'Maximum file size: 10MB',
            'Maximum image size: 50MP (8000x8000)',
            'Images are auto-resized for performance',
            'All processing is done locally in your browser'
        ],
        
        # Controls
        'transformation_controls': '🎛️ Transformation Controls',
        'image_processing': '🎨 Image Processing',
        'upload_image': '📤 Upload Image',
        'upload_help': 'Upload an image to apply transformations (Max: 10MB, 50MP)',
        'file_too_large': '❌ File too large!',
        'max_size': '💡 Maximum:',
        'image_loaded': '✅ Image loaded successfully!',
        
        # Transformation types
        'translation': '🔄 Translation',
        'scaling': '📏 Scaling',
        'rotation': '🔄 Rotation',
        'shearing': '🔀 Shearing',
        'reflection': '🔁 Reflection',
        
        # Image Processing
        'blur': '🌫 Blur',
        'sharpen': '🔍 Sharpen',
        'background_removal': '🎨 Background Removal',
        'blur_intensity': 'Blur Intensity',
        'sharpen_intensity': 'Sharpen Intensity',
        'background_tolerance': 'Background Tolerance',
        'apply_processing': 'Apply Processing',
        'reset_processing': 'Reset Processing',
        
        # Parameters
        'translation_params': 'Translation Parameters',
        'x_translation': 'X Translation (pixels)',
        'y_translation': 'Y Translation (pixels)',
        'scaling_params': 'Scaling Parameters',
        'x_scale_factor': 'X Scale Factor',
        'y_scale_factor': 'Y Scale Factor',
        'rotation_params': 'Rotation Parameters',
        'rotation_angle': 'Rotation Angle (degrees)',
        'shearing_params': 'Shearing Parameters',
        'x_shear_factor': 'X Shear Factor',
        'y_shear_factor': 'Y Shear Factor',
        'reflection_params': 'Reflection Parameters',
        'horizontal_reflection': 'Horizontal Reflection',
        'vertical_reflection': 'Vertical Reflection',
        
        # Current values
        'current': 'Current:',
        
        # Presets
        'preset_transformations': '⚡ Preset Transformations',
        'choose_preset': 'Choose preset:',
        'reset_all': '🔄 Reset All',
        
        # Display
        'original_image': '📷 Original Image',
        'original_label': 'ORIGINAL',
        'transformed_image': '✨ Transformed Image',
        'transformed_label': 'TRANSFORMED',
        'processed_image': '🎨 Processed Image',
        'processed_label': 'PROCESSED',
        'download_image': '💾 Download Transformed Image',
        'download_processed': '💾 Download Processed Image',
        'transformation_matrix': '📊 Transformation Matrix',
        'matrix_explanation': '📖 Matrix Components Explanation',
        'active_transformations': '🔧 Active Transformations:',
        'no_active_transformations': 'ℹ️ No active transformations',
        
        # Features
        'translation_desc': 'Move objects along X and Y axes with pixel precision',
        'scaling_desc': 'Resize objects with independent X and Y scale factors',
        'rotation_desc': 'Rotate objects by any angle with smooth interpolation',
        'shearing_desc': 'Apply skew transformations for artistic effects',
        'reflection_desc': 'Mirror objects horizontally and/or vertically',
        'blur_desc': 'Blur the image with blur effect',
        'sharpen_desc': 'Increase image sharpness',
        'background_removal_desc': 'Remove background from image',
        
        # Profile Page
        'creator_profile': '👨‍💻 Creator Profile',
        'profile_subtitle': 'Yoseph Sihite - Linear Algebra',
        'photo_loading': '⏳',
        'loading_photo': 'Loading photo from GitHub...',
        'upload_profile_photo': '### Upload Profile Photo Manually',
        'choose_profile_photo': 'Choose profile photo',
        'photo_uploaded': '✅ Photo uploaded and processed successfully!',
        'photo_error': '⚠️ Profile photo could not be loaded from GitHub. Make sure foto_yoseph.jpg file exists in Ocepsigma/Matrix-Transform/main/ repository',
        'photo_url': '🔗 URL: https://raw.githubusercontent.com/Ocepsigma/Matrix-Transform/main/foto_yoseph.jpg',
        
        # Development Team
        'lead_developer': '## 👤 Lead Developer',
        'name': 'Name:',
        'student_id': 'Student ID:',
        'group': 'Group:',
        'role': 'Role:',
        
        # Project sections
        'project_overview': '## 🎯 Project Overview',
        'project_description': '**Matrix Transformation Studio** is an interactive web application developed as a **Final Project for Linear Algebra Course**. This application is designed to **visualize matrix transformation concepts** to make them easier to understand through web-based visualization approach.',
        'contributions': '## 💪 Contributions',
        'contributions_description': '**The entire development process of this project was done individually** by Yoseph Sihite. Contributions include concept design and application architecture, matrix transformation algorithm development, and implementation of linear algebra concepts into interactive visual systems. Additionally, web app development, including user interface design, application logic management, and functionality testing, was completed entirely individually due to the absence of other members in Group 2.',
        
        # Language selector
        'select_language': '🌐 Select Language',
        'indonesian': 'Bahasa Indonesia',
        'english': 'English'
    }
}

# Language selector function
def get_text(key):
    """Get translated text based on current language"""
    lang = st.session_state.get('language', 'id')
    return TRANSLATIONS[lang].get(key, key)

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
    
    /* Profile photo styling */
    .profile-photo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    .profile-photo {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #667eea;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'id'
if 'transformer' not in st.session_state:
    st.session_state.transformer = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Image Processing Functions
def apply_blur(image, intensity=1.0):
    """Apply blur effect to image"""
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply Gaussian blur with specified intensity
        radius = int(intensity * 5)  # Scale intensity to radius
        if radius < 1:
            radius = 1
        result = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Ensure result is RGB
        if result.mode != 'RGB':
            result = result.convert('RGB')
            
        return result
    except Exception as e:
        st.error(f"Error applying blur: {str(e)}")
        return image

def apply_sharpen(image, intensity=1.0):
    """Apply sharpen effect to image"""
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply sharpening with specified intensity
        factor = 1.0 + (intensity * 0.5)  # Scale intensity to factor
        result = image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(factor*150), threshold=3))
        
        # Ensure result is RGB
        if result.mode != 'RGB':
            result = result.convert('RGB')
            
        return result
    except Exception as e:
        st.error(f"Error applying sharpen: {str(e)}")
        return image

def remove_background(image, tolerance=30):
    """Remove background from image using simple color-based segmentation"""
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to numpy array
        img_array = np.array(image)
        
        # Create a mask for background removal
        # Simple approach: remove pixels that are close to the edge colors
        h, w = img_array.shape[:2]
        
        # Get edge colors (corners and edges)
        edge_colors = []
        edge_colors.append(img_array[0, 0])  # Top-left
        edge_colors.append(img_array[0, w-1])  # Top-right
        edge_colors.append(img_array[h-1, 0])  # Bottom-left
        edge_colors.append(img_array[h-1, w-1])  # Bottom-right
        
        # Calculate average edge color
        avg_edge_color = np.mean(edge_colors, axis=0)
        
        # Create mask based on color distance from edge color
        mask = np.zeros((h, w), dtype=bool)
        for i in range(h):
            for j in range(w):
                color_diff = np.abs(img_array[i, j] - avg_edge_color)
                if np.all(color_diff <= tolerance):
                    mask[i, j] = True
        
        # Apply mask to create transparent background
        result = img_array.copy()
        result[mask] = [255, 255, 255, 0]  # White with alpha=0 for background
        
        # Convert back to PIL Image
        result = Image.fromarray(result)
        
        # Ensure result is RGB
        if result.mode != 'RGB':
            result = result.convert('RGB')
            
        return result
    except Exception as e:
        st.error(f"Error removing background: {str(e)}")
        return image

class SafeMatrixTransformer:
    """Matrix Transformer dengan proteksi DecompressionBombError dan proper image handling"""
    
    def __init__(self):
        self.image = None
        self.transformed_image = None
        self.processed_image = None
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
                        lang = st.session_state.get('language', 'id')
                        if lang == 'id':
                            st.error("❌ Gambar terlalu besar! Silakan upload gambar yang lebih kecil (< 10MB)")
                        else:
                            st.error("❌ Image too large! Please upload a smaller image (< 10MB)")
                        return False
                    raise e
                    
            elif isinstance(image_source, str):  # File path
                try:
                    self.image = Image.open(image_source).convert('RGB')
                except Exception as e:
                    if "decompression bomb" in str(e).lower():
                        lang = st.session_state.get('language', 'id')
                        if lang == 'id':
                            st.error("❌ Gambar terlalu besar! Silakan pilih gambar yang lebih kecil")
                        else:
                            st.error("❌ Image too large! Please choose a smaller image")
                        return False
                    raise e
                    
            elif isinstance(image_source, Image.Image):  # PIL Image
                self.image = image_source.convert('RGB')
            else:
                lang = st.session_state.get('language', 'id')
                if lang == 'id':
                    st.error("❌ Format gambar tidak didukung")
                else:
                    st.error("❌ Unsupported image format")
                return False
            
            # Cek ukuran gambar
            width, height = self.image.size
            total_pixels = width * height
            
            if total_pixels > 50000000:  # 50MP limit
                lang = st.session_state.get('language', 'id')
                if lang == 'id':
                    st.error("❌ Resolusi gambar terlalu tinggi! Maksimal 50MP")
                else:
                    st.error("❌ Image resolution too high! Maximum 50MP")
                return False
            
            self.original_shape = self.image.size
            
            # Initialize processed_image to original image
            self.processed_image = self.image.copy()
            
            # Auto-resize untuk performance
            max_size = 2000
            if self.image.width > max_size or self.image.height > max_size:
                ratio = min(max_size/self.image.width, max_size/self.image.height)
                new_width = int(self.image.width * ratio)
                new_height = int(self.image.height * ratio)
                self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.processed_image = self.image.copy()
                self.original_shape = self.image.size
            
            # Restore limit
            Image.MAX_IMAGE_PIXELS = 100000000
            return True
            
        except Exception as e:
            lang = st.session_state.get('language', 'id')
            if lang == 'id':
                st.error(f"❌ Error loading image: {str(e)}")
            else:
                st.error(f"❌ Error loading image: {str(e)}")
            # Restore limit
            Image.MAX_IMAGE_PIXELS = 100000000
            
            # Ensure processed_image is None if loading fails
            self.processed_image = None
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
            st.error(f"Error creating transformation matrix: {str(e)}")
            return np.eye(3)
    
    def safe_apply_transformation(self, matrix: np.ndarray) -> Image.Image:
        """Apply transformation with error handling"""
        try:
            if self.image is None:
                raise ValueError("No image loaded")
            
            # Convert image to numpy array
            img_array = np.array(self.image)
            h, w = img_array.shape[:2]
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(x_coords.size)])
            
            # Apply transformation
            transformed_coords = matrix @ coords
            
            # Normalize homogeneous coordinates
            transformed_coords = transformed_coords[:2] / transformed_coords[2]
            
            # Reshape back to image shape
            new_x = transformed_coords[0].reshape(h, w)
            new_y = transformed_coords[1].reshape(h, w)
            
            # Create output image with white background
            result = np.ones_like(img_array) * 255
            
            # Find valid coordinates
            valid_mask = (
                (new_x >= 0) & (new_x < w) &
                (new_y >= 0) & (new_y < h)
            )
            
            # Map pixels
            result[valid_mask] = img_array[
                new_y[valid_mask].astype(int),
                new_x[valid_mask].astype(int)
            ]
            
            # Convert back to PIL Image
            self.transformed_image = Image.fromarray(result.astype(np.uint8))
            
            # Ensure transformed_image is RGB
            if self.transformed_image.mode != 'RGB':
                self.transformed_image = self.transformed_image.convert('RGB')
            
            return self.transformed_image
            
        except Exception as e:
            st.error(f"Error applying transformation: {str(e)}")
            return self.image

def main_app():
    """Main application page"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{get_text('app_title')}</h1>
        <p>{get_text('app_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown(f"### {get_text('upload_image')}")
    uploaded_file = st.file_uploader(
        get_text('upload_help'),
        type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
        help=get_text('upload_help')
    )
    
    if uploaded_file is not None:
        # Check file size
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
            st.error(f"{get_text('file_too_large')} {get_text('max_size')} 10MB")
            return
        
        # Load image
        transformer = SafeMatrixTransformer()
        
        if transformer.safe_load_image(uploaded_file):
            st.success(get_text('image_loaded'))
            st.session_state.transformer = transformer
            st.session_state.uploaded_file = uploaded_file
        else:
            st.error("Failed to load image")
            return
    
    # Continue with transformation if image is loaded
    if st.session_state.transformer is not None:
        transformer = st.session_state.transformer
        
        # Create tabs for different controls
        tab1, tab2, tab3 = st.tabs([
            get_text('transformation_controls'),
            get_text('image_processing'),
            get_text('preset_transformations')
        ])
        
        with tab1:
            st.markdown(f"### {get_text('transformation_controls')}")
            
            # Transformation parameters
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {get_text('translation_params')}")
                tx = st.slider(
                    f"{get_text('x_translation')}",
                    min_value=-200,
                    max_value=200,
                    value=0,
                    step=1,
                    key="translation_x"
                )
                ty = st.slider(
                    f"{get_text('y_translation')}",
                    min_value=-200,
                    max_value=200,
                    value=0,
                    step=1,
                    key="translation_y"
                )
                
                st.markdown(f"#### {get_text('scaling_params')}")
                sx = st.slider(
                    f"{get_text('x_scale_factor')}",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    key="scaling_x"
                )
                sy = st.slider(
                    f"{get_text('y_scale_factor')}",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    key="scaling_y"
                )
            
            with col2:
                st.markdown(f"#### {get_text('rotation_params')}")
                rotation = st.slider(
                    f"{get_text('rotation_angle')}",
                    min_value=-180,
                    max_value=180,
                    value=0,
                    step=1,
                    key="rotation"
                )
                
                st.markdown(f"#### {get_text('shearing_params')}")
                shear_x = st.slider(
                    f"{get_text('x_shear_factor')}",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    key="shearing_x"
                )
                shear_y = st.slider(
                    f"{get_text('y_shear_factor')}",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    key="shearing_y"
                )
                
                st.markdown(f"#### {get_text('reflection_params')}")
                reflect_h = st.checkbox(
                    f"{get_text('horizontal_reflection')}",
                    key="reflection_horizontal"
                )
                reflect_v = st.checkbox(
                    f"{get_text('vertical_reflection')}",
                    key="reflection_vertical"
                )
        
        with tab2:
            st.markdown(f"### {get_text('image_processing')}")
            
            # Image processing controls
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### {get_text('blur')}")
                blur_intensity = st.slider(
                    f"{get_text('blur_intensity')}",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.1,
                    key="blur_intensity",
                    help=get_text('blur_desc')
                )
                
                st.markdown(f"#### {get_text('sharpen')}")
                sharpen_intensity = st.slider(
                    f"{get_text('sharpen_intensity')}",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.1,
                    key="sharpen_intensity",
                    help=get_text('sharpen_desc')
                )
            
            with col2:
                st.markdown(f"#### {get_text('background_removal')}")
                apply_bg_removal = st.checkbox(
                    f"{get_text('background_removal')}",
                    key="apply_bg_removal",
                    help=get_text('background_removal_desc')
                )
                
                if apply_bg_removal:
                    bg_tolerance = st.slider(
                        f"{get_text('background_tolerance')}",
                        min_value=10,
                        max_value=100,
                        value=30,
                        step=5,
                        key="background_tolerance"
                    )
        
        with tab3:
            st.markdown(f"### {get_text('preset_transformations')}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Flip Horizontal"):
                    st.session_state.reflection_horizontal = not st.session_state.get('reflection_horizontal', False)
                
                if st.button("🔄 Flip Vertical"):
                    st.session_state.reflection_vertical = not st.session_state.get('reflection_vertical', False)
                
                if st.button("🔄 Rotate 90°"):
                    st.session_state.rotation = (st.session_state.get('rotation', 0) + 90) % 360
            
            with col2:
                if st.button("🔄 Rotate 180°"):
                    st.session_state.rotation = (st.session_state.get('rotation', 0) + 180) % 360
                
                if st.button("📏 Scale 2x"):
                    st.session_state.scaling_x = 2.0
                    st.session_state.scaling_y = 2.0
                
                if st.button("📏 Scale 0.5x"):
                    st.session_state.scaling_x = 0.5
                    st.session_state.scaling_y = 0.5
            
            with col3:
                if st.button("🔀 Shear Right"):
                    st.session_state.shearing_x = 0.3
                
                if st.button("🔀 Shear Up"):
                    st.session_state.shearing_y = -0.3
                
                if st.button(get_text('reset_all')):
                    # Reset all parameters
                    for key in st.session_state.keys():
                        if key.startswith(('translation_', 'scaling_', 'rotation', 'shearing_', 'reflection_', 'blur_', 'sharpen_', 'background_', 'apply_')):
                            if key in ['translation_x', 'translation_y']:
                                st.session_state[key] = 0
                            elif key in ['scaling_x', 'scaling_y']:
                                st.session_state[key] = 1.0
                            elif key in ['rotation', 'shearing_x', 'shearing_y']:
                                st.session_state[key] = 0.0
                            elif key in ['reflection_horizontal', 'reflection_vertical', 'apply_bg_removal']:
                                st.session_state[key] = False
                            elif key == 'background_tolerance':
                                st.session_state[key] = 30
                            elif key in ['blur_intensity', 'sharpen_intensity']:
                                st.session_state[key] = 0.0
        
        # Apply transformations and display results
        params = {
            'translation_x': st.session_state.get('translation_x', 0),
            'translation_y': st.session_state.get('translation_y', 0),
            'scaling_x': st.session_state.get('scaling_x', 1),
            'scaling_y': st.session_state.get('scaling_y', 1),
            'rotation': st.session_state.get('rotation', 0),
            'shearing_x': st.session_state.get('shearing_x', 0),
            'shearing_y': st.session_state.get('shearing_y', 0),
            'reflection_horizontal': st.session_state.get('reflection_horizontal', False),
            'reflection_vertical': st.session_state.get('reflection_vertical', False)
        }
        
        # Create and apply transformation
        matrix = transformer.create_transformation_matrix(params)
        transformed_image = transformer.safe_apply_transformation(matrix)
        
        # Apply image processing
        blur_intensity = st.session_state.get('blur_intensity', 0.0)
        sharpen_intensity = st.session_state.get('sharpen_intensity', 0.0)
        bg_tolerance = st.session_state.get('background_tolerance', 30)
        apply_bg_removal = st.session_state.get('apply_bg_removal', False)
        
        # Start with transformed image for processing
        processed_image = transformed_image.copy()
        
        if blur_intensity > 0:
            processed_image = apply_blur(processed_image, blur_intensity)
        
        if sharpen_intensity > 0:
            processed_image = apply_sharpen(processed_image, sharpen_intensity)
        
        if apply_bg_removal:
            processed_image = remove_background(processed_image, bg_tolerance)
        
        # Store processed image in transformer
        transformer.processed_image = processed_image
        
        # Ensure processed_image is in correct format for Streamlit
        if transformer.processed_image is not None:
            # Convert to RGB if needed
            if transformer.processed_image.mode != 'RGB':
                transformer.processed_image = transformer.processed_image.convert('RGB')
            
            # Ensure image has proper format attribute
            if not hasattr(transformer.processed_image, 'format') or transformer.processed_image.format is None:
                transformer.processed_image.format = 'PNG'
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader(get_text('original_image'))
            st.image(transformer.image, use_container_width=True, caption=get_text('original_label'))
        
        with col2:
            st.subheader(get_text('transformed_image'))
            st.image(transformed_image, use_container_width=True, caption=get_text('transformed_label'))
            
            # Download button
            if transformed_image is not None:
                img_buffer = io.BytesIO()
                transformed_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label=get_text('download_image'),
                    data=img_buffer,
                    file_name="transformed_image.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        with col3:
            st.subheader(get_text('processed_image'))
            # Safety check for processed_image
            if transformer.processed_image is not None:
                st.image(transformer.processed_image, use_container_width=True, caption=get_text('processed_label'))
            else:
                # Fallback to transformed image if processed_image is None
                if transformed_image is not None:
                    st.image(transformed_image, use_container_width=True, caption=get_text('processed_label'))
                else:
                    st.info("No processed image available")
            
            # Download button for processed image
            if transformer.processed_image is not None:
                img_buffer = io.BytesIO()
                transformer.processed_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label=get_text('download_processed'),
                    data=img_buffer,
                    file_name="processed_image.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        # Display matrix
        st.subheader(get_text('transformation_matrix'))
        
        # Format matrix for display
        matrix_str = ""
        for i in range(matrix.shape[0]):
            row_str = " | ".join([f"{matrix[i,j]:8.3f}" for j in range(matrix.shape[1])])
            matrix_str += f"[ {row_str} ]\n"
        
        st.code(matrix_str, language='text')
        
        # Matrix explanation
        with st.expander(get_text('matrix_explanation')):
            lang = st.session_state.get('language', 'id')
            if lang == 'id':
                st.markdown("""
                **Komponen Matriks Transformasi 3×3:**
                
                | Komponen | Deskripsi | Formula |
                |-----------|-------------|---------|
                | **[0,0], [0,1], [1,0], [1,1]] | Transformasi linear (rotasi, skala, geser) | Kombinasi dari semua transformasi |
                | **[0,2], [1,2]] | Translasi (perpindahan X, Y) | `tx, ty` |
                | **[2,0], [2,1]] | Perspektif (tidak digunakan dalam implementasi ini) | `0, 0` |
                | **[2,2]** | Koordinat homogen | `1` |
                
                **Urutan Matriks:** Refleksi → Skala → Rotasi → Geser → Translasi
                """)
            else:
                st.markdown("""
                **3×3 Transformation Matrix Components:**
                
                | Component | Description | Formula |
                |-----------|-------------|---------|
                | **[0,0], [0,1], [1,0], [1,1]] | Linear transformations (rotation, scale, shear) | Combination of all transforms |
                | **[0,2], [1,2]] | Translation (X, Y displacement) | `tx, ty` |
                | **[2,0], [2,1]] | Perspective (not used in this implementation) | `0, 0` |
                | **[2,2]** | Homogeneous coordinate | `1` |
                
                **Matrix Order:** Reflection → Scale → Rotation → Shear → Translation
                """)
    
    else:
        # Welcome message when no image is uploaded
        st.markdown(f"""
        <div class="info-message">
            <h2>{get_text('welcome_title')}</h2>
            <p>{get_text('welcome_subtitle')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Security notice
        st.markdown(f"### {get_text('security_notice')}")
        for point in TRANSLATIONS[st.session_state.language]['security_points']:
            st.markdown(f"- {point}")

def creator_profile():
    """Creator profile page"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{get_text('creator_profile')}</h1>
        <p>{get_text('profile_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Profile photo section
    st.markdown(f"### {get_text('photo_loading')} {get_text('loading_photo')}")
    
    # Try to load photo from GitHub
    photo_url = "https://raw.githubusercontent.com/Ocepsigma/Matrix-Transform/main/foto_yoseph.jpg"
    
    try:
        response = requests.get(photo_url, timeout=10)
        if response.status_code == 200:
            # Load and display photo
            image = Image.open(io.BytesIO(response.content))
            
            # Resize for consistent display
            image = image.resize((200, 200), Image.Resampling.LANCZOS)
            
            st.markdown('<div class="profile-photo-container">', unsafe_allow_html=True)
            st.image(image, width=200, caption="Yoseph Sihite")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(get_text('photo_error'))
            st.code(get_text('photo_url'))
    except Exception as e:
        st.warning(get_text('photo_error'))
        st.code(get_text('photo_url'))
    
    # Manual upload option
    st.markdown(get_text('upload_profile_photo'))
    uploaded_photo = st.file_uploader(
        get_text('choose_profile_photo'),
        type=['jpg', 'jpeg', 'png'],
        key="profile_photo_upload"
    )
    
    if uploaded_photo is not None:
        try:
            photo_image = Image.open(uploaded_photo)
            photo_image = photo_image.resize((200, 200), Image.Resampling.LANCZOS)
            st.image(photo_image, width=200, caption="Yoseph Sihite")
            st.success(get_text('photo_uploaded'))
        except Exception as e:
            st.error(f"Error loading photo: {str(e)}")
    
    # Developer information
    st.markdown("---")
    st.markdown(get_text('lead_developer'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **{get_text('name')}** Yoseph Sihite  
        **{get_text('student_id')}** [Your Student ID]  
        **{get_text('group')}** Group 2  
        **{get_text('role')}** {get_text('lead_developer').replace('## 👤 ', '')}
        """)
    
    with col2:
        st.markdown(f"""
        **Matematika & Algoritma**  
        **UI/UX Design**  
        **Web Development**  
        **Linear Algebra Implementation**
        """)
    
    # Project overview
    st.markdown("---")
    st.markdown(get_text('project_overview'))
    st.markdown(get_text('project_description'))
    
    # Contributions
    st.markdown(get_text('contributions'))
    st.markdown(get_text('contributions_description'))

def main():
    """Main application entry point"""
    # Language selector in sidebar
    st.sidebar.markdown(f"### {get_text('select_language')}")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🇮🇹 Indonesia"):
            st.session_state.language = 'id'
            st.rerun()
    
    with col2:
        if st.button("🇬🇧 English"):
            st.session_state.language = 'en'
            st.rerun()
    
    # Current language display
    current_lang = "🇮🇹 Bahasa Indonesia" if st.session_state.language == 'id' else "🇬🇧 English"
    st.sidebar.info(f"{get_text('current')} {current_lang}")
    
    # Navigation
    st.sidebar.markdown(f"### {get_text('navigate_to')}")
    
    page = st.sidebar.radio(
        "",
        [get_text('nav_matrix_studio'), get_text('nav_creator_profile')],
        key="page_selection"
    )
    
    # Page content
    if page == get_text('nav_matrix_studio'):
        main_app()
    else:
        creator_profile()

if __name__ == "__main__":
    main()
