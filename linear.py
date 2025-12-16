#!/usr/bin/env python3
"""
🔄 Matrix Transformation Studio - Final Version Complete with Image Processing
Page 1: Main Application with Image Processing Features
Page 2: Creator Profile - Yoseph Sihite
✅ Foto profil dengan zoom yang tepat dan posisi yang seimbang
✅ Development Team tanpa HTML
✅ Fungsi load foto dari GitHub yang diperbaiki
✅ Dukungan 2 Bahasa: Indonesia & English
✅ Profil yang lebih ringkas (tanpa kontribusi utama, teknologi, dan prestasi akademik)
✅ Semua fitur lengkap dan stabil
✅ Image Processing: Blur, Sharpen, Background Removal (digabung dalam satu hasil)
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
        'blur': '🌫 Blur',
        'sharpen': '🔍 Tajamkan',
        'background_removal': '🎨 Hapus Latar Belakang',
        
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
        'blur_intensity': 'Intensitas Blur',
        'sharpen_intensity': 'Intensitas Tajamkan',
        'background_tolerance': 'Toleransi Latar Belakang',
        
        # Current values
        'current': 'Saat ini:',
        
        # Presets
        'preset_transformations': '⚡ Transformasi Preset',
        'choose_preset': 'Pilih preset:',
        'reset_all': '🔄 Reset Semua',
        
        # Display
        'original_image': '📷 Gambar Asli',
        'original_label': 'ASLI',
        'transformed_image': '✨ Gambar Hasil',
        'transformed_label': 'HASIL',
        'download_image': '💾 Unduh Gambar Hasil',
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
        'blur': '🌫 Blur',
        'sharpen': '🔍 Sharpen',
        'background_removal': '🎨 Background Removal',
        
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
        'blur_intensity': 'Blur Intensity',
        'sharpen_intensity': 'Sharpen Intensity',
        'background_tolerance': 'Background Tolerance',
        
        # Current values
        'current': 'Current:',
        
        # Presets
        'preset_transformations': '⚡ Preset Transformations',
        'choose_preset': 'Choose preset:',
        'reset_all': '🔄 Reset All',
        
        # Display
        'original_image': '📷 Original Image',
        'original_label': 'ORIGINAL',
        'transformed_image': '✨ Result Image',
        'transformed_label': 'RESULT',
        'download_image': '💾 Download Result Image',
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
    
    /* Profile photo styling - DIPERBAIKI */
    .profile-photo {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        object-fit: cover;
        object-position: center top;
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
    
    /* Language selector styling */
    .language-selector {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

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

def load_profile_photo():
    """
    Load profile photo dari GitHub dengan processing otomatis
    Sesuai anjuran dosen untuk fungsi yang lebih sederhana
    """
    try:
        # GitHub Raw URL yang benar
        url = "https://raw.githubusercontent.com/Ocepsigma/Matrix-Transform/main/foto_yoseph.jpg"
        
        # Request ke GitHub Raw
        response = requests.get(url)
        
        if response.status_code == 200:
            # Load image untuk diproses
            image = Image.open(io.BytesIO(response.content))
            
            # Proses foto untuk profil dengan zoom yang tepat
            processed_image = process_profile_photo(image)
            
            # Convert processed image ke base64
            img_buffer = io.BytesIO()
            processed_image.save(img_buffer, format='JPEG', quality=90)
            img_bytes = img_buffer.getvalue()
            photo_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            return photo_base64, True
        else:
            return None, False
            
    except Exception as e:
        return None, False

def process_profile_photo(image):
    """
    Process foto profil untuk tampilan optimal dengan zoom yang seimbang
    DIPERBAIKI: Posisi crop yang lebih turun untuk wajah yang lengkap
    """
    try:
        # Convert ke RGB jika perlu
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Dapatkan dimensi asli
        width, height = image.size
        
        # Target size untuk profil (lebih besar untuk kualitas)
        target_size = 400
        
        # Hitung rasio untuk zoom yang lebih seimbang
        # Untuk foto portrait, zoom ke area wajah dengan proporsi lebih baik
        if height > width:  # Portrait orientation
            # DIPERBAIKI: Crop ke area wajah yang lebih turun
            crop_height = int(height * 0.7)  # 70% dari tinggi
            crop_top = int(height * 0.15)     # Mulai dari 15% dari atas (lebih turun)
            crop_bottom = crop_top + crop_height
            
            # Jika lebar terlalu kecil, crop dari samping
            if width < target_size:
                crop_width = width
                crop_left = 0
                crop_right = width
            else:
                crop_width = int(width * 0.85)  # Ambil 85% dari lebar
                crop_left = (width - crop_width) // 2
                crop_right = crop_left + crop_width
            
            # Crop gambar
            cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        else:  # Landscape orientation
            # Crop ke area tengah
            crop_width = int(width * 0.8)
            crop_height = int(height * 0.8)
            crop_left = (width - crop_width) // 2
            crop_top = (height - crop_height) // 2
            crop_right = crop_left + crop_width
            crop_bottom = crop_top + crop_height
            
            cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        # Resize ke target size dengan high quality
        processed_image = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # DIPERBAIKI: Enhancement yang lebih subtle
        enhancer = ImageEnhance.Brightness(processed_image)
        processed_image = enhancer.enhance(1.05)  # 5% brightness (lebih subtle)
        
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(1.05)  # 5% contrast (lebih subtle)
        
        enhancer = ImageEnhance.Sharpness(processed_image)
        processed_image = enhancer.enhance(1.05)  # 5% sharpness (lebih subtle)
        
        return processed_image
        
    except Exception as e:
        # Jika proses gagal, return original image
        return image

class SafeMatrixTransformer:
    """Matrix Transformer dengan proteksi DecompressionBombError dan proper image handling"""
    
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
            
            # Auto-resize untuk performance
            max_size = 2000
            if self.image.width > max_size or self.image.height > max_size:
                ratio = min(max_size/self.image.width, max_size/self.image.height)
                new_width = int(self.image.width * ratio)
                new_height = int(self.image.height * ratio)
                self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
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
    
    def apply_all_effects(self, matrix: np.ndarray, blur_intensity=0.0, sharpen_intensity=0.0, bg_removal=False, bg_tolerance=30) -> Image.Image:
        """Apply transformation and all image processing effects in one step"""
        try:
            # First apply matrix transformation
            result = self.safe_apply_transformation(matrix)
            
            # Then apply image processing effects
            if blur_intensity > 0:
                result = apply_blur(result, blur_intensity)
            
            if sharpen_intensity > 0:
                result = apply_sharpen(result, sharpen_intensity)
            
            if bg_removal:
                result = remove_background(result, bg_tolerance)
            
            # Update transformed_image with final result
            self.transformed_image = result
            
            return result
            
        except Exception as e:
            st.error(f"Error applying effects: {str(e)}")
            return self.image
    
    def get_preset_transformations(self) -> Dict[str, Dict[str, Any]]:
        """Get preset translations"""
        lang = st.session_state.get('language', 'id')
        
        if lang == 'id':
            return {
                "Balik Horizontal": {
                    'translation_x': 0, 'translation_y': 0,
                    'scaling_x': 1, 'scaling_y': 1,
                    'rotation': 0,
                    'shearing_x': 0, 'shearing_y': 0,
                    'reflection_horizontal': True, 'reflection_vertical': False
                },
                "Balik Vertikal": {
                    'translation_x': 0, 'translation_y': 0,
                    'scaling_x': 1, 'scaling_y': 1,
                    'rotation': 0,
                    'shearing_x': 0, 'shearing_y': 0,
                    'reflection_horizontal': False, 'reflection_vertical': True
                },
                "Putar 90°": {
                    'translation_x': 0, 'translation_y': 0,
                    'scaling_x': 1, 'scaling_y': 1,
                    'rotation': 90,
                    'shearing_x': 0, 'shearing_y': 0,
                    'reflection_horizontal': False, 'reflection_vertical': False
                },
                "Putar 180°": {
                    'translation_x': 0, 'translation_y': 0,
                    'scaling_x': 1, 'scaling_y': 1,
                    'rotation': 180,
                    'shearing_x': 0, 'shearing_y': 0,
                    'reflection_horizontal': False, 'reflection_vertical': False
                },
                "Skala 2x": {
                    'translation_x': 0, 'translation_y': 0,
                    'scaling_x': 2, 'scaling_y': 2,
                    'rotation': 0,
                    'shearing_x': 0, 'shearing_y': 0,
                    'reflection_horizontal': False, 'reflection_vertical': False
                },
                "Skala 0.5x": {
                    'translation_x': 0, 'translation_y': 0,
                    'scaling_x': 0.5, 'scaling_y': 0.5,
                    'rotation': 0,
                    'shearing_x': 0, 'shearing_y': 0,
                    'reflection_horizontal': False, 'reflection_vertical': False
                },
                "Geser Kanan": {
                    'translation_x': 0, 'translation_y': 0,
                    'scaling_x': 1, 'scaling_y': 1,
                    'rotation': 0,
                    'shearing_x': 0.3, 'shearing_y': 0,
                    'reflection_horizontal': False, 'reflection_vertical': False
                },
                "Geser Atas": {
                    'translation_x': 0, 'translation_y': 0,
                    'scaling_x': 1, 'scaling_y': 1,
                    'rotation': 0,
                    'shearing_x': 0, 'shearing_y': -0.3,
                    'reflection_horizontal': False, 'reflection_vertical': False
                }
            }
        else:
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
    """Main Matrix Transformation Application with Image Processing"""
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">{get_text('app_title')}</h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.25rem; opacity: 0.9;">{get_text('app_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize transformer
    transformer = SafeMatrixTransformer()
    
    # Sidebar
    with st.sidebar:
        # Language selector
        st.markdown(f"""
        <div class="language-selector">
            <h3 style="margin: 0 0 1rem 0;">{get_text('select_language')}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Language selection
        if 'language' not in st.session_state:
            st.session_state.language = 'id'
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🇮🇩 Indonesia", use_container_width=True):
                st.session_state.language = 'id'
                st.rerun()
        with col2:
            if st.button("🇺🇸 English", use_container_width=True):
                st.session_state.language = 'en'
                st.rerun()
        
        st.markdown("---")
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
            <h3 style="margin: 0; color: white;">{get_text('transformation_controls')}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload dengan proteksi
        uploaded_file = st.file_uploader(
            get_text('upload_image'),
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help=get_text('upload_help')
        )
        
        if uploaded_file is not None:
            # Cek ukuran file
            file_size = uploaded_file.size
            max_file_size = 10 * 1024 * 1024  # 10MB
            
            if file_size > max_file_size:
                st.error(f"{get_text('file_too_large')} ({file_size/1024/1024:.1f}MB)")
                st.info(f"{get_text('max_size')} {max_file_size/1024/1024}MB")
            else:
                if transformer.safe_load_image(uploaded_file):
                    st.markdown(f'<div class="success-message">{get_text("image_loaded")}</div>', unsafe_allow_html=True)
        
        # Only show controls if image is loaded
        if transformer.image is not None:
            st.markdown("---")
            
            # Transformation tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                get_text('translation'), get_text('scaling'), get_text('rotation'), 
                get_text('shearing'), get_text('reflection'), get_text('blur'), get_text('sharpen')
            ])
            
            with tab1:
                st.markdown(f"**{get_text('translation_params')}**")
                tx = st.slider(get_text('x_translation'), -200, 200, 0, key="translation_x")
                ty = st.slider(get_text('y_translation'), -200, 200, 0, key="translation_y")
                st.caption(f"{get_text('current')} X={tx}, Y={ty}")
            
            with tab2:
                st.markdown(f"**{get_text('scaling_params')}**")
                sx = st.slider(get_text('x_scale_factor'), 0.1, 3.0, 1.0, 0.1, key="scaling_x")
                sy = st.slider(get_text('y_scale_factor'), 0.1, 3.0, 1.0, 0.1, key="scaling_y")
                st.caption(f"{get_text('current')} X={sx:.1f}x, Y={sy:.1f}x")
            
            with tab3:
                st.markdown(f"**{get_text('rotation_params')}**")
                rotation = st.slider(get_text('rotation_angle'), -180, 180, 0, key="rotation")
                st.caption(f"{get_text('current')} {rotation}°")
            
            with tab4:
                st.markdown(f"**{get_text('shearing_params')}**")
                shear_x = st.slider(get_text('x_shear_factor'), -1.0, 1.0, 0.0, 0.1, key="shearing_x")
                shear_y = st.slider(get_text('y_shear_factor'), -1.0, 1.0, 0.0, 0.1, key="shearing_y")
                st.caption(f"{get_text('current')} X={shear_x:.1f}, Y={shear_y:.1f}")
            
            with tab5:
                st.markdown(f"**{get_text('reflection_params')}**")
                col1, col2 = st.columns(2)
                with col1:
                    reflect_h = st.checkbox(get_text('horizontal_reflection'), key="reflection_horizontal")
                with col2:
                    reflect_v = st.checkbox(get_text('vertical_reflection'), key="reflection_vertical")
                st.caption(f"{get_text('current')} H={reflect_h}, V={reflect_v}")
            
            with tab6:
                st.markdown(f"**{get_text('blur_desc')}**")
                blur_intensity = st.slider(get_text('blur_intensity'), 0.0, 5.0, 0.0, 0.1, key="blur_intensity")
            
            with tab7:
                st.markdown(f"**{get_text('sharpen_desc')}**")
                sharpen_intensity = st.slider(get_text('sharpen_intensity'), 0.0, 2.0, 0.0, 0.1, key="sharpen_intensity")
            
            # Background removal option
            st.markdown("---")
            st.markdown(f"**{get_text('background_removal_desc')}**")
            apply_bg_removal = st.checkbox(get_text('background_removal'), key="apply_bg_removal")
            
            if apply_bg_removal:
                bg_tolerance = st.slider(get_text('background_tolerance'), 10, 100, 30, 5, key="bg_tolerance")
            else:
                bg_tolerance = 30
            
            # Presets
            st.markdown("---")
            st.subheader(get_text('preset_transformations'))
            presets = transformer.get_preset_transformations()
            selected_preset = st.selectbox(get_text('choose_preset'), ["None"] + list(presets.keys()))
            
            if selected_preset != "None":
                preset = presets[selected_preset]
                st.session_state.update(preset)
                st.rerun()
            
            # Reset button
            if st.button(get_text('reset_all'), use_container_width=True):
                keys_to_remove = ['translation_x', 'translation_y', 'scaling_x', 'scaling_y', 'rotation', 'shearing_x', 'shearing_y', 'reflection_horizontal', 'reflection_vertical', 'blur_intensity', 'sharpen_intensity', 'apply_bg_removal', 'bg_tolerance']
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
        
        # Get image processing parameters
        blur_intensity = st.session_state.get('blur_intensity', 0.0)
        sharpen_intensity = st.session_state.get('sharpen_intensity', 0.0)
        apply_bg_removal = st.session_state.get('apply_bg_removal', False)
        bg_tolerance = st.session_state.get('bg_tolerance', 30)
        
        # Create and apply transformation with all effects
        matrix = transformer.create_transformation_matrix(params)
        transformed_image = transformer.apply_all_effects(
            matrix, 
            blur_intensity, 
            sharpen_intensity, 
            apply_bg_removal, 
            bg_tolerance
        )
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        
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
                | **[0,0], [0,1], [1,0], [1,1]] | Linear transformation (rotation, scale, shear) | Combined from all transforms |
                | **[0,2], [1,2]] | Translation (X, Y displacement) | `tx, ty` |
                | **[2,0], [2,1]] | Perspective (unused in this implementation) | `0, 0` |
                | **[2,2]** | Homogeneous coordinate | `1` |
                
                **Matrix Order:** Reflection → Scaling → Rotation → Shearing → Translation
                """)
        
        # Active transformations
        active = []
        if params.get('translation_x', 0) != 0 or params.get('translation_y', 0) != 0:
            active.append(get_text('translation'))
        if params.get('scaling_x', 1) != 1 or params.get('scaling_y', 1) != 1:
            active.append(get_text('scaling'))
        if params.get('rotation', 0) != 0:
            active.append(get_text('rotation'))
        if params.get('shearing_x', 0) != 0 or params.get('shearing_y', 0) != 0:
            active.append(get_text('shearing'))
        if params.get('reflection_horizontal', False) or params.get('reflection_vertical', False):
            active.append(get_text('reflection'))
        if blur_intensity > 0:
            active.append(get_text('blur'))
        if sharpen_intensity > 0:
            active.append(get_text('sharpen'))
        if apply_bg_removal:
            active.append(get_text('background_removal'))
        
        if active:
            st.info(f"{get_text('active_transformations')} {', '.join(active)}")
        else:
            st.info(get_text('no_active_transformations'))
    
    else:
        # Welcome screen
        st.markdown(f"""
        <div class="card">
            <h3 style="margin: 0 0 1rem 0; color: #1e293b;">{get_text('welcome_title')}</h3>
            <p style="margin: 0 0 1rem 0; color: #64748b;">{get_text('welcome_subtitle')}</p>
            <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>{get_text('security_notice')}</strong><br>
                {'<br>'.join([f"• {point}" for point in get_text('security_points')])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">{get_text('translation')}</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">{get_text('translation_desc')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #764ba2;">{get_text('scaling')}</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">{get_text('scaling_desc')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">{get_text('rotation')}</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">{get_text('rotation_desc')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown(f"""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #764ba2;">{get_text('shearing')}</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">{get_text('shearing_desc')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="card">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">{get_text('reflection')}</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">{get_text('reflection_desc')}</p>
            </div>
            """, unsafe_allow_html=True)

def profile_page():
    """
    Profile page for Yoseph Sihite dengan foto profil yang diperbaiki
    ✅ Development Team TANPA HTML (menggunakan st.write() saja)
    ✅ Foto profil dengan zoom yang seimbang dan posisi yang tepat
    ✅ Dukungan bilingual (Indonesia & English)
    ✅ Profil yang lebih ringkas (tanpa kontribusi utama, teknologi, dan prestasi akademik)
    """
    # Header
    st.markdown(f"""
    <div class="profile-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">{get_text('creator_profile')}</h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.25rem; opacity: 0.9;">{get_text('profile_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load profile photo dengan processing otomatis
    photo_base64, photo_loaded = load_profile_photo()
    
    # Debug info untuk foto
    if not photo_loaded:
        st.warning(get_text('photo_error'))
        st.info(get_text('photo_url'))
        
        # Fallback: Upload foto manual dengan processing yang sama
        st.write(get_text('upload_profile_photo'))
        uploaded_photo = st.file_uploader(get_text('choose_profile_photo'), type=['jpg', 'jpeg', 'png'])
        if uploaded_photo is not None:
            # Load dan proses uploaded file dengan fungsi yang sama
            image = Image.open(uploaded_photo)
            processed_image = process_profile_photo(image)
            
            # Convert processed image ke base64
            img_buffer = io.BytesIO()
            processed_image.save(img_buffer, format='JPEG', quality=90)
            img_bytes = img_buffer.getvalue()
            photo_base64 = base64.b64encode(img_bytes).decode("utf-8")
            photo_loaded = True
            st.success(get_text('photo_uploaded'))
    
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
            st.markdown(f"""
                <div class="photo-loading">
                    {get_text('photo_loading')}
                </div>
                <p style="text-align: center; margin-top: 0.5rem; color: #64748b; font-size: 0.9rem;">{get_text('loading_photo')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            </div>
            <h3 style="text-align: center; margin: 1rem 0 0.5rem 0; color: #1e293b;">Yoseph Sihite</h3>
            <p style="text-align: center; margin: 0.5rem 0; color: #64748b;">📍 Cikarang, Indonesia</p>
            <div style="text-align: center; margin: 1rem 0;">
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # ✅ Development Team content TANPA HTML (hanya st.write())
        st.write("---")
        st.write(get_text('lead_developer'))
        st.write(f"**{get_text('name')}** Yoseph Sihite")
        st.write(f"**{get_text('student_id')}** 004202400113")
        st.write(f"**{get_text('group')}** Group 2 Linear Algebra")
        st.write(f"**{get_text('role')}** Lead Group")
        
        st.write("---")
        st.write(get_text('project_overview'))
        st.write(get_text('project_description'))
        
        st.write("---")
        st.write(get_text('contributions'))
        st.write(get_text('contributions_description'))
        
def main():
    """Main application with multi-page navigation and bilingual support"""
    # Page navigation
    page = st.sidebar.selectbox(
        get_text('navigate_to'),
        [get_text('nav_matrix_studio'), get_text('nav_creator_profile')],
        index=0,
        format_func=lambda x: x.split(" ", 1)[1] if " " in x else x
    )
    
    if page == get_text('nav_matrix_studio'):
        main_app()
    elif page == get_text('nav_creator_profile'):
        profile_page()

if __name__ == "__main__":
    main()
