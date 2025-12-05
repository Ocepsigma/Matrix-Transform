import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from typing import Tuple, Optional

class MatrixTransformer:
    """
    Class untuk melakukan transformasi matriks pada gambar
    """
    
    def __init__(self):
        self.image = None
        self.transformed_image = None
    
    def load_image(self, uploaded_file) -> bool:
        """Load image dari uploaded file"""
        try:
            # Baca file yang diupload
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            self.image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if self.image is None:
                st.error("Gagal memuat gambar. Pastikan file gambar valid.")
                return False
            
            # Convert BGR ke RGB untuk matplotlib
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            return True
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return False
    
    def create_transformation_matrix(self, params: dict) -> np.ndarray:
        """
        Membuat matriks transformasi 3x3 berdasarkan parameter
        """
        # Matriks identitas
        matrix = np.eye(3)
        
        # Ekstrak parameter
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
        
        # Matriks transformasi
        transform_matrix = np.eye(3)
        
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
    
    def get_preset_transformations(self) -> dict:
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

def main():
    st.set_page_config(
        page_title="Matrix Transformation Studio",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔄 Matrix Transformation Studio")
    st.markdown("Aplikasi Image Processing dengan Transformasi Matriks")
    
    # Initialize transformer
    if 'transformer' not in st.session_state:
        st.session_state.transformer = MatrixTransformer()
    
    transformer = st.session_state.transformer
    
    # Sidebar untuk controls
    st.sidebar.header("🎛️ Kontrol Transformasi")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Gambar",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload gambar untuk di-transformasi"
    )
    
    if uploaded_file is not None:
        if transformer.load_image(uploaded_file):
            st.sidebar.success("✅ Gambar berhasil dimuat!")
            
            # Transformasi parameters
            st.sidebar.subheader("📐 Parameter Transformasi")
            
            # Translation
            st.sidebar.markdown("**🔄 Translation (Translasi)**")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                tx = st.slider("X (pixels)", -200, 200, 0, key="tx")
            with col2:
                ty = st.slider("Y (pixels)", -200, 200, 0, key="ty")
            
            # Scaling
            st.sidebar.markdown("**📏 Scaling (Skala)**")
            col3, col4 = st.sidebar.columns(2)
            with col3:
                sx = st.slider("X", 0.1, 3.0, 1.0, 0.1, key="sx")
            with col4:
                sy = st.slider("Y", 0.1, 3.0, 1.0, 0.1, key="sy")
            
            # Rotation
            st.sidebar.markdown("**🔄 Rotation (Rotasi)**")
            rotation = st.slider("Sudut (derajat)", -180, 180, 0, key="rotation")
            
            # Shearing
            st.sidebar.markdown("**🔀 Shearing (Skew)**")
            col5, col6 = st.sidebar.columns(2)
            with col5:
                shear_x = st.slider("X", -1.0, 1.0, 0.0, 0.1, key="shear_x")
            with col6:
                shear_y = st.slider("Y", -1.0, 1.0, 0.0, 0.1, key="shear_y")
            
            # Reflection
            st.sidebar.markdown("**🔁 Reflection (Refleksi)**")
            col7, col8 = st.sidebar.columns(2)
            with col7:
                reflect_h = st.checkbox("Horizontal", key="reflect_h")
            with col8:
                reflect_v = st.checkbox("Vertical", key="reflect_v")
            
            # Preset transformations
            st.sidebar.subheader("⚡ Preset Transformations")
            presets = transformer.get_preset_transformations()
            
            selected_preset = st.sidebar.selectbox(
                "Pilih Preset:",
                ["None"] + list(presets.keys())
            )
            
            if selected_preset != "None":
                preset_params = presets[selected_preset]
                # Update session state dengan preset values
                st.session_state.tx = preset_params['translation_x']
                st.session_state.ty = preset_params['translation_y']
                st.session_state.sx = preset_params['scaling_x']
                st.session_state.sy = preset_params['scaling_y']
                st.session_state.rotation = preset_params['rotation']
                st.session_state.shear_x = preset_params['shearing_x']
                st.session_state.shear_y = preset_params['shearing_y']
                st.session_state.reflect_h = preset_params['reflection_horizontal']
                st.session_state.reflect_v = preset_params['reflection_vertical']
                st.rerun()
            
            # Reset button
            if st.sidebar.button("🔄 Reset All", type="secondary"):
                for key in st.session_state.keys():
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
            
            # Create transformation matrix
            transform_matrix = transformer.create_transformation_matrix(params)
            
            # Apply transformation
            transformed_image = transformer.apply_transformation(transform_matrix)
            
            # Main content area
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(transformer.image)
                ax.set_title("Original")
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("✨ Transformed Image")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(transformed_image)
                ax.set_title("Transformed")
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                
                # Download button
                if st.button("💾 Download Transformed Image"):
                    # Convert image to bytes
                    pil_image = Image.fromarray(transformed_image)
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download PNG",
                        data=img_buffer,
                        file_name="transformed_image.png",
                        mime="image/png"
                    )
            
            # Display transformation matrix
            display_matrix(transform_matrix)
            
            # Active transformations indicator
            active_transforms = []
            if tx != 0 or ty != 0:
                active_transforms.append("Translation")
            if sx != 1 or sy != 1:
                active_transforms.append("Scaling")
            if rotation != 0:
                active_transforms.append("Rotation")
            if shear_x != 0 or shear_y != 0:
                active_transforms.append("Shearing")
            if reflect_h or reflect_v:
                active_transforms.append("Reflection")
            
            if active_transforms:
                st.info(f"🔧 Active Transformations: {', '.join(active_transforms)}")
            else:
                st.info("ℹ️ No active transformations")
    
    else:
        st.info("👆 Silakan upload gambar untuk memulai transformasi")
        
        # Display sample transformations info
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

if __name__ == "__main__":
    main()
