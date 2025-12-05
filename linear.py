#!/usr/bin/env python3
"""
🔄 Matrix Transformation Studio - Ultra Robust Version
Fixed for parameter editing crashes
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import traceback
from typing import Dict, Any

# Set page configuration
st.set_page_config(
    page_title="Matrix Transformation Studio",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

class UltraRobustTransformer:
    """Ultra robust transformation engine with comprehensive error handling"""
    
    def __init__(self):
        self.image = None
        self.transformed_image = None
        self.last_successful_params = None
    
    def load_image(self, image_source) -> bool:
        """Load image with comprehensive error handling"""
        try:
            if hasattr(image_source, 'read'):  # UploadedFile
                # Reset file pointer
                image_source.seek(0)
                self.image = Image.open(image_source).convert('RGB')
            elif isinstance(image_source, str):  # File path
                self.image = Image.open(image_source).convert('RGB')
            elif isinstance(image_source, Image.Image):  # PIL Image
                self.image = image_source.convert('RGB')
            else:
                st.error("❌ Unsupported image source type")
                return False
            
            # Auto-resize for performance
            max_size = 600
            if self.image.width > max_size or self.image.height > max_size:
                ratio = min(max_size / self.image.width, max_size / self.image.height)
                new_width = int(self.image.width * ratio)
                new_height = int(self.image.height * ratio)
                self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            return False
    
    def safe_float(self, value, default=0.0):
        """Safely convert to float"""
        try:
            return float(value)
        except:
            return default
    
    def create_transformation_matrix(self, params: Dict[str, Any]) -> np.ndarray:
        """Create transformation matrix with error handling"""
        try:
            # Extract parameters with safe defaults
            tx = self.safe_float(params.get('translation_x', 0))
            ty = self.safe_float(params.get('translation_y', 0))
            sx = self.safe_float(params.get('scaling_x', 1))
            sy = self.safe_float(params.get('scaling_y', 1))
            rotation = self.safe_float(params.get('rotation', 0))
            shear_x = self.safe_float(params.get('shearing_x', 0))
            shear_y = self.safe_float(params.get('shearing_y', 0))
            reflect_h = bool(params.get('reflection_horizontal', False))
            reflect_v = bool(params.get('reflection_vertical', False))
            
            # Validate parameters
            sx = max(0.1, min(5.0, sx))  # Limit scale range
            sy = max(0.1, min(5.0, sy))
            tx = max(-500, min(500, tx))  # Limit translation
            ty = max(-500, min(500, ty))
            rotation = max(-360, min(360, rotation))  # Limit rotation
            shear_x = max(-2.0, min(2.0, shear_x))  # Limit shear
            shear_y = max(-2.0, min(2.0, shear_y))
            
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
            
            # Store successful parameters
            self.last_successful_params = params.copy()
            
            return matrix
            
        except Exception as e:
            st.error(f"❌ Error creating transformation matrix: {str(e)}")
            # Return identity matrix if error
            return np.eye(3)
    
    def apply_transformation(self, matrix: np.ndarray) -> Image.Image:
        """Apply transformation with comprehensive error handling"""
        try:
            if self.image is None:
                raise ValueError("No image loaded")
            
            width, height = self.image.size
            
            # Create a larger canvas
            canvas_size = max(width, height) * 4
            canvas = Image.new('RGB', (canvas_size, canvas_size), 'white')
            
            # Calculate center position
            center_x = canvas_size // 2
            center_y = canvas_size // 2
            
            # Paste original image in center
            paste_x = center_x - width // 2
            paste_y = center_y - height // 2
            canvas.paste(self.image, (paste_x, paste_y))
            
            # Apply transformations step by step with validation
            transformed = canvas
            
            # Extract transformation parameters safely
            try:
                tx = float(matrix[0, 2])
                ty = float(matrix[1, 2])
            except:
                tx, ty = 0, 0
            
            # Apply translation
            if abs(tx) > 0 or abs(ty) > 0:
                try:
                    new_size = (canvas_size + abs(int(tx)) * 2, canvas_size + abs(int(ty)) * 2)
                    new_canvas = Image.new('RGB', new_size, 'white')
                    new_x = max(0, new_size[0] // 2 - canvas_size // 2 + int(tx))
                    new_y = max(0, new_size[1] // 2 - canvas_size // 2 + int(ty))
                    new_canvas.paste(transformed, (new_x, new_y))
                    transformed = new_canvas
                except Exception as e:
                    st.warning(f"⚠️ Translation warning: {str(e)}")
            
            # Apply rotation
            try:
                rotation_angle = np.arctan2(matrix[1, 0], matrix[0, 0]) * 180 / np.pi
                if abs(rotation_angle) > 0.1:
                    transformed = transformed.rotate(-rotation_angle, expand=True, fillcolor='white')
            except Exception as e:
                st.warning(f"⚠️ Rotation warning: {str(e)}")
            
            # Apply scaling
            try:
                scale_x = abs(float(matrix[0, 0]))
                scale_y = abs(float(matrix[1, 1]))
                if abs(scale_x - 1) > 0.01 or abs(scale_y - 1) > 0.01:
                    new_width = max(1, int(transformed.width * scale_x))
                    new_height = max(1, int(transformed.height * scale_y))
                    if new_width < 5000 and new_height < 5000:  # Prevent huge images
                        transformed = transformed.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except Exception as e:
                st.warning(f"⚠️ Scaling warning: {str(e)}")
            
            # Apply reflection
            try:
                if matrix[0, 0] < 0:  # Horizontal reflection
                    transformed = transformed.transpose(Image.FLIP_LEFT_RIGHT)
                if matrix[1, 1] < 0:  # Vertical reflection
                    transformed = transformed.transpose(Image.FLIP_TOP_BOTTOM)
            except Exception as e:
                st.warning(f"⚠️ Reflection warning: {str(e)}")
            
            # Crop to content
            try:
                bbox = transformed.getbbox()
                if bbox:
                    transformed = transformed.crop(bbox)
            except Exception as e:
                st.warning(f"⚠️ Cropping warning: {str(e)}")
            
            # Final validation
            if transformed is None or transformed.size[0] <= 0 or transformed.size[1] <= 0:
                st.warning("⚠️ Transformation failed, returning original image")
                return self.image
            
            self.transformed_image = transformed
            return transformed
            
        except Exception as e:
            st.error(f"❌ Error applying transformation: {str(e)}")
            st.error("🔧 Returning original image")
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

def create_sample_image():
    """Create sample image with error handling"""
    try:
        img = Image.new('RGB', (400, 300), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw shapes
        draw.rectangle([50, 50, 150, 150], fill='red', outline='darkred', width=2)
        draw.ellipse([250, 100, 350, 200], fill='green', outline='darkgreen', width=2)
        draw.line([(0, 0), (400, 300)], fill='blue', width=3)
        
        # Add text
        try:
            draw.text((200, 250), "DEMO", fill='black', anchor='mm')
        except:
            pass  # Skip text if font fails
        
        return img
    except Exception as e:
        st.error(f"❌ Error creating sample image: {str(e)}")
        return Image.new('RGB', (400, 300), 'white')

def display_image_safely(image: Image.Image, title: str, label: str = None):
    """Safely display image"""
    try:
        if image is None:
            st.warning(f"No image to display for {title}")
            return
        
        st.subheader(title)
        st.image(image, use_container_width=True, caption=label)
        
    except Exception as e:
        st.error(f"❌ Error displaying image: {str(e)}")

def display_matrix_safely(matrix: np.ndarray, title: str = "Transformation Matrix"):
    """Safely display matrix"""
    try:
        st.subheader(title)
        
        # Format matrix for display
        matrix_str = ""
        for i in range(matrix.shape[0]):
            row_str = " | ".join([f"{matrix[i,j]:8.3f}" for j in range(matrix.shape[1])])
            matrix_str += f"[ {row_str} ]\n"
        
        st.code(matrix_str, language='text')
        
    except Exception as e:
        st.error(f"❌ Error displaying matrix: {str(e)}")

def get_safe_session_state(key, default=0):
    """Get session state value safely"""
    try:
        return st.session_state.get(key, default)
    except:
        return default

def set_safe_session_state(key, value):
    """Set session state value safely"""
    try:
        st.session_state[key] = value
    except Exception as e:
        st.warning(f"⚠️ Session state warning: {str(e)}")

def main():
    """Main application with robust error handling"""
    try:
        # Header
        st.title("🔄 Matrix Transformation Studio")
        st.markdown("Advanced Image Processing with Matrix Operations")
        st.markdown("---")
        
        # Initialize transformer
        transformer = UltraRobustTransformer()
        
        # Sidebar
        with st.sidebar:
            st.header("🎛️ Transformation Controls")
            
            # File upload
            uploaded_file = st.file_uploader(
                "📤 Upload Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload an image to apply transformations"
            )
            
            # Demo mode
            demo_mode = st.checkbox("🎨 Demo Mode", help="Use sample image")
            
            if demo_mode:
                sample_img = create_sample_image()
                transformer.load_image(sample_img)
                st.success("✅ Demo image loaded!")
            elif uploaded_file is not None:
                if transformer.load_image(uploaded_file):
                    st.success("✅ Image loaded successfully!")
            
            # Only show controls if image is loaded
            if transformer.image is not None:
                st.markdown("---")
                st.subheader("📐 Transformation Parameters")
                
                # Get current values safely
                tx = get_safe_session_state('tx', 0)
                ty = get_safe_session_state('ty', 0)
                sx = get_safe_session_state('sx', 1.0)
                sy = get_safe_session_state('sy', 1.0)
                rotation = get_safe_session_state('rotation', 0)
                reflect_h = get_safe_session_state('reflect_h', False)
                reflect_v = get_safe_session_state('reflect_v', False)
                
                # Translation
                st.markdown("**🔄 Translation**")
                tx = st.slider("X (pixels)", -200, 200, int(tx), key="tx")
                ty = st.slider("Y (pixels)", -200, 200, int(ty), key="ty")
                
                # Scaling
                st.markdown("**📏 Scaling**")
                sx = st.slider("X factor", 0.1, 3.0, float(sx), 0.1, key="sx")
                sy = st.slider("Y factor", 0.1, 3.0, float(sy), 0.1, key="sy")
                
                # Rotation
                st.markdown("**🔄 Rotation**")
                rotation = st.slider("Angle (degrees)", -180, 180, int(rotation), key="rotation")
                
                # Reflection
                st.markdown("**🔁 Reflection**")
                col1, col2 = st.columns(2)
                with col1:
                    reflect_h = st.checkbox("Horizontal", reflect_h, key="reflect_h")
                with col2:
                    reflect_v = st.checkbox("Vertical", reflect_v, key="reflect_v")
                
                # Presets
                st.markdown("---")
                st.subheader("⚡ Preset Transformations")
                presets = transformer.get_preset_transformations()
                selected_preset = st.selectbox("Choose preset:", ["None"] + list(presets.keys()))
                
                if selected_preset != "None":
                    preset = presets[selected_preset]
                    set_safe_session_state('tx', preset['translation_x'])
                    set_safe_session_state('ty', preset['translation_y'])
                    set_safe_session_state('sx', preset['scaling_x'])
                    set_safe_session_state('sy', preset['scaling_y'])
                    set_safe_session_state('rotation', preset['rotation'])
                    set_safe_session_state('reflect_h', preset['reflection_horizontal'])
                    set_safe_session_state('reflect_v', preset['reflection_vertical'])
                    st.rerun()
                
                # Reset button
                if st.button("🔄 Reset All", use_container_width=True):
                    keys_to_remove = ['tx', 'ty', 'sx', 'sy', 'rotation', 'reflect_h', 'reflect_v']
                    for key in keys_to_remove:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        
        # Main content
        if transformer.image is not None:
            # Get current parameters safely
            params = {
                'translation_x': get_safe_session_state('tx', 0),
                'translation_y': get_safe_session_state('ty', 0),
                'scaling_x': get_safe_session_state('sx', 1.0),
                'scaling_y': get_safe_session_state('sy', 1.0),
                'rotation': get_safe_session_state('rotation', 0),
                'shearing_x': 0,
                'shearing_y': 0,
                'reflection_horizontal': get_safe_session_state('reflect_h', False),
                'reflection_vertical': get_safe_session_state('reflect_v', False)
            }
            
            # Create and apply transformation with error handling
            try:
                matrix = transformer.create_transformation_matrix(params)
                transformed_image = transformer.apply_transformation(matrix)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    display_image_safely(transformer.image, "📷 Original Image", "ORIGINAL")
                
                with col2:
                    display_image_safely(transformed_image, "✨ Transformed Image", "TRANSFORMED")
                    
                    # Download button
                    if transformed_image is not None:
                        try:
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
                        except Exception as e:
                            st.error(f"❌ Download error: {str(e)}")
                
                # Display matrix
                display_matrix_safely(matrix)
                
                # Active transformations
                active = []
                if params.get('translation_x', 0) != 0 or params.get('translation_y', 0) != 0:
                    active.append("Translation")
                if params.get('scaling_x', 1) != 1 or params.get('scaling_y', 1) != 1:
                    active.append("Scaling")
                if params.get('rotation', 0) != 0:
                    active.append("Rotation")
                if params.get('reflection_horizontal', False) or params.get('reflection_vertical', False):
                    active.append("Reflection")
                
                if active:
                    st.info(f"🔧 Active Transformations: {', '.join(active)}")
                else:
                    st.info("ℹ️ No active transformations")
                    
            except Exception as e:
                st.error(f"❌ Transformation error: {str(e)}")
                st.error("🔧 Please adjust parameters and try again")
                if transformer.last_successful_params:
                    st.info("💡 Using last successful parameters")
        
        else:
            st.info("👆 Upload an image or enable Demo Mode to start transforming")
            
            # Feature info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🔄 Translation**
                - Move objects along X and Y axes
                - Range: -200 to 200 pixels
                """)
            
            with col2:
                st.markdown("""
                **📏 Scaling**
                - Resize objects with scale factors
                - Range: 0.1x to 3.0x
                """)
            
            with col3:
                st.markdown("""
                **🔄 Rotation**
                - Rotate objects by angle
                - Range: -180° to 180°
                """)
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.error("🔧 Please refresh the page and try again")
        # Show traceback for debugging
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
