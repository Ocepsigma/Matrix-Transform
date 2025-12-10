import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2
import io
import base64
import requests
import os
from urllib.parse import urlparse

# Set page configuration
st.set_page_config(
    page_title="Matrix Transformation Studio",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to safely load images with decompression bomb protection
def safe_load_image(uploaded_file):
    """
    Safely load an image with decompression bomb protection
    """
    try:
        # Set PIL's decompression bomb protection limit
        Image.MAX_IMAGE_PIXELS = 100000000  # 100MP limit
        
        # Read the file bytes
        image_bytes = uploaded_file.getvalue()
        
        # Check file size first (limit to 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            st.error("Image file is too large. Please upload an image smaller than 10MB.")
            return None
            
        # Open the image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except UnidentifiedImageError:
        st.error("Cannot identify image file. Please upload a valid image.")
        return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

# Function to load profile photo from GitHub with fallback options
def load_profile_photo():
    """
    Load profile photo from GitHub Raw URL with comprehensive error handling
    """
    # GitHub Raw URL for the profile photo
    github_url = "https://raw.githubusercontent.com/yosephsihite/matrix-transformation-studio/main/profile_photo.jpg"
    
    # Try multiple approaches
    approaches = [
        ("GitHub Raw URL", github_url),
        ("Alternative GitHub URL", "https://raw.githubusercontent.com/yosephsihite/matrix-transformation-studio/main/profile.png"),
        ("Direct GitHub API", "https://api.github.com/repos/yosephsihite/matrix-transformation-studio/contents/profile_photo.jpg"),
    ]
    
    for approach_name, url in approaches:
        try:
            st.write(f"Trying: {approach_name}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                # Check if it's actually image data
                if 'image' in response.headers.get('content-type', ''):
                    img = Image.open(io.BytesIO(response.content))
                    st.write(f"✅ Successfully loaded profile photo using {approach_name}")
                    return img
                else:
                    st.write(f"❌ {approach_name} returned non-image content")
            else:
                st.write(f"❌ {approach_name} failed with status code: {response.status_code}")
        except Exception as e:
            st.write(f"❌ {approach_name} failed with error: {str(e)}")
    
    # If all approaches fail, return None
    st.write("❌ All automatic loading methods failed")
    return None

# Function to create transformation matrix
def create_transformation_matrix(transformation_type, **params):
    """
    Create a 3x3 transformation matrix for the specified transformation type
    """
    matrix = np.eye(3)
    
    if transformation_type == "translation":
        dx, dy = params.get('dx', 0), params.get('dy', 0)
        matrix[0, 2] = dx
        matrix[1, 2] = dy
    elif transformation_type == "scaling":
        sx, sy = params.get('sx', 1), params.get('sy', 1)
        matrix[0, 0] = sx
        matrix[1, 1] = sy
    elif transformation_type == "rotation":
        angle = params.get('angle', 0)
        theta = np.radians(angle)
        matrix[0, 0] = np.cos(theta)
        matrix[0, 1] = -np.sin(theta)
        matrix[1, 0] = np.sin(theta)
        matrix[1, 1] = np.cos(theta)
    elif transformation_type == "reflection":
        axis = params.get('axis', 'x')
        if axis == 'x':
            matrix[1, 1] = -1
        elif axis == 'y':
            matrix[0, 0] = -1
        elif axis == 'origin':
            matrix[0, 0] = -1
            matrix[1, 1] = -1
    elif transformation_type == "shearing":
        shx, shy = params.get('shx', 0), params.get('shy', 0)
        matrix[0, 1] = shx
        matrix[1, 0] = shy
    
    return matrix

# Function to apply transformation matrix to image
def apply_transformation(image, matrix):
    """
    Apply a 3x3 transformation matrix to an image
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Get image dimensions
    h, w = img_array.shape[:2]
    
    # Create coordinate grid
    y, x = np.mgrid[:h, :w]
    ones = np.ones_like(x)
    coordinates = np.stack([x.ravel(), y.ravel(), ones], axis=1)
    
    # Apply transformation
    transformed = coordinates @ matrix.T
    
    # Normalize homogeneous coordinates
    transformed = transformed[:, :2] / transformed[:, 2:3]
    
    # Reshape to image shape
    x_transformed = transformed[:, 0].reshape(h, w)
    y_transformed = transformed[:, 1].reshape(h, w)
    
    # Determine the bounds of the transformed image
    min_x, max_x = np.min(x_transformed), np.max(x_transformed)
    min_y, max_y = np.min(y_transformed), np.max(y_transformed)
    
    # Calculate new dimensions
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))
    
    # Create output image
    if len(img_array.shape) == 3:
        output = np.zeros((new_h, new_w, img_array.shape[2]), dtype=img_array.dtype)
    else:
        output = np.zeros((new_h, new_w), dtype=img_array.dtype)
    
    # Calculate offset to make all coordinates positive
    offset_x = -min_x
    offset_y = -min_y
    
    # Apply offset
    x_transformed += offset_x
    y_transformed += offset_y
    
    # Clip coordinates to valid range
    x_transformed = np.clip(x_transformed, 0, new_w-1).astype(int)
    y_transformed = np.clip(y_transformed, 0, new_h-1).astype(int)
    
    # Map pixels
    if len(img_array.shape) == 3:
        output[y_transformed, x_transformed] = img_array
    else:
        output[y_transformed, x_transformed] = img_array
    
    # Convert back to PIL image
    return Image.fromarray(output)

# Function to get preset transformations
def get_preset_transformations():
    """
    Get a dictionary of preset transformations
    """
    return {
        "Flip Horizontal": create_transformation_matrix("reflection", axis="y"),
        "Flip Vertical": create_transformation_matrix("reflection", axis="x"),
        "Rotate 90°": create_transformation_matrix("rotation", angle=90),
        "Rotate 180°": create_transformation_matrix("rotation", angle=180),
        "Scale 2x": create_transformation_matrix("scaling", sx=2, sy=2),
        "Scale 0.5x": create_transformation_matrix("scaling", sx=0.5, sy=0.5),
        "Shear Right": create_transformation_matrix("shearing", shx=0.3, shy=0),
        "Shear Up": create_transformation_matrix("shearing", shx=0, shy=-0.3),
    }

# Function for the profile page
def profile_page():
    """
    Display the creator profile page with Development Team section
    """
    st.title("Creator Profile")
    
    # Load and display profile photo
    profile_photo = load_profile_photo()
    
    if profile_photo:
        st.image(profile_photo, width=200)
    else:
        # Manual upload option
        st.write("⚠️ Unable to load profile photo automatically")
        st.write("You can upload your profile photo manually:")
        uploaded_photo = st.file_uploader("Upload Profile Photo", type=["jpg", "jpeg", "png"])
        if uploaded_photo:
            st.image(uploaded_photo, width=200)
    
    # Creator information with HTML
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #2c3e50; margin-bottom: 10px;">Yoseph Sihite</h1>
        <h2 style="color: #34495e; font-weight: 300; margin-bottom: 20px;">Linear Algebra Visionary</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Development Team section - TANPA HTML (hanya st.write)
    st.write("---")
    st.write("## Development Team")
    st.write("💼 Lead Developer")
    st.write("**Name:** Yoseph Sihite")
    st.write("**Student ID:** 004202400113")
    st.write("**Group:** Group 2 Linear Algebra")
    st.write("**Department:** Matematika & Algoritma")
    st.write("")
    st.write("**Development Note:** All development work was completed individually by Yoseph Sihite. No other team members were involved in the creation of this application.")
    st.write("")
    st.write("**Technical Contributions:**")
    st.write("- Matrix transformation algorithms implementation")
    st.write("- Image processing pipeline development")
    st.write("- User interface design and development")
    st.write("- Mathematical engine optimization")
    st.write("- Testing and debugging")
    st.write("- Documentation and deployment")

# Main application function
def main_app():
    """
    Main application for matrix transformation
    """
    st.title("Matrix Transformation Studio")
    st.write("Upload an image and apply matrix transformations to see the effects.")
    
    # Sidebar for transformation controls
    st.sidebar.title("Transformation Controls")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image safely
        original_image = safe_load_image(uploaded_file)
        
        if original_image is not None:
            # Display original image
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
            
            # Transformation options
            transformation_type = st.sidebar.selectbox(
                "Select Transformation",
                ["Translation", "Scaling", "Rotation", "Reflection", "Shearing", "Custom Matrix"]
            )
            
            # Get transformation matrix
            if transformation_type == "Custom Matrix":
                st.sidebar.subheader("Custom 3x3 Matrix")
                matrix_values = []
                for i in range(3):
                    row = []
                    for j in range(3):
                        val = st.sidebar.number_input(
                            f"M[{i+1},{j+1}]",
                            value=1.0 if i == j else 0.0,
                            key=f"m_{i}_{j}"
                        )
                        row.append(val)
                    matrix_values.append(row)
                transformation_matrix = np.array(matrix_values)
            else:
                if transformation_type == "Translation":
                    dx = st.sidebar.slider("X Translation", -100, 100, 0)
                    dy = st.sidebar.slider("Y Translation", -100, 100, 0)
                    transformation_matrix = create_transformation_matrix(
                        "translation", dx=dx, dy=dy
                    )
                elif transformation_type == "Scaling":
                    sx = st.sidebar.slider("X Scale", 0.1, 3.0, 1.0, 0.1)
                    sy = st.sidebar.slider("Y Scale", 0.1, 3.0, 1.0, 0.1)
                    transformation_matrix = create_transformation_matrix(
                        "scaling", sx=sx, sy=sy
                    )
                elif transformation_type == "Rotation":
                    angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
                    transformation_matrix = create_transformation_matrix(
                        "rotation", angle=angle
                    )
                elif transformation_type == "Reflection":
                    axis = st.sidebar.selectbox("Reflection Axis", ["x", "y", "origin"])
                    transformation_matrix = create_transformation_matrix(
                        "reflection", axis=axis
                    )
                elif transformation_type == "Shearing":
                    shx = st.sidebar.slider("X Shear Factor", -1.0, 1.0, 0.0, 0.1)
                    shy = st.sidebar.slider("Y Shear Factor", -1.0, 1.0, 0.0, 0.1)
                    transformation_matrix = create_transformation_matrix(
                        "shearing", shx=shx, shy=shy
                    )
            
            # Display transformation matrix
            st.subheader("Transformation Matrix")
            st.write(transformation_matrix)
            
            # Apply transformation
            transformed_image = apply_transformation(original_image, transformation_matrix)
            
            # Display transformed image
            st.subheader("Transformed Image")
            st.image(transformed_image, use_column_width=True)
            
            # Preset transformations
            st.subheader("Preset Transformations")
            presets = get_preset_transformations()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("Flip Horizontal"):
                    transformed_image = apply_transformation(original_image, presets["Flip Horizontal"])
                    st.image(transformed_image, use_column_width=True)
            
            with col2:
                if st.button("Flip Vertical"):
                    transformed_image = apply_transformation(original_image, presets["Flip Vertical"])
                    st.image(transformed_image, use_column_width=True)
            
            with col3:
                if st.button("Rotate 90°"):
                    transformed_image = apply_transformation(original_image, presets["Rotate 90°"])
                    st.image(transformed_image, use_column_width=True)
            
            with col4:
                if st.button("Rotate 180°"):
                    transformed_image = apply_transformation(original_image, presets["Rotate 180°"])
                    st.image(transformed_image, use_column_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("Scale 2x"):
                    transformed_image = apply_transformation(original_image, presets["Scale 2x"])
                    st.image(transformed_image, use_column_width=True)
            
            with col2:
                if st.button("Scale 0.5x"):
                    transformed_image = apply_transformation(original_image, presets["Scale 0.5x"])
                    st.image(transformed_image, use_column_width=True)
            
            with col3:
                if st.button("Shear Right"):
                    transformed_image = apply_transformation(original_image, presets["Shear Right"])
                    st.image(transformed_image, use_column_width=True)
            
            with col4:
                if st.button("Shear Up"):
                    transformed_image = apply_transformation(original_image, presets["Shear Up"])
                    st.image(transformed_image, use_column_width=True)

# Main navigation
def main():
    """
    Main function with navigation
    """
    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Matrix Transformation Studio", "Creator Profile"])
    
    if page == "Matrix Transformation Studio":
        main_app()
    elif page == "Creator Profile":
        profile_page()

if __name__ == "__main__":
    main()
