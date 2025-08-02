import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import requests
import json

# Configure Streamlit page
st.set_page_config(
    page_title="🔧 Enhanced OCR Extractor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_dependencies():
    """Check and display dependency status"""
    dependencies = {
        'opencv-python': ('cv2', 'Computer Vision operations'),
        'pillow': ('PIL', 'Image processing'),
        'numpy': ('numpy', 'Numerical operations'),
        'requests': ('requests', 'API calls')
    }
    
    missing = []
    available = []
    
    for package, (module, description) in dependencies.items():
        try:
            __import__(module)
            available.append(f"✅ {package} - {description}")
        except ImportError:
            missing.append(f"❌ {package} - {description}")
    
    return available, missing

def install_easyocr_instructions():
    """Display installation instructions for EasyOCR"""
    st.error("🚨 EasyOCR is not installed!")
    
    with st.expander("📦 Installation Instructions", expanded=True):
        st.markdown("""
        ### Option 1: Install EasyOCR (Recommended)
        ```bash
        pip install easyocr
        ```
        
        ### Option 2: Install with all dependencies
        ```bash
        pip install easyocr opencv-python pillow torch torchvision
        ```
        
        ### Option 3: For Conda users
        ```bash
        conda install -c conda-forge easyocr
        ```
        
        ### If you're on Streamlit Cloud:
        Add `easyocr` to your `requirements.txt` file.
        
        **Note:** EasyOCR requires PyTorch, which might take a few minutes to install.
        """)

def try_import_easyocr():
    """Try to import EasyOCR with helpful error messages"""
    try:
        import easyocr
        return easyocr, None
    except ImportError as e:
        return None, str(e)

# Try to import EasyOCR
easyocr_module, import_error = try_import_easyocr()

@st.cache_resource
def load_ocr_reader(languages=['en']):
    """Initialize EasyOCR reader with specified languages"""
    if easyocr_module is None:
        return None
    
    try:
        return easyocr_module.Reader(languages, gpu=False)
    except Exception as e:
        st.error(f"Error loading OCR reader: {str(e)}")
        return None

def advanced_image_enhancement(img, method='auto_adaptive'):
    """Apply advanced enhancement methods to improve OCR accuracy"""
    try:
        # Convert PIL to OpenCV format
        if isinstance(img, Image.Image):
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            cv_img = img
        
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        if method == 'number_optimized':
            # Specialized enhancement for number recognition
            denoised = cv2.fastNlMeansDenoising(gray, h=12)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6,6))
            enhanced = clahe.apply(denoised)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            smoothed = cv2.bilateralFilter(cleaned, 9, 80, 80)
            binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            result = cv2.dilate(binary, kernel_dilate, iterations=1)

        elif method == 'measurement_enhanced':
            # For measurements like "12.51m", "3.4kg", etc.
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
            enhanced = clahe.apply(denoised)
            kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            result = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)

        elif method == 'digit_sharpening':
            # For digital/printed numbers with maximum sharpness
            denoised = cv2.fastNlMeansDenoising(gray, h=8)
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.5)
            unsharp = cv2.addWeighted(denoised, 2.0, gaussian, -1.0, 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(unsharp)
            laplacian = cv2.Laplacian(enhanced, cv2.CV_64F, ksize=3)
            laplacian = np.uint8(np.absolute(laplacian))
            result = cv2.addWeighted(enhanced, 0.9, laplacian, 0.1, 0)

        elif method == 'auto_adaptive':
            # Comprehensive adaptive enhancement
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
            gaussian = cv2.GaussianBlur(smoothed, (0, 0), 2.0)
            result = cv2.addWeighted(smoothed, 1.5, gaussian, -0.5, 0)

        elif method == 'handwriting_optimized':
            # Specifically for handwriting
            denoised = cv2.fastNlMeansDenoising(gray, h=8)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6,6))
            enhanced = clahe.apply(denoised)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            result = cv2.filter2D(closed, -1, kernel_sharp)

        elif method == 'high_contrast':
            # Maximum contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            gamma = 0.8
            lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(enhanced, lookup_table)

        elif method == 'noise_reduction':
            # Advanced noise reduction
            denoised1 = cv2.fastNlMeansDenoising(gray, h=10)
            denoised2 = cv2.bilateralFilter(denoised1, 9, 80, 80)
            result = cv2.medianBlur(denoised2, 3)

        elif method == 'edge_sharpening':
            # Edge enhancement and sharpening
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            result = cv2.addWeighted(gray, 0.8, laplacian, 0.2, 0)

        elif method == 'brightness_contrast':
            alpha = 1.3  # Contrast control
            beta = 20    # Brightness control
            result = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        elif method == 'histogram_equalization':
            result = cv2.equalizeHist(gray)

        elif method == 'unsharp_masking':
            gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
            result = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)

        elif method == 'morphological':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        else:
            result = gray

        # Convert back to RGB
        if len(result.shape) == 2:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return Image.fromarray(result_rgb)
    
    except Exception as e:
        st.error(f"Enhancement error: {str(e)}")
        return img

def get_easyocr_settings(mode, languages):
    """Get EasyOCR settings based on mode"""
    settings = {
        'allowlist': None,
        'blocklist': None,
        'width_ths': 0.7,
        'height_ths': 0.7,
        'paragraph': False
    }
    
    if mode in ['numbers_precise', 'measurements', 'scientific_notation', 'currency', 'coordinates']:
        settings['allowlist'] = '0123456789.,-+()[]{}mkcglbftinMKCGLBFTIN°%$€£¥₹eE×x*'
        settings['paragraph'] = False
        
    elif mode == 'handwriting':
        settings['paragraph'] = True
        
    return settings

def crop_image(image, x1, y1, x2, y2):
    """Crop image based on coordinates"""
    try:
        # Ensure coordinates are within image bounds
        if isinstance(image, Image.Image):
            width, height = image.size
            x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
            return image.crop((x1, y1, x2, y2))
        else:
            height, width = image.shape[:2]
            x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
            return image[y1:y2, x1:x2]
    except Exception as e:
        st.error(f"Cropping error: {str(e)}")
        return image

def perform_ocr(image, reader, mode='auto', languages=['en']):
    """Perform OCR using EasyOCR"""
    if reader is None:
        return "❌ OCR reader not available. Please install EasyOCR.", []
    
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Get settings for the mode
        settings = get_easyocr_settings(mode, languages)
        
        # Perform OCR
        results = reader.readtext(img_array, **settings)
        
        # Extract text
        extracted_text = []
        confidence_threshold = 0.3
        
        for (bbox, text, confidence) in results:
            if confidence > confidence_threshold:
                extracted_text.append(text.strip())
                
        return ' '.join(extracted_text), results
        
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", []

def simple_ocr_fallback(image):
    """Simple OCR fallback using basic image processing"""
    st.warning("🔄 Using fallback OCR method (limited functionality)")
    
    try:
        # Convert to grayscale and apply basic preprocessing
        if isinstance(image, Image.Image):
            gray = image.convert('L')
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = Image.fromarray(gray)
        
        # Apply some basic enhancements
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # This is a placeholder - in a real scenario, you'd use
        # Tesseract or another OCR engine
        return "⚠️ Install EasyOCR for full OCR functionality", []
        
    except Exception as e:
        return f"Fallback OCR error: {str(e)}", []

def main():
    st.title("🔧 Enhanced Image OCR Extractor")
    st.markdown("**Powered by EasyOCR with Advanced Image Enhancement**")
    
    # Check dependencies
    available_deps, missing_deps = check_dependencies()
    
    # Show dependency status
    with st.expander("📦 Dependency Status"):
        for dep in available_deps:
            st.markdown(dep)
        for dep in missing_deps:
            st.markdown(dep)
    
    # Check EasyOCR availability
    if easyocr_module is None:
        install_easyocr_instructions()
        st.info("⚠️ You can still use the image enhancement features without EasyOCR!")
    else:
        st.success("✅ EasyOCR is available!")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎨 Enhancement Settings")
        
        enhancement_method = st.selectbox(
            "Choose Enhancement Method",
            options=[
                'auto_adaptive', 'number_optimized', 'measurement_enhanced', 
                'digit_sharpening', 'handwriting_optimized', 'high_contrast',
                'noise_reduction', 'edge_sharpening', 'brightness_contrast',
                'histogram_equalization', 'unsharp_masking', 'morphological'
            ],
            format_func=lambda x: {
                'number_optimized': '🔢 Number Recognition Optimized',
                'measurement_enhanced': '📏 Measurement Text Enhanced',
                'digit_sharpening': '🎯 Digital/Printed Numbers',
                'auto_adaptive': '🤖 Auto Adaptive Enhancement',
                'handwriting_optimized': '✍️ Handwriting Optimized',
                'high_contrast': '⚡ High Contrast Boost',
                'noise_reduction': '🧹 Advanced Noise Reduction',
                'edge_sharpening': '📐 Edge Sharpening',
                'brightness_contrast': '💡 Brightness & Contrast',
                'histogram_equalization': '📊 Histogram Equalization',
                'unsharp_masking': '🔍 Unsharp Masking',
                'morphological': '🔄 Morphological Enhancement'
            }[x],
            index=0
        )
        
        if easyocr_module is not None:
            st.header("🔍 OCR Settings")
            
            ocr_mode = st.selectbox(
                "OCR Mode",
                options=[
                    'auto', 'numbers_precise', 'measurements', 'scientific_notation',
                    'currency', 'coordinates', 'handwriting', 'print', 'mixed'
                ],
                format_func=lambda x: {
                    'auto': '🔄 Auto Detection',
                    'numbers_precise': '🔢 Precise Number Recognition',
                    'measurements': '📏 Measurements (12.51m, 3.4kg, etc.)',
                    'scientific_notation': '🧪 Scientific Numbers',
                    'currency': '💰 Currency & Financial',
                    'coordinates': '🗺️ Coordinates & GPS',
                    'handwriting': '📝 Handwriting Optimized',
                    'print': '🖨️ Printed Text',
                    'mixed': '🔀 Mixed Text'
                }[x],
                index=0
            )
            
            languages = st.multiselect(
                "Languages",
                options=['en', 'fr', 'de', 'es', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru'],
                default=['en'],
                help="Select languages for OCR recognition"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Minimum confidence for text detection"
            )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image file for OCR processing"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        try:
            original_image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(original_image, use_column_width=True)
                st.info(f"Size: {original_image.size[0]}×{original_image.size[1]} pixels")
            
            # Generate enhanced image
            if st.button("🚀 Generate Enhanced Image", type="primary"):
                with st.spinner("Enhancing image..."):
                    enhanced_image = advanced_image_enhancement(original_image, enhancement_method)
                    st.session_state.enhanced_image = enhanced_image
                    st.session_state.original_image = original_image
                    st.success("✅ Image enhanced successfully!")
            
            # Display enhanced image if available
            if 'enhanced_image' in st.session_state:
                with col2:
                    st.subheader("✨ Enhanced Image")
                    st.image(st.session_state.enhanced_image, use_column_width=True)
                    st.success(f"Enhanced using: {enhancement_method}")
                
                # OCR Processing section
                st.header("🔍 OCR Processing")
                
                if easyocr_module is not None:
                    # Initialize OCR reader
                    with st.spinner("Loading OCR model..."):
                        reader = load_ocr_reader(languages)
                    
                    if reader is not None:
                        # Create tabs for different processing options
                        tab1, tab2, tab3 = st.tabs(["📄 Full Image OCR", "✂️ Crop & Extract", "🔄 Multiple Attempts"])
                        
                        with tab1:
                            st.subheader("Process Full Enhanced Image")
                            if st.button("🔍 Extract Text from Full Image"):
                                with st.spinner("Performing OCR on full image..."):
                                    extracted_text, results = perform_ocr(
                                        st.session_state.enhanced_image, 
                                        reader, 
                                        ocr_mode, 
                                        languages
                                    )
                                    
                                    if extracted_text and not extracted_text.startswith("❌"):
                                        st.success("✅ Text Extracted Successfully!")
                                        st.text_area("Extracted Text:", extracted_text, height=150)
                                        
                                        # Display confidence scores
                                        if results:
                                            st.subheader("📊 Detection Results")
                                            for i, (bbox, text, confidence) in enumerate(results):
                                                if confidence > confidence_threshold:
                                                    st.write(f"**Text {i+1}:** {text} (Confidence: {confidence:.2f})")
                                    else:
                                        st.warning("No text detected in the image.")
                                        if extracted_text.startswith("❌"):
                                            st.error(extracted_text)
                        
                        with tab2:
                            st.subheader("Crop Selection for Targeted OCR")
                            st.info("Use the sliders below to define a crop area, then extract text from that region.")
                            
                            # Get image dimensions
                            img_width, img_height = st.session_state.enhanced_image.size
                            
                            # Crop controls
                            col_x1, col_y1, col_x2, col_y2 = st.columns(4)
                            
                            with col_x1:
                                x1 = st.slider("Left (X1)", 0, img_width, 0, key="x1")
                            with col_y1:
                                y1 = st.slider("Top (Y1)", 0, img_height, 0, key="y1")
                            with col_x2:
                                x2 = st.slider("Right (X2)", 0, img_width, img_width, key="x2")
                            with col_y2:
                                y2 = st.slider("Bottom (Y2)", 0, img_height, img_height, key="y2")
                            
                            # Show crop preview
                            if x2 > x1 and y2 > y1:
                                cropped_image = crop_image(st.session_state.enhanced_image, x1, y1, x2, y2)
                                st.image(cropped_image, caption=f"Crop Preview ({x2-x1}×{y2-y1} pixels)")
                                
                                if st.button("✍️ Extract Text from Selection"):
                                    with st.spinner("Performing OCR on selected area..."):
                                        extracted_text, results = perform_ocr(
                                            cropped_image, 
                                            reader, 
                                            ocr_mode, 
                                            languages
                                        )
                                        
                                        if extracted_text and not extracted_text.startswith("❌"):
                                            st.success("✅ Text Extracted from Selection!")
                                            st.text_area("Extracted Text:", extracted_text, height=100, key="crop_text")
                                            
                                            # Display confidence scores
                                            if results:
                                                for i, (bbox, text, confidence) in enumerate(results):
                                                    if confidence > confidence_threshold:
                                                        st.write(f"**Text {i+1}:** {text} (Confidence: {confidence:.2f})")
                                        else:
                                            st.warning("No text detected in the selected area.")
                        
                        with tab3:
                            st.subheader("Multiple OCR Attempts")
                            st.info("Try different enhancement and OCR combinations for better results.")
                            
                            if st.button("🔄 Run Multiple Attempts"):
                                methods = ['auto_adaptive', 'number_optimized', 'high_contrast', 'handwriting_optimized']
                                modes = ['auto', 'numbers_precise', 'measurements', 'handwriting']
                                
                                results_data = []
                                
                                progress_bar = st.progress(0)
                                total_attempts = len(methods) * len(modes)
                                attempt = 0
                                
                                for method in methods:
                                    enhanced_img = advanced_image_enhancement(original_image, method)
                                    
                                    for mode in modes:
                                        attempt += 1
                                        progress_bar.progress(attempt / total_attempts)
                                        
                                        try:
                                            text, ocr_results = perform_ocr(enhanced_img, reader, mode, languages)
                                            if text.strip() and not text.startswith("❌"):
                                                results_data.append({
                                                    'Enhancement': method,
                                                    'OCR Mode': mode,
                                                    'Text': text.strip(),
                                                    'Length': len(text.strip()),
                                                    'Words': len(text.strip().split())
                                                })
                                        except Exception as e:
                                            continue
                                
                                progress_bar.empty()
                                
                                if results_data:
                                    st.success(f"✅ Found {len(results_data)} valid results!")
                                    
                                    # Sort by length (longer is usually better)
                                    results_data.sort(key=lambda x: x['Length'], reverse=True)
                                    
                                    # Display results
                                    for i, result in enumerate(results_data, 1):
                                        with st.expander(f"Result {i}: {result['Enhancement']} + {result['OCR Mode']} ({result['Words']} words)"):
                                            st.code(result['Text'])
                                    
                                    # Show best result
                                    best_result = results_data[0]
                                    st.subheader("🎯 Best Result:")
                                    st.success(f"**Method:** {best_result['Enhancement']} + {best_result['OCR Mode']}")
                                    st.text_area("Best Extracted Text:", best_result['Text'], height=100, key="best_text")
                                else:
                                    st.warning("No text detected with any combination.")
                    else:
                        st.error("Failed to load OCR reader.")
                else:
                    # Show fallback options when EasyOCR is not available
                    st.warning("⚠️ EasyOCR not available. Image enhancement is still functional!")
                    
                    tab1, tab2 = st.tabs(["🖼️ Enhanced Image Download", "📝 Manual Text Entry"])
                    
                    with tab1:
                        st.subheader("Download Enhanced Image")
                        st.info("You can download the enhanced image and use it with other OCR tools.")
                        
                        # Convert PIL image to bytes for download
                        buf = io.BytesIO()
                        st.session_state.enhanced_image.save(buf, format='PNG')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download Enhanced Image",
                            data=buf.getvalue(),
                            file_name=f"enhanced_{uploaded_file.name}",
                            mime="image/png"
                        )
                    
                    with tab2:
                        st.subheader("Manual Text Entry")
                        st.info("You can manually type the text you see in the enhanced image.")
                        
                        manual_text = st.text_area(
                            "Enter text from the image:",
                            height=150,
                            placeholder="Type the text you can see in the enhanced image..."
                        )
                        
                        if manual_text:
                            st.success(f"✅ Entered {len(manual_text)} characters")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Installation and tips
    with st.expander("💡 Installation & Tips"):
        st.markdown("""
        ### 📦 Quick Installation
        ```bash
        # Install EasyOCR (recommended)
        pip install easyocr
        
        # Or install all dependencies at once
        pip install easyocr opencv-python pillow streamlit numpy
        ```
        
        ### 🎯 Best Practices for Number Recognition:
        - **Pure Numbers:** Use "Number Recognition Optimized" enhancement + "Precise Number Recognition" OCR
        - **Measurements:** Use "Measurement Text Enhanced" + "Measurements" mode for "12.51m", "3.4kg", etc.
        - **Digital Displays:** Use "Digital/Printed Numbers" enhancement for LCD/LED numbers
        - **Scientific Numbers:** Use "Scientific Numbers" mode for "1.5e-3", "2×10⁵", etc.
        
        ### 🎨 Enhancement Method Guide:
        - **Auto Adaptive:** Best overall results for mixed content
        - **Handwriting Optimized:** Specifically for handwritten text
        - **High Contrast Boost:** For faded or low-contrast images
        - **Noise Reduction:** For noisy/grainy images
        - **Edge Sharpening:** For blurry text
        
        ### 🔍 OCR Tips:
        - Try cropping to focus on specific text areas
        - Use multiple attempts to test different combinations
        - Select appropriate languages for better recognition
        - Adjust confidence threshold based on your needs
        - Compare original vs enhanced images to see improvement
        
        ### 🚨 Troubleshooting:
        - If EasyOCR installation fails, try: `pip install --upgrade pip` first
        - On Windows, you might need Visual Studio Build Tools
        - On Linux, install: `sudo apt-get install python3-dev`
        - For M1 Macs, use: `conda install easyocr -c conda-forge`
        """)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, OpenCV, and EasyOCR")

if __name__ == "__main__":
    main()
