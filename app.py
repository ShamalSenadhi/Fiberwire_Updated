import streamlit as st
import io
import base64
import tempfile
import os

# Handle imports with error checking
try:
    import pytesseract
    # Configure tesseract path for different environments
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        # Try common paths
        possible_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
except ImportError:
    st.error("❌ PyTesseract is not installed. Please check your requirements.txt file.")
    st.stop()

try:
    import cv2
    import numpy as np
except ImportError:
    st.error("❌ OpenCV or NumPy is not installed. Please check your requirements.txt file.")
    st.stop()

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    st.error("❌ Pillow is not installed. Please check your requirements.txt file.")
    st.stop()

try:
    from scipy import ndimage
    from skimage import morphology
except ImportError:
    st.error("❌ SciPy or scikit-image is not installed. Please check your requirements.txt file.")
    st.stop()

def check_tesseract_installation():
    """Check if tesseract is properly installed and configured"""
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract version: {version}"
    except Exception as e:
        return False, f"Tesseract error: {str(e)}"

def check_system_status():
    """Display system status for debugging"""
    with st.expander("🔧 System Status (Click to expand)"):
        # Check Python packages
        st.write("**Python Packages:**")
        packages = {
            'pytesseract': pytesseract.__version__ if 'pytesseract' in globals() else 'Not found',
            'cv2': cv2.__version__ if 'cv2' in globals() else 'Not found',
            'numpy': np.__version__ if 'np' in globals() else 'Not found',
            'PIL': Image.__version__ if 'Image' in globals() else 'Not found'
        }
        for pkg, version in packages.items():
            st.write(f"- {pkg}: {version}")
        
        # Check tesseract
        st.write("**Tesseract OCR:**")
        is_working, message = check_tesseract_installation()
        if is_working:
            st.success(message)
            # Try to get available languages
            try:
                langs = pytesseract.get_languages()
                st.write(f"Available languages: {', '.join(langs)}")
            except:
                st.warning("Could not retrieve available languages")
        else:
            st.error(message)
            st.write("Tesseract command path:", getattr(pytesseract.pytesseract, 'tesseract_cmd', 'Not set'))
        
        # Check OpenCV
        st.write("**OpenCV:**")
        try:
            st.success(f"OpenCV build info available: {len(cv2.getBuildInformation()) > 0}")
        except:
            st.error("OpenCV build info not available")

# Configure page
st.set_page_config(
    page_title="✍️ Handwriting & Text OCR Extractor",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .tips-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .result-box {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
    }
    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_for_handwriting(img, mode='auto'):
    """Apply specialized preprocessing for handwriting recognition"""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if mode == 'high_contrast':
        # Aggressive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif mode == 'noise_reduction':
        # Focus on noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        enhanced = cv2.equalizeHist(denoised)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif mode == 'edge_enhance':
        # Enhance edges for handwriting
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(gray, -1, kernel)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif mode == 'minimal':
        # Minimal processing
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:  # auto
        # Auto mode - comprehensive processing
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        binary = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    return Image.fromarray(binary)

def get_ocr_config(mode, language):
    """Get OCR configuration based on mode"""
    handwriting_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,+-=()[]{}/"' + "'"

    configs = {
        'handwriting': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={handwriting_chars}',
        'print': f'--oem 3 --psm 6 -l {language}',
        'mixed': f'--oem 3 --psm 3 -l {language}',
        'numbers': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist=0123456789.,+-=()[]m',
        'single_word': f'--oem 3 --psm 8 -l {language}'
    }
    return configs.get(mode, configs['handwriting'])

def extract_text_from_image(img, ocr_mode, language, preprocess_mode):
    """Extract text from image using OCR"""
    try:
        # Apply preprocessing
        processed_img = preprocess_for_handwriting(img, preprocess_mode)
        
        # Get OCR config
        config = get_ocr_config(ocr_mode, language)
        
        # Extract text
        text = pytesseract.image_to_string(processed_img, config=config)
        cleaned_text = text.strip()
        
        # If result is poor, try with original image
        if len(cleaned_text) < 2:
            text_original = pytesseract.image_to_string(img, config=config)
            if len(text_original.strip()) > len(cleaned_text):
                cleaned_text = text_original.strip()
        
        return cleaned_text, processed_img
        
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", None

def multi_attempt_ocr(img, language):
    """Try multiple OCR approaches and return all results"""
    results = []
    modes = ['handwriting', 'numbers', 'single_word', 'print']
    preprocess_modes = ['auto', 'high_contrast', 'minimal', 'edge_enhance']
    
    progress_bar = st.progress(0)
    total_attempts = len(modes) * len(preprocess_modes)
    current_attempt = 0
    
    for ocr_mode in modes:
        for prep_mode in preprocess_modes:
            try:
                processed = preprocess_for_handwriting(img, prep_mode)
                config = get_ocr_config(ocr_mode, language)
                text = pytesseract.image_to_string(processed, config=config).strip()
                if text and text not in results:
                    results.append(text)
            except:
                pass
            
            current_attempt += 1
            progress_bar.progress(current_attempt / total_attempts)
    
    progress_bar.empty()
    return results

def crop_image(img, x, y, width, height):
    """Crop image based on coordinates"""
    return img.crop((x, y, x + width, y + height))

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

# Header
st.markdown("<h1 class='main-header'>✍️ Handwriting & Text OCR Extractor</h1>", unsafe_allow_html=True)

# Add system status check
check_system_status()

# Sidebar controls
st.sidebar.header("🔧 OCR Settings")

ocr_mode = st.sidebar.selectbox(
    "OCR Mode",
    options=['handwriting', 'print', 'mixed', 'numbers', 'single_word'],
    format_func=lambda x: {
        'handwriting': '📝 Handwriting Optimized',
        'print': '🖨️ Printed Text',
        'mixed': '🔀 Mixed Text',
        'numbers': '🔢 Numbers/Measurements',
        'single_word': '📄 Single Word'
    }[x]
)

language = st.sidebar.selectbox(
    "Language",
    options=['eng', 'eng+ara', 'eng+chi_sim', 'eng+fra', 'eng+deu', 'eng+spa', 'eng+rus'],
    format_func=lambda x: {
        'eng': 'English',
        'eng+ara': 'English + Arabic',
        'eng+chi_sim': 'English + Chinese',
        'eng+fra': 'English + French',
        'eng+deu': 'English + German',
        'eng+spa': 'English + Spanish',
        'eng+rus': 'English + Russian'
    }[x]
)

preprocess_mode = st.sidebar.selectbox(
    "Preprocessing",
    options=['auto', 'high_contrast', 'noise_reduction', 'edge_enhance', 'minimal'],
    format_func=lambda x: {
        'auto': '🤖 Auto Enhance',
        'high_contrast': '⚡ High Contrast',
        'noise_reduction': '🧹 Noise Reduction',
        'edge_enhance': '📐 Edge Enhancement',
        'minimal': '🎯 Minimal Processing'
    }[x]
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Image Upload")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing handwritten or printed text"
    )
    
    if uploaded_file is not None:
        # Load and display image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Image cropping interface
        st.subheader("✂️ Image Processing Options")
        
        use_full_image = st.checkbox("Process Full Image", value=True)
        
        if not use_full_image:
            st.info("Manual cropping coordinates (pixels from top-left):")
            col_x, col_y, col_w, col_h = st.columns(4)
            with col_x:
                crop_x = st.number_input("X", min_value=0, max_value=img.width, value=0)
            with col_y:
                crop_y = st.number_input("Y", min_value=0, max_value=img.height, value=0)
            with col_w:
                crop_w = st.number_input("Width", min_value=1, max_value=img.width, value=min(200, img.width))
            with col_h:
                crop_h = st.number_input("Height", min_value=1, max_value=img.height, value=min(100, img.height))
            
            # Show crop preview
            if crop_x + crop_w <= img.width and crop_y + crop_h <= img.height:
                cropped_img = crop_image(img, crop_x, crop_y, crop_w, crop_h)
                st.image(cropped_img, caption="Crop Preview", width=300)
                processing_img = cropped_img
            else:
                st.error("Invalid crop coordinates!")
                processing_img = img
        else:
            processing_img = img

with col2:
    st.subheader("🎛️ Actions")
    
    if uploaded_file is not None:
        # Extract Text Button
        if st.button("✍️ Extract Text", type="primary"):
            with st.spinner("Processing..."):
                extracted_text, processed_img = extract_text_from_image(
                    processing_img, ocr_mode, language, preprocess_mode
                )
                st.session_state.extracted_text = extracted_text
                st.session_state.processed_image = processed_img
        
        # Multiple Attempts Button
        if st.button("🔄 Multiple Attempts"):
            with st.spinner("Trying multiple approaches..."):
                results = multi_attempt_ocr(processing_img, language)
                if results:
                    st.session_state.extracted_text = results[0]  # Best result
                    st.success(f"Found {len(results)} different results!")
                    
                    # Show all results
                    st.subheader("📋 All Results:")
                    for i, result in enumerate(results, 1):
                        st.text_area(f"Attempt {i}", result, height=50, key=f"result_{i}")
        
        # Clear Results Button
        if st.button("🗑️ Clear Results"):
            st.session_state.extracted_text = ""
            st.session_state.processed_image = None
    
    # Tips section
    st.markdown("""
    <div class="tips-box">
        <h4>💡 Tips for Better Recognition:</h4>
        <ul>
            <li>Ensure good contrast between text and background</li>
            <li>Try different preprocessing modes if first attempt fails</li>
            <li>Use "Multiple Attempts" for difficult text</li>
            <li>For measurements like "12.51m", use "Numbers/Measurements" mode</li>
            <li>Crop tightly around the text for better results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Results section
if st.session_state.extracted_text or st.session_state.processed_image:
    st.subheader("📄 Results")
    
    if st.session_state.extracted_text:
        st.text_area(
            "Extracted Text",
            st.session_state.extracted_text,
            height=150,
            help="Copy this text to use elsewhere"
        )
        
        # Statistics
        text_stats = {
            "Characters": len(st.session_state.extracted_text),
            "Words": len(st.session_state.extracted_text.split()),
            "Lines": len(st.session_state.extracted_text.split('\n'))
        }
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        col_stats1.metric("Characters", text_stats["Characters"])
        col_stats2.metric("Words", text_stats["Words"])
        col_stats3.metric("Lines", text_stats["Lines"])
    
    else:
        st.warning("No text detected. Try different settings or preprocessing modes.")
    
    # Show processed image
    if st.session_state.processed_image:
        st.subheader("🔍 Processed Image")
        st.image(st.session_state.processed_image, caption="Image after preprocessing", width=400)

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit and Tesseract OCR | "
    "Optimized for handwriting recognition"
)
