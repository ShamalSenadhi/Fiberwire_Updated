import streamlit as st
import subprocess, sys

# Ensure cv2 is installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python==4.8.1.78"])

import cv2
# …rest of your imports…

import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from skimage import morphology, exposure, restoration, filters
import io
import base64
import os

# Configure page
st.set_page_config(
    page_title="Enhanced OCR Extractor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .step-header {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .tips-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .result-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def advanced_image_enhancement(img, method='auto_adaptive'):
    """Apply advanced enhancement methods to generate improved image"""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
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
        # Brightness and contrast adjustment
        alpha = 1.3  # Contrast control
        beta = 20    # Brightness control
        result = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    elif method == 'histogram_equalization':
        # Histogram equalization
        result = cv2.equalizeHist(gray)

    elif method == 'unsharp_masking':
        # Unsharp masking for sharpening
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        result = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)

    elif method == 'morphological':
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    elif method == 'wiener_deconvolution':
        # Wiener deconvolution for blur removal
        try:
            psf = np.ones((5, 5)) / 25
            result_float = restoration.wiener(gray, psf, balance=0.1)
            result = (result_float * 255).astype(np.uint8)
        except:
            result = gray

    else:
        result = gray

    # Convert back to RGB
    if len(result.shape) == 2:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)

def get_ocr_config(mode, language):
    """Get OCR configuration based on mode with specialized number recognition"""
    
    # Comprehensive character sets for different number types
    basic_numbers = '0123456789'
    decimal_numbers = '0123456789.,'
    measurement_chars = '0123456789.,-+()[]{}mkcglbftinMKCGLBFTIN°%'
    scientific_chars = '0123456789.,-+eE()[]{}x×*'
    currency_chars = '0123456789.,$€£¥₹₽¢₩₪₦₨₱₡₲₴₵₸₹'
    coordinate_chars = '0123456789.,-+°′″NSEW'
    handwriting_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,+-=()[]{}/"' + "'"

    configs = {
        'numbers_precise': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={decimal_numbers} -c classify_bln_numeric_mode=1',
        'measurements': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={measurement_chars} -c classify_bln_numeric_mode=1',
        'scientific_notation': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={scientific_chars} -c classify_bln_numeric_mode=1',
        'currency': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={currency_chars} -c classify_bln_numeric_mode=1',
        'coordinates': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={coordinate_chars} -c classify_bln_numeric_mode=1',
        'handwriting': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={handwriting_chars}',
        'print': f'--oem 3 --psm 6 -l {language}',
        'mixed': f'--oem 3 --psm 3 -l {language}',
        'numbers': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={measurement_chars}',
        'single_word': f'--oem 3 --psm 8 -l {language}'
    }
    return configs.get(mode, configs['numbers_precise'])

def perform_ocr(img, ocr_mode='numbers_precise', language='eng'):
    """Perform OCR on image"""
    try:
        config = get_ocr_config(ocr_mode, language)
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def crop_image(img, crop_coords):
    """Crop image based on coordinates"""
    if crop_coords is None:
        return img
    
    x, y, w, h = crop_coords
    return img.crop((x, y, x + w, y + h))

def multi_attempt_ocr(img, language='eng'):
    """Try multiple OCR approaches on the image with focus on numbers"""
    results = []
    
    # Try different OCR modes with priority on number recognition
    number_modes = ['numbers_precise', 'measurements', 'scientific_notation', 'currency', 'coordinates']
    other_modes = ['handwriting', 'single_word', 'print', 'mixed']
    all_modes = number_modes + other_modes

    # First pass: Try all modes on original image
    for ocr_mode in all_modes:
        try:
            config = get_ocr_config(ocr_mode, language)
            text = pytesseract.image_to_string(img, config=config).strip()
            if text and text not in [r.split('] ', 1)[1] if '] ' in r else r for r in results]:
                results.append(f"[{ocr_mode}] {text}")
        except:
            continue

    # Second pass: Try with additional processing optimized for numbers
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        # Multiple threshold techniques
        threshold_methods = [
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        ]

        for thresh_method in threshold_methods:
            try:
                _, binary = cv2.threshold(gray, 0, 255, thresh_method)
                binary_img = Image.fromarray(binary)

                # Try number-specific modes on threshold images
                for ocr_mode in number_modes:
                    try:
                        config = get_ocr_config(ocr_mode, language)
                        text = pytesseract.image_to_string(binary_img, config=config).strip()
                        result_text = f"[{ocr_mode}_thresh] {text}"
                        if text and result_text not in results:
                            results.append(result_text)
                    except:
                        continue
            except:
                continue
    except:
        pass

    # Clean up results
    clean_results = []
    for result in results:
        if '] ' in result:
            clean_text = result.split('] ', 1)[1]
            if clean_text and clean_text not in clean_results:
                clean_results.append(clean_text)

    return results, clean_results

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'crop_coords' not in st.session_state:
    st.session_state.crop_coords = None

# Main App
st.markdown('<h1 class="main-header">🔧 Enhanced Image OCR Extractor</h1>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload an image containing text or numbers to extract"
    )
    
    # Enhancement method selection
    enhancement_method = st.selectbox(
        "🎨 Enhancement Method",
        options=[
            'number_optimized', 'measurement_enhanced', 'digit_sharpening',
            'auto_adaptive', 'handwriting_optimized', 'high_contrast',
            'noise_reduction', 'edge_sharpening', 'brightness_contrast',
            'histogram_equalization', 'unsharp_masking', 'morphological',
            'wiener_deconvolution'
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
            'morphological': '🔄 Morphological Enhancement',
            'wiener_deconvolution': '🌟 Wiener Deconvolution'
        }[x],
        help="Choose the best enhancement method for your image type"
    )
    
    # OCR mode selection
    ocr_mode = st.selectbox(
        "🔍 OCR Mode",
        options=[
            'numbers_precise', 'measurements', 'scientific_notation',
            'currency', 'coordinates', 'handwriting', 'print', 'mixed',
            'numbers', 'single_word'
        ],
        format_func=lambda x: {
            'numbers_precise': '🔢 Precise Number Recognition',
            'measurements': '📏 Measurements (12.51m, 3.4kg, etc.)',
            'scientific_notation': '🧪 Scientific Numbers (1.5e-3, etc.)',
            'currency': '💰 Currency & Financial Numbers',
            'coordinates': '🗺️ Coordinates & GPS Numbers',
            'handwriting': '📝 Handwriting Optimized',
            'print': '🖨️ Printed Text',
            'mixed': '🔀 Mixed Text',
            'numbers': '🔢 Basic Numbers',
            'single_word': '📄 Single Word'
        }[x],
        help="Choose the OCR mode that matches your content type"
    )
    
    # Language selection
    language = st.selectbox(
        "🌐 Language",
        options=['eng', 'eng+ara', 'eng+chi_sim', 'eng+fra', 'eng+deu', 'eng+spa', 'eng+rus'],
        format_func=lambda x: {
            'eng': 'English',
            'eng+ara': 'English + Arabic',
            'eng+chi_sim': 'English + Chinese',
            'eng+fra': 'English + French',
            'eng+deu': 'English + German',
            'eng+spa': 'English + Spanish',
            'eng+rus': 'English + Russian'
        }[x],
        help="Select the language(s) for OCR recognition"
    )

# Main content area
if uploaded_file is not None:
    # Load original image
    st.session_state.original_image = Image.open(uploaded_file)
    
    # Step 1: Image Enhancement
    st.markdown('<div class="step-header">🎨 Step 1: Image Enhancement</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Original Image")
        st.image(st.session_state.original_image, use_column_width=True)
        st.info(f"Size: {st.session_state.original_image.size[0]}×{st.session_state.original_image.size[1]}")
    
    with col2:
        if st.button("🚀 Generate Enhanced Image", type="primary", use_container_width=True):
            with st.spinner("Enhancing image..."):
                st.session_state.enhanced_image = advanced_image_enhancement(
                    st.session_state.original_image, 
                    enhancement_method
                )
            st.success("✅ Image enhanced successfully!")
        
        if st.session_state.enhanced_image is not None:
            st.subheader("✨ Enhanced Image")
            st.image(st.session_state.enhanced_image, use_column_width=True)
            st.info(f"Enhancement: {enhancement_method}")

    # Step 2: OCR Processing (only if enhanced image exists)
    if st.session_state.enhanced_image is not None:
        st.markdown('<div class="step-header">🔍 Step 2: OCR Processing</div>', unsafe_allow_html=True)
        
        # Create columns for buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Process Full Enhanced Image", use_container_width=True):
                with st.spinner("Processing full image..."):
                    result = perform_ocr(st.session_state.enhanced_image, ocr_mode, language)
                    
                st.subheader("📋 OCR Results")
                if result:
                    st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
                    st.success(f"✅ Extracted {len(result)} characters")
                    
                    # Download button
                    st.download_button(
                        label="💾 Download Results",
                        data=result,
                        file_name="ocr_results.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("⚠️ No text detected")
        
        with col2:
            if st.button("🔄 Multiple Attempts", use_container_width=True):
                with st.spinner("Trying multiple OCR approaches..."):
                    all_results, clean_results = multi_attempt_ocr(st.session_state.enhanced_image, language)
                
                st.subheader("🎯 Multiple Attempt Results")
                
                if clean_results:
                    # Show all attempts
                    with st.expander("📊 All Attempts", expanded=True):
                        for i, result in enumerate(all_results, 1):
                            st.write(f"**Attempt {i}:** {result}")
                    
                    # Show clean results
                    st.subheader("🏆 Clean Results")
                    for i, result in enumerate(clean_results, 1):
                        st.markdown(f'<div class="result-box">Result {i}: {result}</div>', unsafe_allow_html=True)
                    
                    # Best result
                    best_result = max(clean_results, key=len) if clean_results else ""
                    if best_result:
                        st.markdown(f'<div class="success-box"><strong>🎯 Best Result:</strong> {best_result}</div>', unsafe_allow_html=True)
                        
                        # Download best result
                        st.download_button(
                            label="💾 Download Best Result",
                            data=best_result,
                            file_name="best_ocr_result.txt",
                            mime="text/plain"
                        )
                else:
                    st.warning("⚠️ No text detected in any attempt")
        
        with col3:
            # Image crop functionality (simplified for Streamlit)
            st.info("🎯 For precise text selection, use the coordinate inputs below")
            
            # Coordinate inputs for cropping
            with st.expander("✂️ Crop Settings"):
                crop_x = st.number_input("X coordinate", min_value=0, value=0)
                crop_y = st.number_input("Y coordinate", min_value=0, value=0)
                crop_w = st.number_input("Width", min_value=1, value=100)
                crop_h = st.number_input("Height", min_value=1, value=100)
                
                if st.button("✂️ Process Cropped Area"):
                    crop_coords = (crop_x, crop_y, crop_w, crop_h)
                    cropped_img = crop_image(st.session_state.enhanced_image, crop_coords)
                    
                    # Show cropped image
                    st.image(cropped_img, caption="Cropped Area", use_column_width=True)
                    
                    # Process cropped area
                    with st.spinner("Processing cropped area..."):
                        result = perform_ocr(cropped_img, ocr_mode, language)
                    
                    if result:
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
                        st.success(f"✅ Extracted {len(result)} characters from cropped area")
                    else:
                        st.warning("⚠️ No text detected in cropped area")

# Tips section
st.markdown('<div class="step-header">💡 Tips for Better Results</div>', unsafe_allow_html=True)

tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    <div class="tips-box">
        <h4>🎨 Enhancement Tips:</h4>
        <ul>
            <li><strong>Number Recognition Optimized:</strong> Best for pure numbers and digits</li>
            <li><strong>Measurement Text Enhanced:</strong> Perfect for measurements like "12.51m", "3.4kg"</li>
            <li><strong>Digital/Printed Numbers:</strong> Optimized for LCD/LED displays</li>
            <li><strong>Handwriting Optimized:</strong> Use for handwritten notes</li>
            <li><strong>High Contrast Boost:</strong> For faded or low-contrast text</li>
            <li><strong>Auto Adaptive:</strong> Good overall enhancement for mixed content</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tips_col2:
    st.markdown("""
    <div class="tips-box">
        <h4>🔍 OCR Mode Tips:</h4>
        <ul>
            <li><strong>Precise Number Recognition:</strong> Pure numbers with decimals</li>
            <li><strong>Measurements:</strong> Numbers with units (m, kg, cm, ft, etc.)</li>
            <li><strong>Scientific Numbers:</strong> Scientific notation (1.5e-3, 2×10⁵)</li>
            <li><strong>Currency:</strong> Money amounts ($123.45, €99.99)</li>
            <li><strong>Coordinates:</strong> GPS coordinates (40.7128°N)</li>
            <li><strong>Multiple Attempts:</strong> Try this for best results - tests all methods</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**🔧 Enhanced OCR Extractor** - Optimized for Number Recognition | Built with Streamlit")
