import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import base64
from scipy import ndimage
from skimage import morphology, exposure, restoration, filters

# Set page config
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
    }
    .enhancement-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
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
    .tips-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def advanced_image_enhancement(img, method='auto_adaptive'):
    """Apply advanced enhancement methods to improve OCR accuracy"""
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
    
    # Character sets for different number types
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

def perform_ocr(img, ocr_mode, language):
    """Perform OCR on the image"""
    try:
        config = get_ocr_config(ocr_mode, language)
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def multiple_ocr_attempts(img, language):
    """Try multiple OCR approaches with focus on numbers"""
    results = []
    
    # Try different OCR modes with priority on number recognition
    number_modes = ['numbers_precise', 'measurements', 'scientific_notation', 'currency', 'coordinates']
    other_modes = ['handwriting', 'single_word', 'print', 'mixed']
    
    all_modes = number_modes + other_modes
    
    # First pass: Try all modes on original enhanced image
    for ocr_mode in all_modes:
        try:
            config = get_ocr_config(ocr_mode, language)
            text = pytesseract.image_to_string(img, config=config).strip()
            if text and text not in results:
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
                        if text and f"[{ocr_mode}_thresh] {text}" not in results:
                            results.append(f"[{ocr_mode}_thresh] {text}")
                    except:
                        continue
            except:
                continue
    except:
        pass
    
    return results

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🔧 Enhanced Image OCR Extractor</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎨 Configuration")
    
    # Enhancement method selection
    enhancement_method = st.sidebar.selectbox(
        "Enhancement Method",
        [
            "number_optimized", "measurement_enhanced", "digit_sharpening", 
            "auto_adaptive", "handwriting_optimized", "high_contrast",
            "noise_reduction", "edge_sharpening", "brightness_contrast",
            "histogram_equalization", "unsharp_masking", "morphological"
        ],
        index=0,
        help="Choose the best enhancement method for your image type"
    )
    
    # OCR mode selection
    ocr_mode = st.sidebar.selectbox(
        "OCR Mode",
        [
            "numbers_precise", "measurements", "scientific_notation",
            "currency", "coordinates", "handwriting", "print", "mixed",
            "numbers", "single_word"
        ],
        index=0,
        help="Choose the OCR mode optimized for your content type"
    )
    
    # Language selection
    language = st.sidebar.selectbox(
        "Language",
        ["eng", "eng+ara", "eng+chi_sim", "eng+fra", "eng+deu", "eng+spa", "eng+rus"],
        index=0,
        help="Select the language(s) for OCR"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing text or numbers you want to extract"
    )
    
    if uploaded_file is not None:
        # Load original image
        original_img = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_img, caption="Original Image", use_column_width=True)
        
        # Enhancement step
        st.markdown('<div class="enhancement-box">', unsafe_allow_html=True)
        st.subheader("🎨 Step 1: Image Enhancement")
        
        if st.button("🚀 Generate Enhanced Image", type="primary"):
            with st.spinner("Enhancing image..."):
                enhanced_img = advanced_image_enhancement(original_img, enhancement_method)
                st.session_state.enhanced_img = enhanced_img
                st.session_state.enhancement_method = enhancement_method
                st.success(f"✅ Image enhanced using: {enhancement_method}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show enhanced image if available
        if 'enhanced_img' in st.session_state:
            with col2:
                st.subheader("✨ Enhanced Image")
                st.image(st.session_state.enhanced_img, caption="Enhanced Image (Working Image)", use_column_width=True)
            
            # OCR Processing section
            st.subheader("🔍 Step 2: OCR Processing")
            
            # Create columns for different OCR options
            ocr_col1, ocr_col2, ocr_col3 = st.columns(3)
            
            with ocr_col1:
                if st.button("📄 Process Full Enhanced Image", type="secondary"):
                    with st.spinner("Processing full image..."):
                        result = perform_ocr(st.session_state.enhanced_img, ocr_mode, language)
                        st.session_state.ocr_result = result
            
            with ocr_col2:
                if st.button("🔄 Multiple Attempts", type="secondary"):
                    with st.spinner("Trying multiple approaches..."):
                        results = multiple_ocr_attempts(st.session_state.enhanced_img, language)
                        st.session_state.multiple_results = results
            
            with ocr_col3:
                if st.button("🗑️ Clear Results", type="secondary"):
                    if 'ocr_result' in st.session_state:
                        del st.session_state.ocr_result
                    if 'multiple_results' in st.session_state:
                        del st.session_state.multiple_results
            
            # Display OCR results
            if 'ocr_result' in st.session_state:
                st.subheader("📋 OCR Result")
                result_text = st.session_state.ocr_result if st.session_state.ocr_result else "No text detected"
                st.markdown(f'<div class="result-box">{result_text}</div>', unsafe_allow_html=True)
                
                if st.session_state.ocr_result:
                    if st.button("📋 Copy Result"):
                        st.code(st.session_state.ocr_result, language="text")
                        st.success("Result displayed above - you can copy it from the code block")
            
            # Display multiple attempt results
            if 'multiple_results' in st.session_state:
                st.subheader("🔢 Multiple Attempt Results")
                
                # Clean up results and remove method labels for display
                clean_results = []
                for result in st.session_state.multiple_results:
                    if '] ' in result:
                        clean_text = result.split('] ', 1)[1]
                        if clean_text and clean_text not in clean_results:
                            clean_results.append(clean_text)
                
                # Show all attempts
                with st.expander("View All Attempts", expanded=False):
                    for i, result in enumerate(st.session_state.multiple_results, 1):
                        st.text(f"Attempt {i}: {result}")
                
                # Show clean results
                if clean_results:
                    st.write("🎯 **Clean Results:**")
                    for i, result in enumerate(clean_results, 1):
                        st.write(f"{i}. {result}")
                    
                    # Find best result (longest non-empty result)
                    best_result = max(clean_results, key=len, default="")
                    if best_result:
                        st.success(f"🎯 **Best Result:** {best_result}")
                        if st.button("📋 Copy Best Result"):
                            st.code(best_result, language="text")
                            st.success("Best result displayed above - you can copy it from the code block")
    
    # Tips and information
    st.markdown('<div class="tips-box">', unsafe_allow_html=True)
    st.subheader("💡 Tips for Better Results")
    
    st.markdown("""
    **For Number Recognition:**
    - **Pure Numbers:** Use "number_optimized" enhancement + "numbers_precise" OCR
    - **Measurements:** Use "measurement_enhanced" + "measurements" mode for "12.51m", "3.4kg", etc.
    - **Scientific Numbers:** Use "scientific_notation" mode for "1.5e-3", "2.4×10⁵", etc.
    - **Currency:** Use "currency" mode for "$123.45", "€99.99", etc.
    - **GPS/Coordinates:** Use "coordinates" mode for "40.7128°N", etc.
    - **Digital Displays:** Use "digit_sharpening" enhancement for LCD/LED numbers
    
    **General Tips:**
    - Try "Multiple Attempts" - it tests all number recognition methods
    - Use "handwriting_optimized" for handwritten notes
    - Use "high_contrast" for faded or low-contrast text
    - Compare original vs enhanced to see the improvement
    - The enhanced image provides significantly better accuracy
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
