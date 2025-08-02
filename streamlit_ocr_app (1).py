import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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
    """Apply advanced enhancement methods using PIL and numpy"""
    
    # Convert to grayscale if not already
    if img.mode != 'L':
        gray_img = img.convert('L')
    else:
        gray_img = img.copy()
    
    # Convert to numpy array for processing
    img_array = np.array(gray_img)
    
    if method == 'number_optimized':
        # Specialized enhancement for number recognition
        # Noise reduction using scipy
        denoised = ndimage.median_filter(img_array, size=2)
        
        # Enhance contrast
        enhanced = exposure.equalize_adapthist(denoised, clip_limit=0.03)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # Sharpening
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = ndimage.convolve(enhanced, sharpen_kernel)
        result = np.clip(sharpened, 0, 255).astype(np.uint8)

    elif method == 'measurement_enhanced':
        # For measurements like "12.51m", "3.4kg", etc.
        # Gaussian blur for smoothing
        smoothed = ndimage.gaussian_filter(img_array, sigma=0.5)
        
        # Local histogram equalization
        enhanced = exposure.equalize_adapthist(smoothed, clip_limit=0.02)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # Edge enhancement
        edges = filters.sobel(enhanced)
        result = np.clip(enhanced + 0.1 * edges * 255, 0, 255).astype(np.uint8)

    elif method == 'digit_sharpening':
        # For digital/printed numbers with maximum sharpness
        # Unsharp masking
        gaussian = ndimage.gaussian_filter(img_array, sigma=2.0)
        unsharp = img_array + 1.5 * (img_array - gaussian)
        
        # Contrast enhancement
        enhanced = exposure.rescale_intensity(unsharp)
        result = (enhanced * 255).astype(np.uint8)

    elif method == 'auto_adaptive':
        # Comprehensive adaptive enhancement
        # Noise reduction
        denoised = ndimage.median_filter(img_array, size=2)
        
        # Adaptive histogram equalization
        enhanced = exposure.equalize_adapthist(denoised, clip_limit=0.02)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # Slight sharpening
        gaussian = ndimage.gaussian_filter(enhanced, sigma=1.0)
        result = np.clip(enhanced + 0.5 * (enhanced - gaussian), 0, 255).astype(np.uint8)

    elif method == 'handwriting_optimized':
        # Specifically for handwriting
        # Light denoising
        denoised = ndimage.gaussian_filter(img_array, sigma=0.8)
        
        # Contrast enhancement
        enhanced = exposure.rescale_intensity(denoised, out_range=(0, 255))
        
        # Morphological closing to connect broken characters
        from skimage.morphology import disk, closing
        selem = disk(1)
        result = closing(enhanced.astype(np.uint8), selem)

    elif method == 'high_contrast':
        # Maximum contrast enhancement
        enhanced = exposure.equalize_adapthist(img_array, clip_limit=0.05)
        
        # Gamma correction
        gamma = 0.7
        result = (255 * (enhanced ** gamma)).astype(np.uint8)

    elif method == 'noise_reduction':
        # Advanced noise reduction
        denoised1 = ndimage.median_filter(img_array, size=3)
        denoised2 = ndimage.gaussian_filter(denoised1, sigma=1.0)
        result = denoised2.astype(np.uint8)

    elif method == 'edge_sharpening':
        # Edge enhancement and sharpening
        edges = filters.sobel(img_array)
        result = np.clip(img_array + 0.3 * edges * 255, 0, 255).astype(np.uint8)

    elif method == 'brightness_contrast':
        # Brightness and contrast adjustment
        enhanced = exposure.rescale_intensity(img_array, out_range=(20, 235))
        result = enhanced.astype(np.uint8)

    elif method == 'histogram_equalization':
        # Histogram equalization
        result = exposure.equalize_hist(img_array)
        result = (result * 255).astype(np.uint8)

    elif method == 'unsharp_masking':
        # Unsharp masking for sharpening
        gaussian = ndimage.gaussian_filter(img_array, sigma=2.0)
        unsharp = img_array + 2.0 * (img_array - gaussian)
        result = np.clip(unsharp, 0, 255).astype(np.uint8)

    elif method == 'morphological':
        # Morphological operations
        from skimage.morphology import disk, opening, closing
        selem = disk(2)
        opened = opening(img_array, selem)
        result = closing(opened, selem)

    elif method == 'pil_enhanced':
        # Using PIL's built-in enhancement methods
        # Convert back to PIL for PIL operations
        pil_img = Image.fromarray(img_array)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(2.0)
        
        # Apply unsharp mask filter
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        return enhanced.convert('RGB')

    else:
        result = img_array

    # Convert back to PIL Image and then to RGB
    result_img = Image.fromarray(result, mode='L')
    return result_img.convert('RGB')

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
        'single_word': f'--oem 3 --psm 8 -l {language}',
        'single_char': f'--oem 3 --psm 10 -l {language} -c tessedit_char_whitelist={decimal_numbers}',
        'digits_only': f'--oem 3 --psm 7 -l {language} -c tessedit_char_whitelist={basic_numbers}'
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

def perform_ocr_with_preprocessing(img, ocr_mode, language):
    """Perform OCR with additional preprocessing options"""
    results = []
    
    try:
        # Original image
        config = get_ocr_config(ocr_mode, language)
        text = pytesseract.image_to_string(img, config=config).strip()
        if text:
            results.append(f"[Original] {text}")
        
        # Convert to grayscale and try different thresholds
        gray_img = img.convert('L')
        img_array = np.array(gray_img)
        
        # Otsu thresholding using skimage
        try:
            threshold_value = filters.threshold_otsu(img_array)
            binary1 = img_array > threshold_value
            binary1_img = Image.fromarray((binary1 * 255).astype(np.uint8), mode='L').convert('RGB')
            text = pytesseract.image_to_string(binary1_img, config=config).strip()
            if text and text not in [r.split('] ')[1] for r in results]:
                results.append(f"[Otsu] {text}")
        except:
            pass
        
        # Try with inverted colors
        try:
            inverted = ImageOps.invert(gray_img)
            inverted_rgb = inverted.convert('RGB')
            text = pytesseract.image_to_string(inverted_rgb, config=config).strip()
            if text and text not in [r.split('] ')[1] for r in results]:
                results.append(f"[Inverted] {text}")
        except:
            pass
        
        # Try with different scaling
        try:
            # Scale up 2x
            width, height = img.size
            scaled_img = img.resize((width*2, height*2), Image.LANCZOS)
            text = pytesseract.image_to_string(scaled_img, config=config).strip()
            if text and text not in [r.split('] ')[1] for r in results]:
                results.append(f"[Scaled_2x] {text}")
        except:
            pass
            
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
    
    return results

def multiple_ocr_attempts(img, language):
    """Try multiple OCR approaches with focus on numbers"""
    results = []
    
    # Try different OCR modes with priority on number recognition
    number_modes = ['numbers_precise', 'measurements', 'scientific_notation', 'currency', 'coordinates', 'digits_only', 'single_char']
    other_modes = ['handwriting', 'single_word', 'print', 'mixed']
    
    all_modes = number_modes + other_modes
    
    # Try each mode
    for ocr_mode in all_modes:
        ocr_results = perform_ocr_with_preprocessing(img, ocr_mode, language)
        for result in ocr_results:
            method_name = result.split('] ')[0] + ']'
            text = result.split('] ')[1]
            combined_name = f"[{ocr_mode}_{method_name[1:-1]}]"
            full_result = f"{combined_name} {text}"
            if full_result not in results and text:
                results.append(full_result)
    
    return results

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🔧 Enhanced Image OCR Extractor (PIL Version)</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎨 Configuration")
    
    # Enhancement method selection
    enhancement_method = st.sidebar.selectbox(
        "Enhancement Method",
        [
            "number_optimized", "measurement_enhanced", "digit_sharpening", 
            "auto_adaptive", "handwriting_optimized", "high_contrast",
            "noise_reduction", "edge_sharpening", "brightness_contrast",
            "histogram_equalization", "unsharp_masking", "morphological", "pil_enhanced"
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
            "numbers", "single_word", "digits_only", "single_char"
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
    
    # Additional options
    st.sidebar.subheader("📋 Additional Options")
    use_multiple_preprocessing = st.sidebar.checkbox("Use Multiple Preprocessing", value=True, help="Apply different preprocessing techniques")
    show_debug_info = st.sidebar.checkbox("Show Debug Info", value=False, help="Show additional processing information")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing text or numbers you want to extract"
    )
    
    if uploaded_file is not None:
        # Load original image
        original_img = Image.open(uploaded_file)
        
        if show_debug_info:
            st.info(f"Original image size: {original_img.size}, Mode: {original_img.mode}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_img, caption="Original Image", use_column_width=True)
        
        # Enhancement step
        st.markdown('<div class="enhancement-box">', unsafe_allow_html=True)
        st.subheader("🎨 Step 1: Image Enhancement")
        
        if st.button("🚀 Generate Enhanced Image", type="primary"):
            with st.spinner("Enhancing image..."):
                try:
                    enhanced_img = advanced_image_enhancement(original_img, enhancement_method)
                    st.session_state.enhanced_img = enhanced_img
                    st.session_state.enhancement_method = enhancement_method
                    st.success(f"✅ Image enhanced using: {enhancement_method}")
                    
                    if show_debug_info:
                        st.info(f"Enhanced image size: {enhanced_img.size}, Mode: {enhanced_img.mode}")
                        
                except Exception as e:
                    st.error(f"Enhancement error: {str(e)}")
                    # Fallback to original image
                    st.session_state.enhanced_img = original_img.convert('RGB')
                    st.warning("Using original image as fallback")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show enhanced image if available
        if 'enhanced_img' in st.session_state:
            with col2:
                st.subheader("✨ Enhanced Image")
                st.image(st.session_state.enhanced_img, caption=f"Enhanced Image ({st.session_state.enhancement_method})", use_column_width=True)
            
            # OCR Processing section
            st.subheader("🔍 Step 2: OCR Processing")
            
            # Create columns for different OCR options
            ocr_col1, ocr_col2, ocr_col3 = st.columns(3)
            
            with ocr_col1:
                if st.button("📄 Single OCR Mode", type="secondary"):
                    with st.spinner("Processing with single mode..."):
                        try:
                            if use_multiple_preprocessing:
                                results = perform_ocr_with_preprocessing(st.session_state.enhanced_img, ocr_mode, language)
                                st.session_state.single_ocr_results = results
                            else:
                                result = perform_ocr(st.session_state.enhanced_img, ocr_mode, language)
                                st.session_state.single_ocr_result = result
                        except Exception as e:
                            st.error(f"OCR processing error: {str(e)}")
            
            with ocr_col2:
                if st.button("🔄 Multiple OCR Attempts", type="secondary"):
                    with st.spinner("Trying multiple OCR approaches..."):
                        try:
                            results = multiple_ocr_attempts(st.session_state.enhanced_img, language)
                            st.session_state.multiple_results = results
                        except Exception as e:
                            st.error(f"Multiple OCR error: {str(e)}")
            
            with ocr_col3:
                if st.button("🗑️ Clear Results", type="secondary"):
                    # Clear all result variables
                    for key in ['single_ocr_result', 'single_ocr_results', 'multiple_results']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.success("Results cleared!")
            
            # Display single OCR result
            if 'single_ocr_result' in st.session_state:
                st.subheader("📋 Single OCR Result")
                result_text = st.session_state.single_ocr_result if st.session_state.single_ocr_result else "No text detected"
                st.markdown(f'<div class="result-box">{result_text}</div>', unsafe_allow_html=True)
                
                if st.session_state.single_ocr_result:
                    if st.button("📋 Copy Single Result"):
                        st.code(st.session_state.single_ocr_result, language="text")
                        st.success("Result displayed in code block above")
            
            # Display single OCR with preprocessing results
            if 'single_ocr_results' in st.session_state:
                st.subheader("📋 Single Mode with Preprocessing")
                if st.session_state.single_ocr_results:
                    for i, result in enumerate(st.session_state.single_ocr_results, 1):
                        st.write(f"{i}. {result}")
                    
                    # Show best result
                    clean_results = [r.split('] ')[1] for r in st.session_state.single_ocr_results if '] ' in r]
                    if clean_results:
                        best_result = max(clean_results, key=len)
                        st.success(f"🎯 **Best Result:** {best_result}")
                else:
                    st.write("No text detected with any preprocessing method")
            
            # Display multiple attempt results
            if 'multiple_results' in st.session_state:
                st.subheader("🔢 Multiple OCR Attempts")
                
                if st.session_state.multiple_results:
                    # Clean up results
                    clean_results = []
                    for result in st.session_state.multiple_results:
                        if '] ' in result:
                            clean_text = result.split('] ', 1)[1]
                            if clean_text and clean_text not in clean_results:
                                clean_results.append(clean_text)
                    
                    # Show all attempts in expander
                    with st.expander("View All Attempts", expanded=False):
                        for i, result in enumerate(st.session_state.multiple_results, 1):
                            st.text(f"Attempt {i}: {result}")
                    
                    # Show unique results
                    if clean_results:
                        st.write("🎯 **Unique Results Found:**")
                        for i, result in enumerate(clean_results, 1):
                            st.write(f"{i}. `{result}`")
                        
                        # Find best result (longest meaningful result)
                        best_result = max(clean_results, key=lambda x: len(x.strip()))
                        if best_result.strip():
                            st.success(f"🎯 **Best Result:** `{best_result}`")
                            if st.button("📋 Copy Best Result"):
                                st.code(best_result, language="text")
                                st.success("Best result displayed in code block above")
                    else:
                        st.warning("No meaningful text detected")
                else:
                    st.write("No results from multiple attempts")
    
    # Tips and information
    st.markdown('<div class="tips-box">', unsafe_allow_html=True)
    st.subheader("💡 Tips for Better Results")
    
    st.markdown("""
    **This version uses PIL and scikit-image instead of cv2 for better compatibility:**
    
    **For Number Recognition:**
    - **Pure Numbers:** Use "number_optimized" enhancement + "numbers_precise" or "digits_only" OCR
    - **Measurements:** Use "measurement_enhanced" + "measurements" mode for "12.51m", "3.4kg", etc.
    - **Scientific Numbers:** Use "scientific_notation" mode for "1.5e-3", "2.4×10⁵", etc.
    - **Currency:** Use "currency" mode for "$123.45", "€99.99", etc.
    - **Single Characters:** Use "single_char" mode for individual digits
    - **Digital Displays:** Use "digit_sharpening" enhancement for LCD/LED numbers
    
    **New Features:**
    - **Multiple Preprocessing:** Automatically tries different preprocessing techniques
    - **PIL Enhanced:** Uses PIL's built-in enhancement methods
    - **Better Error Handling:** More robust error handling and fallbacks
    - **Debug Info:** Optional debugging information
    
    **General Tips:**
    - Enable "Multiple Preprocessing" for automatic optimization
    - Try "Multiple OCR Attempts" for comprehensive results
    - Use "pil_enhanced" method for general text enhancement
    - The system now handles different image formats better
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System requirements note
    st.info("📝 **Note:** This version requires `pytesseract`, `Pillow`, `numpy`, `scipy`, and `scikit-image`. Install with: `pip install pytesseract Pillow numpy scipy scikit-image`")

if __name__ == "__main__":
    main()
