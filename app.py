import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import base64

# Try to import pytesseract with error handling
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    # Try to set tesseract path for different environments
    try:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    except:
        pass
except ImportError as e:
    TESSERACT_AVAILABLE = False
    st.error(f"❌ pytesseract import failed: {e}")
    st.info("📋 Debug info: pytesseract module is not available")
except Exception as e:
    TESSERACT_AVAILABLE = False
    st.error(f"❌ pytesseract error: {e}")

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

def basic_image_enhancement(img, method='auto_adaptive'):
    """Apply basic enhancement methods using PIL only"""
    
    # Convert to grayscale if not already
    if img.mode != 'L':
        gray_img = img.convert('L')
    else:
        gray_img = img.copy()
    
    if method == 'high_contrast':
        # Enhance contrast using PIL
        enhancer = ImageEnhance.Contrast(gray_img)
        enhanced = enhancer.enhance(2.0)
        return enhanced.convert('RGB')
    
    elif method == 'sharpening':
        # Apply sharpening filter
        enhanced = gray_img.filter(ImageFilter.SHARPEN)
        # Additional unsharp mask
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        return enhanced.convert('RGB')
    
    elif method == 'brightness_contrast':
        # Enhance both brightness and contrast
        enhancer = ImageEnhance.Brightness(gray_img)
        enhanced = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.5)
        return enhanced.convert('RGB')
    
    elif method == 'edge_enhance':
        # Edge enhancement
        enhanced = gray_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return enhanced.convert('RGB')
    
    elif method == 'smooth_sharpen':
        # Smooth then sharpen
        enhanced = gray_img.filter(ImageFilter.SMOOTH)
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        return enhanced.convert('RGB')
    
    elif method == 'threshold_binary':
        # Simple thresholding
        img_array = np.array(gray_img)
        threshold = np.mean(img_array)
        binary = np.where(img_array > threshold, 255, 0)
        result_img = Image.fromarray(binary.astype(np.uint8), mode='L')
        return result_img.convert('RGB')
    
    elif method == 'auto_adaptive':
        # Comprehensive enhancement using PIL
        # First enhance contrast
        enhancer = ImageEnhance.Contrast(gray_img)
        enhanced = enhancer.enhance(1.5)
        
        # Then enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.8)
        
        # Apply slight smoothing to reduce noise
        enhanced = enhanced.filter(ImageFilter.SMOOTH_MORE)
        
        # Final sharpening
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        
        return enhanced.convert('RGB')
    
    elif method == 'invert_enhance':
        # Invert and enhance (good for dark text on light background)
        inverted = ImageOps.invert(gray_img)
        enhancer = ImageEnhance.Contrast(inverted)
        enhanced = enhancer.enhance(2.0)
        return enhanced.convert('RGB')
    
    else:
        # Default: just convert to RGB
        return gray_img.convert('RGB')

def get_ocr_config(mode, language):
    """Get OCR configuration based on mode"""
    
    # Character sets for different types
    basic_numbers = '0123456789'
    decimal_numbers = '0123456789.,'
    measurement_chars = '0123456789.,-+()[]{}mkcglbftinMKCGLBFTIN°%'
    scientific_chars = '0123456789.,-+eE()[]{}x×*'
    currency_chars = '0123456789.,$€£¥₹₽¢₩₪₦₨₱₡₲₴₵₸₹'
    coordinate_chars = '0123456789.,-+°′″NSEW'
    alphanumeric = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,+-=()[]{}/"\'';

    configs = {
        'numbers_only': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={basic_numbers}',
        'decimal_numbers': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={decimal_numbers}',
        'measurements': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={measurement_chars}',
        'scientific': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={scientific_chars}',
        'currency': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={currency_chars}',
        'coordinates': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={coordinate_chars}',
        'single_word': f'--oem 3 --psm 8 -l {language}',
        'single_char': f'--oem 3 --psm 10 -l {language}',
        'line_of_text': f'--oem 3 --psm 7 -l {language}',
        'mixed_text': f'--oem 3 --psm 6 -l {language}',
        'full_page': f'--oem 3 --psm 3 -l {language}',
        'alphanumeric': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={alphanumeric}'
    }
    return configs.get(mode, configs['decimal_numbers'])

def perform_ocr(img, ocr_mode, language):
    """Perform OCR on the image"""
    if not TESSERACT_AVAILABLE:
        return "❌ Tesseract not available"
    
    try:
        config = get_ocr_config(ocr_mode, language)
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def multiple_preprocessing_ocr(img, ocr_mode, language):
    """Try OCR with multiple preprocessing approaches"""
    if not TESSERACT_AVAILABLE:
        return ["❌ Tesseract not available"]
    
    results = []
    
    try:
        # Original image
        config = get_ocr_config(ocr_mode, language)
        text = pytesseract.image_to_string(img, config=config).strip()
        if text:
            results.append(f"[Original] {text}")
        
        # Convert to grayscale
        gray_img = img.convert('L')
        text = pytesseract.image_to_string(gray_img, config=config).strip()
        if text and f"[Grayscale] {text}" not in results:
            results.append(f"[Grayscale] {text}")
        
        # Try with threshold
        img_array = np.array(gray_img)
        threshold = np.mean(img_array)
        binary = np.where(img_array > threshold, 255, 0)
        binary_img = Image.fromarray(binary.astype(np.uint8), mode='L')
        text = pytesseract.image_to_string(binary_img, config=config).strip()
        if text and f"[Threshold] {text}" not in results:
            results.append(f"[Threshold] {text}")
        
        # Try with inverted colors
        try:
            inverted = ImageOps.invert(gray_img)
            text = pytesseract.image_to_string(inverted, config=config).strip()
            if text and f"[Inverted] {text}" not in results:
                results.append(f"[Inverted] {text}")
        except:
            pass
        
        # Try with scaling
        try:
            width, height = img.size
            # Scale up 2x
            scaled_img = img.resize((width*2, height*2), Image.LANCZOS)
            text = pytesseract.image_to_string(scaled_img, config=config).strip()
            if text and f"[Scaled_2x] {text}" not in results:
                results.append(f"[Scaled_2x] {text}")
        except:
            pass
            
    except Exception as e:
        st.error(f"Preprocessing OCR Error: {str(e)}")
    
    return results

def multiple_ocr_modes(img, language):
    """Try multiple OCR modes"""
    if not TESSERACT_AVAILABLE:
        return ["❌ Tesseract not available"]
    
    results = []
    
    # Priority modes for different content types
    modes = [
        'decimal_numbers', 'numbers_only', 'measurements', 
        'scientific', 'currency', 'coordinates',
        'single_word', 'line_of_text', 'mixed_text', 'alphanumeric'
    ]
    
    for mode in modes:
        try:
            text = perform_ocr(img, mode, language)
            if text and f"[{mode}] {text}" not in results:
                results.append(f"[{mode}] {text}")
        except:
            continue
    
    return results

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🔧 Enhanced OCR Extractor (Streamlit Cloud Compatible)</h1>', unsafe_allow_html=True)
    
    # Check tesseract availability first
    if not TESSERACT_AVAILABLE:
        st.error("❌ **Tesseract OCR is not available!**")
        st.markdown("""
        **To fix this issue:**
        
        1. Make sure your `requirements.txt` contains:
        ```
        streamlit>=1.28.0
        pytesseract==0.3.10
        Pillow>=9.0.0
        numpy>=1.21.0
        ```
        
        2. Make sure your `packages.txt` contains:
        ```
        tesseract-ocr
        tesseract-ocr-eng
        libtesseract-dev
        ```
        
        3. Both files should be in your **repo root directory**
        4. Commit and push both files to GitHub
        5. Wait for Streamlit Cloud to redeploy
        
        If it still doesn't work, try:
        - Clear cache in Streamlit Cloud (⋮ menu → Clear cache)
        - Restart the app
        - Check the deployment logs for specific errors
        """)
        st.stop()
    
    st.success("✅ Tesseract OCR is available!")
    
    # Sidebar configuration
    st.sidebar.header("🎨 Configuration")
    
    # Enhancement method selection
    enhancement_method = st.sidebar.selectbox(
        "Enhancement Method",
        [
            "auto_adaptive", "high_contrast", "sharpening", 
            "brightness_contrast", "edge_enhance", "smooth_sharpen",
            "threshold_binary", "invert_enhance"
        ],
        index=0,
        help="Choose the best enhancement method for your image type"
    )
    
    # OCR mode selection
    ocr_mode = st.sidebar.selectbox(
        "OCR Mode",
        [
            "decimal_numbers", "numbers_only", "measurements", 
            "scientific", "currency", "coordinates",
            "single_word", "single_char", "line_of_text", 
            "mixed_text", "full_page", "alphanumeric"
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
    st.sidebar.subheader("📋 Options")
    use_multiple_preprocessing = st.sidebar.checkbox("Multiple Preprocessing", value=True)
    show_confidence = st.sidebar.checkbox("Show All Results", value=True)
    
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
        
        if st.button("🚀 Enhance & Process", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Enhance image
                    enhanced_img = basic_image_enhancement(original_img, enhancement_method)
                    st.session_state.enhanced_img = enhanced_img
                    st.session_state.enhancement_method = enhancement_method
                    
                    with col2:
                        st.subheader("✨ Enhanced Image")
                        st.image(enhanced_img, caption=f"Enhanced ({enhancement_method})", use_column_width=True)
                    
                    st.success(f"✅ Image enhanced using: {enhancement_method}")
                    
                    # Perform OCR
                    st.subheader("🔍 Step 2: OCR Results")
                    
                    if use_multiple_preprocessing:
                        # Try multiple preprocessing methods
                        results = multiple_preprocessing_ocr(enhanced_img, ocr_mode, language)
                        if results:
                            st.success(f"Found {len(results)} results with different preprocessing methods:")
                            for i, result in enumerate(results, 1):
                                st.write(f"{i}. {result}")
                            
                            # Show best result
                            clean_results = [r.split('] ')[1] for r in results if '] ' in r]
                            if clean_results:
                                best_result = max(clean_results, key=len) if len(set(clean_results)) > 1 else clean_results[0]
                                st.markdown(f"### 🎯 Best Result: `{best_result}`")
                                
                                if st.button("📋 Copy Best Result"):
                                    st.code(best_result, language="text")
                        else:
                            st.warning("No text detected with preprocessing methods")
                    
                    # Try multiple OCR modes
                    if show_confidence:
                        st.subheader("🔄 Multiple OCR Modes")
                        with st.spinner("Trying different OCR modes..."):
                            mode_results = multiple_ocr_modes(enhanced_img, language)
                            if mode_results:
                                with st.expander("View All Mode Results", expanded=False):
                                    for result in mode_results:
                                        st.text(result)
                                
                                # Get unique results
                                unique_results = list(set([r.split('] ')[1] for r in mode_results if '] ' in r and r.split('] ')[1].strip()]))
                                if unique_results:
                                    st.write("**Unique results across all modes:**")
                                    for i, result in enumerate(unique_results, 1):
                                        st.write(f"{i}. `{result}`")
                                    
                                    # Best overall result
                                    overall_best = max(unique_results, key=len)
                                    st.success(f"🏆 **Overall Best:** `{overall_best}`")
                            else:
                                st.info("No additional results from different OCR modes")
                        
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                    st.info("Trying with original image...")
                    try:
                        # Fallback to original image
                        result = perform_ocr(original_img, ocr_mode, language)
                        if result:
                            st.success(f"Fallback result: `{result}`")
                        else:
                            st.warning("No text detected")
                    except Exception as e2:
                        st.error(f"Fallback also failed: {str(e2)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips and information
    st.markdown('<div class="tips-box">', unsafe_allow_html=True)
    st.subheader("💡 Tips for Better Results")
    
    st.markdown("""
    **This version is optimized for Streamlit Cloud compatibility:**
    
    **For Number Recognition:**
    - **Pure Numbers:** Use "numbers_only" mode for just digits (0-9)
    - **Decimal Numbers:** Use "decimal_numbers" for numbers with decimals
    - **Measurements:** Use "measurements" for "12.51m", "3.4kg", etc.
    - **Scientific:** Use "scientific" for "1.5e-3", "2.4×10⁵", etc.
    - **Currency:** Use "currency" for "$123.45", "€99.99", etc.
    
    **Enhancement Methods:**
    - **auto_adaptive:** Best general-purpose enhancement
    - **high_contrast:** For faded or low-contrast text
    - **sharpening:** For blurry text
    - **threshold_binary:** For very clear black/white text
    - **invert_enhance:** For white text on dark background
    
    **General Tips:**
    - Enable "Multiple Preprocessing" for automatic optimization
    - Try different enhancement methods if first attempt fails
    - The app will show the best result automatically
    - Images with clear, high-contrast text work best
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System requirements
    st.info("📝 **Compatible with Streamlit Cloud** - Only requires: `pytesseract`, `Pillow`, and `numpy`")

if __name__ == "__main__":
    main()
