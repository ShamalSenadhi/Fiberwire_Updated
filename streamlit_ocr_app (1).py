import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64

# Import handling with better error messages
missing_packages = []

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    missing_packages.append('matplotlib')

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    missing_packages.append('pandas')

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    missing_packages.append('opencv-python-headless')

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    missing_packages.append('pytesseract')

try:
    from scipy import ndimage
    from skimage import morphology, exposure, restoration, filters
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    missing_packages.append('scipy scikit-image')

# Configure page
st.set_page_config(
    page_title="Enhanced OCR Number Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display missing packages warning
if missing_packages:
    st.error(f"""
    ⚠️ **Missing Required Packages**
    
    The following packages are required but not installed:
    - {', '.join(missing_packages)}
    
    **To fix this, create a `requirements.txt` file with the following content:**
    
    ```
    streamlit>=1.28.0
    numpy>=1.24.0
    Pillow>=9.5.0
    opencv-python-headless>=4.8.0
    pytesseract>=0.3.10
    matplotlib>=3.7.0
    pandas>=2.0.0
    scipy>=1.11.0
    scikit-image>=0.21.0
    ```
    
    Then install with: `pip install -r requirements.txt`
    """)
    
    if not CV2_AVAILABLE:
        st.stop()

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.section-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
    border-bottom: 2px solid #ff7f0e;
    padding-bottom: 0.5rem;
}
.enhancement-box {
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.result-box {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid #dee2e6;
    font-family: 'Courier New', monospace;
    white-space: pre-wrap;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.tips-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 4px solid #2196f3;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric-container {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stButton > button {
    background: linear-gradient(135deg, #1f77b4 0%, #1565c0 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(31, 119, 180, 0.3);
}
</style>
""", unsafe_allow_html=True)

class EnhancedOCR:
    def __init__(self):
        self.original_image = None
        self.enhanced_image = None
        self.cropped_region = None
        
    def advanced_image_enhancement(self, img, method='auto_adaptive'):
        """Apply advanced enhancement methods using available libraries"""
        
        if not CV2_AVAILABLE:
            # Fallback to PIL-based enhancement
            return self.pil_based_enhancement(img, method)
        
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
            
        elif method == 'high_contrast':
            # Maximum contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            gamma = 0.8
            lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(enhanced, lookup_table)
            
        elif method == 'auto_adaptive':
            # Comprehensive adaptive enhancement
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
            gaussian = cv2.GaussianBlur(smoothed, (0, 0), 2.0)
            result = cv2.addWeighted(smoothed, 1.5, gaussian, -0.5, 0)
            
        else:
            result = gray
        
        # Convert back to RGB
        if len(result.shape) == 2:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(result_rgb)
    
    def pil_based_enhancement(self, img, method):
        """Fallback enhancement using PIL when OpenCV is not available"""
        
        # Convert to grayscale
        if img.mode != 'L':
            gray_img = img.convert('L')
        else:
            gray_img = img.copy()
        
        if method == 'high_contrast':
            enhancer = ImageEnhance.Contrast(gray_img)
            enhanced = enhancer.enhance(2.0)
            
        elif method == 'brightness_contrast':
            enhancer = ImageEnhance.Brightness(gray_img)
            bright = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Contrast(bright)
            enhanced = enhancer.enhance(1.5)
            
        elif method == 'edge_sharpening':
            enhanced = gray_img.filter(ImageFilter.SHARPEN)
            
        else:  # auto_adaptive
            enhancer = ImageEnhance.Contrast(gray_img)
            contrast_enhanced = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(contrast_enhanced)
            enhanced = enhancer.enhance(1.2)
        
        # Convert back to RGB
        return enhanced.convert('RGB')
    
    def get_ocr_config(self, mode, language):
        """Get OCR configuration based on mode"""
        
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
    
    def perform_ocr(self, img, ocr_mode, language):
        """Perform OCR on image"""
        if not TESSERACT_AVAILABLE:
            return "⚠️ Tesseract OCR not available. Please install pytesseract."
        
        try:
            config = self.get_ocr_config(ocr_mode, language)
            text = pytesseract.image_to_string(img, config=config)
            return text.strip() if text.strip() else "No text detected"
        except Exception as e:
            return f"OCR Error: {str(e)}"
    
    def multiple_attempts_ocr(self, img, language):
        """Try multiple OCR approaches"""
        if not TESSERACT_AVAILABLE:
            return [("⚠️ Tesseract not available", "Please install pytesseract")]
        
        results = []
        number_modes = ['numbers_precise', 'measurements', 'scientific_notation', 'currency', 'coordinates']
        other_modes = ['handwriting', 'single_word', 'print', 'mixed']
        all_modes = number_modes + other_modes

        for ocr_mode in all_modes:
            try:
                config = self.get_ocr_config(ocr_mode, language)
                text = pytesseract.image_to_string(img, config=config).strip()
                if text and f"[{ocr_mode}] {text}" not in [r[0] for r in results]:
                    results.append((f"[{ocr_mode}] {text}", text))
            except Exception as e:
                results.append((f"[{ocr_mode}] Error", f"Error: {str(e)}"))
                continue

        return results if results else [("No results", "No text detected with any method")]

# Initialize the OCR class
@st.cache_resource
def get_ocr_instance():
    return EnhancedOCR()

# Main App
def main():
    ocr = get_ocr_instance()
    
    st.markdown('<h1 class="main-header">🔢 Enhanced OCR Number Recognition</h1>', unsafe_allow_html=True)

    # Display system status
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    
    with col_status1:
        status = "✅ Available" if CV2_AVAILABLE else "❌ Missing"
        st.markdown(f'<div class="metric-container"><strong>OpenCV</strong><br>{status}</div>', unsafe_allow_html=True)
    
    with col_status2:
        status = "✅ Available" if TESSERACT_AVAILABLE else "❌ Missing"
        st.markdown(f'<div class="metric-container"><strong>Tesseract</strong><br>{status}</div>', unsafe_allow_html=True)
    
    with col_status3:
        status = "✅ Available" if SCIPY_AVAILABLE else "❌ Missing"
        st.markdown(f'<div class="metric-container"><strong>SciPy/Skimage</strong><br>{status}</div>', unsafe_allow_html=True)
    
    with col_status4:
        status = "✅ Available" if MATPLOTLIB_AVAILABLE else "❌ Missing"
        st.markdown(f'<div class="metric-container"><strong>Matplotlib</strong><br>{status}</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Configuration")
        
        # Enhancement method selection
        available_methods = [
            "auto_adaptive",
            "high_contrast", 
            "brightness_contrast",
            "edge_sharpening"
        ]
        
        if CV2_AVAILABLE:
            available_methods.extend([
                "number_optimized",
                "measurement_enhanced", 
                "digit_sharpening",
                "handwriting_optimized",
                "noise_reduction",
                "histogram_equalization",
                "unsharp_masking",
                "morphological"
            ])
        
        enhancement_method = st.selectbox(
            "🎨 Enhancement Method",
            available_methods,
            format_func=lambda x: {
                "number_optimized": "🔢 Number Recognition Optimized",
                "measurement_enhanced": "📏 Measurement Text Enhanced",
                "digit_sharpening": "🎯 Digital/Printed Numbers",
                "auto_adaptive": "🤖 Auto Adaptive Enhancement",
                "handwriting_optimized": "✍️ Handwriting Optimized",
                "high_contrast": "⚡ High Contrast Boost",
                "noise_reduction": "🧹 Advanced Noise Reduction",
                "edge_sharpening": "📐 Edge Sharpening",
                "brightness_contrast": "💡 Brightness & Contrast",
                "histogram_equalization": "📊 Histogram Equalization",
                "unsharp_masking": "🔍 Unsharp Masking",
                "morphological": "🔄 Morphological Enhancement"
            }.get(x, x)
        )
        
        # OCR mode selection
        ocr_mode = st.selectbox(
            "🔍 OCR Mode",
            [
                "numbers_precise",
                "measurements", 
                "scientific_notation",
                "currency",
                "coordinates",
                "handwriting",
                "print",
                "mixed",
                "numbers",
                "single_word"
            ],
            format_func=lambda x: {
                "numbers_precise": "🔢 Precise Number Recognition",
                "measurements": "📏 Measurements (12.51m, 3.4kg, etc.)",
                "scientific_notation": "🧪 Scientific Numbers (1.5e-3, etc.)",
                "currency": "💰 Currency & Financial Numbers",
                "coordinates": "🗺️ Coordinates & GPS Numbers",
                "handwriting": "📝 Handwriting Optimized",
                "print": "🖨️ Printed Text",
                "mixed": "🔀 Mixed Text",
                "numbers": "🔢 Basic Numbers",
                "single_word": "📄 Single Word"
            }[x]
        )
        
        # Language selection
        language = st.selectbox(
            "🌐 Language",
            ["eng", "eng+ara", "eng+chi_sim", "eng+fra", "eng+deu", "eng+spa", "eng+rus"],
            format_func=lambda x: {
                "eng": "English",
                "eng+ara": "English + Arabic",
                "eng+chi_sim": "English + Chinese",
                "eng+fra": "English + French", 
                "eng+deu": "English + German",
                "eng+spa": "English + Spanish",
                "eng+rus": "English + Russian"
            }[x]
        )
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        tip_text = """
        **For Best Number Recognition:**
        - Upload clear, high-resolution images
        - Ensure good contrast between text and background
        - Try 'Multiple Attempts' for difficult cases
        """
        
        if CV2_AVAILABLE:
            tip_text += "\n- Use 'Number Recognition Optimized' for pure numbers"
        
        st.info(tip_text)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h2 class="section-header">📤 Upload & Enhance</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing numbers or text to extract"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            ocr.original_image = Image.open(uploaded_file)
            
            st.markdown("**Original Image:**")
            st.image(ocr.original_image, caption="Original Image", use_column_width=True)
            
            # Enhancement button
            if st.button("🚀 Generate Enhanced Image", type="primary"):
                with st.spinner("Enhancing image..."):
                    try:
                        ocr.enhanced_image = ocr.advanced_image_enhancement(ocr.original_image, enhancement_method)
                        st.success("✅ Image enhanced successfully!")
                    except Exception as e:
                        st.error(f"Enhancement failed: {str(e)}")
                        
            # Display enhanced image if available
            if ocr.enhanced_image:
                st.markdown("**Enhanced Image (Working Image):**")
                st.image(ocr.enhanced_image, caption=f"Enhanced using: {enhancement_method}", use_column_width=True)

    with col2:
        st.markdown('<h2 class="section-header">🔍 OCR Processing</h2>', unsafe_allow_html=True)
        
        if ocr.enhanced_image:
            # Processing options
            processing_option = st.radio(
                "Select processing option:",
                ["Full Enhanced Image", "Multiple Attempts"]
            )
            
            if processing_option == "Full Enhanced Image":
                if st.button("📄 Process Full Enhanced Image"):
                    with st.spinner("Performing OCR..."):
                        result = ocr.perform_ocr(ocr.enhanced_image, ocr_mode, language)
                        
                        st.markdown("### 📋 OCR Result:")
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
                        
                        if result and result != "No text detected" and not result.startswith("⚠️"):
                            st.download_button(
                                "💾 Download Result",
                                result,
                                file_name="ocr_result.txt",
                                mime="text/plain"
                            )
            
            elif processing_option == "Multiple Attempts":
                if st.button("🔄 Try Multiple Methods"):
                    with st.spinner("Trying multiple OCR approaches..."):
                        results = ocr.multiple_attempts_ocr(ocr.enhanced_image, language)
                        
                        st.markdown("### 📋 Multiple Attempt Results:")
                        
                        if results and PANDAS_AVAILABLE:
                            # Create DataFrame for better display
                            df_results = pd.DataFrame(results, columns=["Method", "Text"])
                            st.dataframe(df_results, use_container_width=True)
                        
                        for i, (method, text) in enumerate(results):
                            with st.expander(f"Attempt {i+1}: {method}"):
                                st.code(text)
                        
                        # Find best result (longest non-empty)
                        clean_results = [text for _, text in results if text.strip() and not text.startswith("Error")]
                        if clean_results:
                            best_result = max(clean_results, key=len)
                            st.markdown("### 🎯 Best Result:")
                            st.markdown(f'<div class="result-box">{best_result}</div>', unsafe_allow_html=True)
                            
                            st.download_button(
                                "💾 Download Best Result",
                                best_result,
                                file_name="best_ocr_result.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("No valid text detected with any method.")
        
        else:
            st.info("👆 Please upload an image and generate the enhanced version first.")

    # Bottom section with tips and information
    st.markdown("---")

    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown('<div class="tips-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Enhancement Methods Guide")
        enhancement_guide = """
        - **Auto Adaptive**: Best overall results for mixed content
        - **High Contrast**: Increases contrast for faded text
        - **Brightness & Contrast**: Adjusts overall image brightness
        - **Edge Sharpening**: Enhances text edges
        """
        
        if CV2_AVAILABLE:
            enhancement_guide += """
        - **Number Recognition Optimized**: Best for pure numbers and digits
        - **Measurement Text Enhanced**: Perfect for measurements like '12.51m', '3.4kg'
        - **Digital/Printed Numbers**: Optimized for LCD/LED displays
        - **Handwriting Optimized**: Specifically for handwritten text
        """
        
        st.markdown(enhancement_guide)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="tips-box">', unsafe_allow_html=True)
        st.markdown("### 🔢 OCR Modes Guide")
        st.markdown("""
        - **Precise Number Recognition**: Pure numbers with decimals
        - **Measurements**: Numbers with units (m, kg, cm, ft, etc.)
        - **Scientific Numbers**: Scientific notation (1.5e-3, 2×10⁵)
        - **Currency**: Money amounts ($123.45, €99.99, ¥1000)
        - **Coordinates**: GPS coordinates (40.7128°N, -74.0060°W)
        - **Handwriting**: Optimized for handwritten text
        - **Print**: Standard printed text
        - **Mixed**: Mixed content types
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🔢 Enhanced OCR Number Recognition App | Built with Streamlit & OpenCV<br>
        <small>Install missing packages using the requirements.txt file above</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
