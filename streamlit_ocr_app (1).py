import streamlit as st
import streamlit as st
import sys

# Check Python version
st.sidebar.write(f"Python version: {sys.version}")

# Try importing OpenCV with detailed error info
try:
    import cv2
    st.sidebar.success(f"✅ OpenCV imported: {cv2.__version__}")
except ImportError as e:
    st.error("❌ OpenCV import failed!")
    st.error(f"Error details: {str(e)}")
    st.info("Please check that opencv-python-headless is in requirements.txt")
    
    # Try alternative import
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-c", "import cv2; print(cv2.__version__)"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            st.success(f"OpenCV is actually available: {result.stdout}")
        else:
            st.error(f"OpenCV test failed: {result.stderr}")
    except:
        pass
    
    st.stop()

# Continue with other imports only if OpenCV works
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
# ... rest of your imports
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import base64
from scipy import ndimage
from skimage import morphology, exposure, restoration, filters
import matplotlib.pyplot as plt
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Enhanced OCR Number Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
}
.enhancement-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
.result-box {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #dee2e6;
    font-family: monospace;
    white-space: pre-wrap;
}
.tips-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #2196f3;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class EnhancedOCR:
    def __init__(self):
        self.original_image = None
        self.enhanced_image = None
        self.cropped_region = None
        
    def advanced_image_enhancement(self, img, method='auto_adaptive'):
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
            
        else:
            result = gray
        
        # Convert back to RGB
        if len(result.shape) == 2:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(result_rgb)
    
    def get_ocr_config(self, mode, language):
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
    
    def perform_ocr(self, img, ocr_mode, language):
        """Perform OCR on image"""
        try:
            config = self.get_ocr_config(ocr_mode, language)
            text = pytesseract.image_to_string(img, config=config)
            return text.strip()
        except Exception as e:
            return f"OCR Error: {str(e)}"
    
    def multiple_attempts_ocr(self, img, language):
        """Try multiple OCR approaches"""
        results = []
        
        # Try different OCR modes with priority on number recognition
        number_modes = ['numbers_precise', 'measurements', 'scientific_notation', 'currency', 'coordinates']
        other_modes = ['handwriting', 'single_word', 'print', 'mixed']
        
        all_modes = number_modes + other_modes

        # First pass: Try all modes on enhanced image
        for ocr_mode in all_modes:
            try:
                config = self.get_ocr_config(ocr_mode, language)
                text = pytesseract.image_to_string(img, config=config).strip()
                if text and f"[{ocr_mode}] {text}" not in [r[0] for r in results]:
                    results.append((f"[{ocr_mode}] {text}", text))
            except:
                continue

        # Second pass: Try with threshold processing
        try:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            
            threshold_methods = [
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            ]
            
            for thresh_method in threshold_methods:
                try:
                    _, binary = cv2.threshold(gray, 0, 255, thresh_method)
                    binary_img = Image.fromarray(binary)
                    
                    for ocr_mode in number_modes:
                        try:
                            config = self.get_ocr_config(ocr_mode, language)
                            text = pytesseract.image_to_string(binary_img, config=config).strip()
                            result_label = f"[{ocr_mode}_thresh] {text}"
                            if text and result_label not in [r[0] for r in results]:
                                results.append((result_label, text))
                        except:
                            continue
                except:
                    continue
        except:
            pass

        return results

# Initialize the OCR class
@st.cache_resource
def get_ocr_instance():
    return EnhancedOCR()

ocr = get_ocr_instance()

# Main App
st.markdown('<h1 class="main-header">🔢 Enhanced OCR Number Recognition</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    
    # Enhancement method selection
    enhancement_method = st.selectbox(
        "🎨 Enhancement Method",
        [
            "number_optimized",
            "measurement_enhanced", 
            "digit_sharpening",
            "auto_adaptive",
            "handwriting_optimized",
            "high_contrast",
            "noise_reduction",
            "edge_sharpening",
            "brightness_contrast",
            "histogram_equalization",
            "unsharp_masking",
            "morphological"
        ],
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
        }[x]
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
    st.info("""
    **For Best Number Recognition:**
    - Use 'Number Recognition Optimized' enhancement
    - Choose 'Precise Number Recognition' for pure numbers
    - Try 'Multiple Attempts' for difficult cases
    """)

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
                ocr.enhanced_image = ocr.advanced_image_enhancement(ocr.original_image, enhancement_method)
                st.success("✅ Image enhanced successfully!")
                
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
            ["Full Enhanced Image", "Crop Selection", "Multiple Attempts"]
        )
        
        if processing_option == "Full Enhanced Image":
            if st.button("📄 Process Full Enhanced Image"):
                with st.spinner("Performing OCR..."):
                    result = ocr.perform_ocr(ocr.enhanced_image, ocr_mode, language)
                    
                    st.markdown("### 📋 OCR Result:")
                    st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
                    
                    if result and result != "No text detected":
                        st.download_button(
                            "💾 Download Result",
                            result,
                            file_name="ocr_result.txt",
                            mime="text/plain"
                        )
        
        elif processing_option == "Crop Selection":
            st.info("⚠️ Crop selection requires manual implementation in Streamlit. Use the full image processing for now, or consider using the Colab version for crop selection.")
            
        elif processing_option == "Multiple Attempts":
            if st.button("🔄 Try Multiple Methods"):
                with st.spinner("Trying multiple OCR approaches..."):
                    results = ocr.multiple_attempts_ocr(ocr.enhanced_image, language)
                    
                    st.markdown("### 📋 Multiple Attempt Results:")
                    
                    if results:
                        # Create DataFrame for better display
                        df_results = pd.DataFrame(results, columns=["Method", "Text"])
                        
                        for i, (method, text) in enumerate(results):
                            with st.expander(f"Attempt {i+1}: {method}"):
                                st.code(text)
                        
                        # Find best result (longest non-empty)
                        clean_results = [text for _, text in results if text.strip()]
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
                        st.warning("No text detected with any method.")
    
    else:
        st.info("👆 Please upload an image and generate the enhanced version first.")

# Bottom section with tips and information
st.markdown("---")

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown('<div class="tips-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Enhancement Methods Guide")
    st.markdown("""
    - **Number Recognition Optimized**: Best for pure numbers and digits
    - **Measurement Text Enhanced**: Perfect for measurements like '12.51m', '3.4kg'
    - **Digital/Printed Numbers**: Optimized for LCD/LED displays
    - **Auto Adaptive**: Best overall results for mixed content
    - **Handwriting Optimized**: Specifically for handwritten text
    """)
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
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🔢 Enhanced OCR Number Recognition App | Built with Streamlit & OpenCV
</div>
""", unsafe_allow_html=True)
