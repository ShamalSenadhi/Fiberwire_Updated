# Enhanced Image OCR Streamlit App
# File: app.py

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import base64
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
.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class EnhancedOCR:
    def __init__(self):
        self.original_image = None
        self.enhanced_image = None
        self.cropped_region = None
        
    def pil_image_enhancement(self, img, method='auto_adaptive'):
        """Apply PIL-based enhancement methods for improved OCR"""
        
        if method == 'number_optimized':
            # Convert to grayscale
            gray = img.convert('L')
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray)
            contrast_enhanced = enhancer.enhance(2.0)
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(contrast_enhanced)
            sharp = sharpness_enhancer.enhance(2.0)
            # Apply threshold
            threshold = 128
            result = sharp.point(lambda x: 255 if x > threshold else 0, mode='1')
            return result.convert('RGB')
            
        elif method == 'measurement_enhanced':
            # For measurements like "12.51m", "3.4kg", etc.
            gray = img.convert('L')
            # Enhance brightness
            brightness_enhancer = ImageEnhance.Brightness(gray)
            bright = brightness_enhancer.enhance(1.2)
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(bright)
            contrast = contrast_enhancer.enhance(1.8)
            # Sharpen
            sharpness_enhancer = ImageEnhance.Sharpness(contrast)
            result = sharpness_enhancer.enhance(1.5)
            return result.convert('RGB')
            
        elif method == 'digit_sharpening':
            # Maximum sharpness for digital numbers
            gray = img.convert('L')
            # Apply unsharp mask filter
            result = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(2.5)
            return result.convert('RGB')
            
        elif method == 'auto_adaptive':
            # Comprehensive adaptive enhancement
            gray = img.convert('L')
            # Equalize histogram using PIL
            result = ImageOps.equalize(gray)
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.5)
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(result)
            result = sharpness_enhancer.enhance(1.3)
            return result.convert('RGB')
            
        elif method == 'handwriting_optimized':
            # For handwriting
            gray = img.convert('L')
            # Smooth first
            result = gray.filter(ImageFilter.SMOOTH)
            # Then sharpen
            result = result.filter(ImageFilter.SHARPEN)
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.8)
            return result.convert('RGB')
            
        elif method == 'high_contrast':
            # Maximum contrast enhancement
            gray = img.convert('L')
            enhancer = ImageEnhance.Contrast(gray)
            result = enhancer.enhance(3.0)
            return result.convert('RGB')
            
        elif method == 'noise_reduction':
            # Noise reduction using smoothing
            gray = img.convert('L')
            result = gray.filter(ImageFilter.SMOOTH_MORE)
            result = result.filter(ImageFilter.MedianFilter(size=3))
            return result.convert('RGB')
            
        elif method == 'edge_sharpening':
            # Edge enhancement
            gray = img.convert('L')
            result = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.5)
            return result.convert('RGB')
            
        elif method == 'brightness_contrast':
            # Brightness and contrast adjustment
            gray = img.convert('L')
            # Enhance brightness
            brightness_enhancer = ImageEnhance.Brightness(gray)
            bright = brightness_enhancer.enhance(1.3)
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(bright)
            result = contrast_enhancer.enhance(2.0)
            return result.convert('RGB')
            
        elif method == 'histogram_equalization':
            # Histogram equalization
            gray = img.convert('L')
            result = ImageOps.equalize(gray)
            return result.convert('RGB')
            
        elif method == 'unsharp_masking':
            # Unsharp masking
            gray = img.convert('L')
            result = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=2))
            return result.convert('RGB')
            
        elif method == 'morphological':
            # Basic morphological operations using PIL
            gray = img.convert('L')
            # Apply erosion effect by using min filter
            result = gray.filter(ImageFilter.MinFilter(size=3))
            # Apply dilation effect by using max filter
            result = result.filter(ImageFilter.MaxFilter(size=3))
            return result.convert('RGB')
            
        else:
            return img.convert('RGB')
    
    def simulate_ocr(self, img, ocr_mode, language):
        """Simulate OCR results - replace with actual OCR when available"""
        # This is a placeholder for demonstration
        # In a real implementation, you would use pytesseract here
        
        simulated_results = {
            'numbers_precise': "123.45",
            'measurements': "12.51m",
            'scientific_notation': "1.5e-3",
            'currency': "$123.45",
            'coordinates': "40.7128°N",
            'handwriting': "Sample text",
            'print': "Printed text sample",
            'mixed': "Mixed content 123",
            'numbers': "42",
            'single_word': "Word"
        }
        
        return f"[SIMULATED] {simulated_results.get(ocr_mode, 'No text detected')}"
    
    def multiple_attempts_simulation(self, img, language):
        """Simulate multiple OCR attempts"""
        results = []
        
        modes = ['numbers_precise', 'measurements', 'scientific_notation', 'currency', 'coordinates']
        
        for mode in modes:
            result = self.simulate_ocr(img, mode, language)
            results.append((f"[{mode}]", result))
            
        return results

# Initialize the OCR class
@st.cache_resource
def get_ocr_instance():
    return EnhancedOCR()

ocr = get_ocr_instance()

# Main App
st.markdown('<h1 class="main-header">🔢 Enhanced OCR Number Recognition</h1>', unsafe_allow_html=True)

# Warning about dependencies
st.markdown('<div class="warning-box">', unsafe_allow_html=True)
st.warning("""
⚠️ **Dependency Notice**: This demo version uses PIL-based image enhancement instead of OpenCV. 
For full functionality with pytesseract OCR, you'll need to install:
- `pip install opencv-python pytesseract scikit-image`
- Install Tesseract OCR binary on your system

Current version shows image enhancement capabilities with simulated OCR results.
""")
st.markdown('</div>', unsafe_allow_html=True)

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
                ocr.enhanced_image = ocr.pil_image_enhancement(ocr.original_image, enhancement_method)
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
            ["Full Enhanced Image", "Multiple Attempts"]
        )
        
        if processing_option == "Full Enhanced Image":
            if st.button("📄 Process Full Enhanced Image"):
                with st.spinner("Performing OCR..."):
                    result = ocr.simulate_ocr(ocr.enhanced_image, ocr_mode, language)
                    
                    st.markdown("### 📋 OCR Result:")
                    st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
                    
                    if result and "No text detected" not in result:
                        st.download_button(
                            "💾 Download Result",
                            result,
                            file_name="ocr_result.txt",
                            mime="text/plain"
                        )
            
        elif processing_option == "Multiple Attempts":
            if st.button("🔄 Try Multiple Methods"):
                with st.spinner("Trying multiple OCR approaches..."):
                    results = ocr.multiple_attempts_simulation(ocr.enhanced_image, language)
                    
                    st.markdown("### 📋 Multiple Attempt Results:")
                    
                    if results:
                        for i, (method, text) in enumerate(results):
                            with st.expander(f"Attempt {i+1}: {method}"):
                                st.code(text)
                        
                        # Find best result
                        if results:
                            best_result = results[0][1]  # First result as "best"
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

# Installation Instructions
with st.expander("📦 Installation Instructions for Full Functionality"):
    st.markdown("""
    ### To run the full version with actual OCR capabilities:
    
    #### 1. Install Python Dependencies:
    ```bash
    pip install streamlit opencv-python pytesseract scikit-image pillow numpy pandas matplotlib scipy
    ```
    
    #### 2. Install Tesseract OCR:
    
    **Windows:**
    - Download from: https://github.com/UB-Mannheim/tesseract/wiki
    - Add to PATH or set `pytesseract.pytesseract.tesseract_cmd`
    
    **macOS:**
    ```bash
    brew install tesseract
    ```
    
    **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr tesseract-ocr-eng
    ```
    
    #### 3. For additional languages:
    ```bash
    sudo apt install tesseract-ocr-ara tesseract-ocr-chi-sim tesseract-ocr-fra
    ```
    
    #### 4. Update the code to use actual OCR:
    Replace the `simulate_ocr` function with actual pytesseract calls.
    """)

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
    🔢 Enhanced OCR Number Recognition App | Built with Streamlit & PIL (Demo Version)
</div>
""", unsafe_allow_html=True)
