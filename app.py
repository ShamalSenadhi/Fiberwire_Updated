import streamlit as st
import os
import io
import base64
import tempfile

# Set page config first
st.set_page_config(
    page_title="✍️ Handwriting & Text OCR Extractor",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import with better error handling
@st.cache_resource
def load_dependencies():
    """Load all dependencies with proper error handling"""
    errors = []
    modules = {}
    
    try:
        import pytesseract
        import shutil
        # Set tesseract path
        tesseract_cmd = shutil.which('tesseract') or '/usr/bin/tesseract'
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        modules['pytesseract'] = pytesseract
    except Exception as e:
        errors.append(f"PyTesseract: {str(e)}")
    
    try:
        import cv2
        modules['cv2'] = cv2
    except Exception as e:
        errors.append(f"OpenCV: {str(e)}")
    
    try:
        import numpy as np
        modules['np'] = np
    except Exception as e:
        errors.append(f"NumPy: {str(e)}")
    
    try:
        from PIL import Image
        modules['Image'] = Image
    except Exception as e:
        errors.append(f"PIL: {str(e)}")
    
    return modules, errors

# Load dependencies
modules, errors = load_dependencies()

# Check if we have critical errors
if errors:
    st.error("❌ Missing Dependencies:")
    for error in errors:
        st.write(f"- {error}")
    
    st.info("📋 Expected files for Streamlit Cloud deployment:")
    st.code("""
requirements.txt
packages.txt (or apt.txt)
app.py
    """)
    
    st.info("💡 Try redeploying with a fresh commit to trigger package reinstallation.")
    st.stop()

# Extract modules
pytesseract = modules.get('pytesseract')
cv2 = modules.get('cv2')
np = modules.get('np')
Image = modules.get('Image')

# Custom CSS
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
    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_system_status():
    """Check system status"""
    with st.expander("🔧 System Status"):
        st.write("**Tesseract Status:**")
        try:
            version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract {version} is working")
            
            # Check languages
            try:
                langs = pytesseract.get_languages()
                st.write(f"📋 Available languages: {', '.join(langs[:10])}...")
            except:
                st.warning("Could not list languages")
                
        except Exception as e:
            st.error(f"❌ Tesseract error: {str(e)}")
            st.write(f"Tesseract path: {getattr(pytesseract.pytesseract, 'tesseract_cmd', 'Not set')}")
        
        st.write("**Package Versions:**")
        st.write(f"- OpenCV: {cv2.__version__}")
        st.write(f"- NumPy: {np.__version__}")
        st.write(f"- PIL: {Image.__version__}")

def preprocess_image(img, mode='auto'):
    """Preprocess image for better OCR"""
    try:
        # Convert PIL to cv2
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        if mode == 'high_contrast':
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif mode == 'noise_reduction':
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            enhanced = cv2.equalizeHist(denoised)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif mode == 'edge_enhance':
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(gray, -1, kernel)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif mode == 'minimal':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        else:  # auto
            denoised = cv2.fastNlMeansDenoising(gray, h=8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        return Image.fromarray(binary)
    
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return img

def get_ocr_config(mode, language):
    """Get OCR configuration"""
    configs = {
        'handwriting': f'--oem 3 --psm 8 -l {language}',
        'print': f'--oem 3 --psm 6 -l {language}',
        'mixed': f'--oem 3 --psm 3 -l {language}',
        'numbers': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist=0123456789.,+-=()[]m',
        'single_word': f'--oem 3 --psm 8 -l {language}'
    }
    return configs.get(mode, configs['handwriting'])

def extract_text(img, ocr_mode, language, preprocess_mode):
    """Extract text from image"""
    try:
        # Preprocess image
        processed_img = preprocess_image(img, preprocess_mode)
        
        # Get OCR config
        config = get_ocr_config(ocr_mode, language)
        
        # Extract text
        text = pytesseract.image_to_string(processed_img, config=config).strip()
        
        # Fallback to original image if no text found
        if len(text) < 2:
            text_original = pytesseract.image_to_string(img, config=config).strip()
            if len(text_original) > len(text):
                text = text_original
        
        return text, processed_img
        
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", None

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# Header
st.markdown("<h1 class='main-header'>✍️ Handwriting & Text OCR Extractor</h1>", unsafe_allow_html=True)

# System status
check_system_status()

# Sidebar
st.sidebar.header("🔧 Settings")

ocr_mode = st.sidebar.selectbox(
    "OCR Mode",
    ['handwriting', 'print', 'mixed', 'numbers', 'single_word'],
    format_func=lambda x: {
        'handwriting': '📝 Handwriting',
        'print': '🖨️ Print',
        'mixed': '🔀 Mixed',
        'numbers': '🔢 Numbers',
        'single_word': '📄 Single Word'
    }[x]
)

language = st.sidebar.selectbox(
    "Language",
    ['eng', 'eng+ara', 'eng+chi_sim', 'eng+fra', 'eng+deu', 'eng+spa'],
    format_func=lambda x: x.replace('eng', 'English').replace('+', ' + ')
)

preprocess_mode = st.sidebar.selectbox(
    "Preprocessing",
    ['auto', 'high_contrast', 'noise_reduction', 'edge_enhance', 'minimal'],
    format_func=lambda x: x.replace('_', ' ').title()
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("🎛️ Actions")
    
    if uploaded_file:
        if st.button("✍️ Extract Text", type="primary"):
            with st.spinner("Processing..."):
                text, processed = extract_text(img, ocr_mode, language, preprocess_mode)
                st.session_state.extracted_text = text
                st.session_state.processed_image = processed
        
        if st.button("🔄 Multiple Attempts"):
            with st.spinner("Trying different approaches..."):
                results = []
                modes = ['handwriting', 'numbers', 'print']
                prep_modes = ['auto', 'high_contrast', 'minimal']
                
                for mode in modes:
                    for prep in prep_modes:
                        try:
                            text, _ = extract_text(img, mode, language, prep)
                            if text and text not in results:
                                results.append(text)
                        except:
                            continue
                
                if results:
                    st.session_state.extracted_text = results[0]
                    st.success(f"Found {len(results)} results!")
        
        if st.button("🗑️ Clear"):
            st.session_state.extracted_text = ""
            st.session_state.processed_image = None

# Results
if st.session_state.extracted_text or st.session_state.processed_image:
    st.subheader("📄 Results")
    
    if st.session_state.extracted_text:
        st.text_area("Extracted Text", st.session_state.extracted_text, height=150)
        
        # Stats
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        text = st.session_state.extracted_text
        col_stats1.metric("Characters", len(text))
        col_stats2.metric("Words", len(text.split()))
        col_stats3.metric("Lines", len(text.split('\n')))
    else:
        st.warning("No text detected")
    
    if st.session_state.processed_image:
        st.image(st.session_state.processed_image, caption="Processed Image", width=400)

# Tips
st.markdown("""
<div class="tips-box">
    <h4>💡 Tips:</h4>
    <ul>
        <li>Ensure good contrast between text and background</li>
        <li>Try different OCR modes for better results</li>
        <li>Use "Multiple Attempts" for difficult handwriting</li>
        <li>For measurements, use "Numbers" mode</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Tesseract OCR")
