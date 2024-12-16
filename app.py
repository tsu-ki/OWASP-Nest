!pip install paddleocr paddlepaddle easyocr  albumentations opencv-python-headless torch torchvision --quiet
!pip install fuzzywuzzy
import os
import cv2
import json
import torch
import traceback
import numpy as np
import re
from paddleocr import PaddleOCR
from easyocr import Reader
from transformers import AutoTokenizer, AutoModelForCausalLM
from fuzzywuzzy import process

# Fix random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Predefined constants
product_categories = {
    'dairy': ['Ghee'],
    'grains': ['Atta', 'Rice'],
    'sweeteners': ['Sugar'],
    'snacks': ['Biscuit', 'Namkeen'],
    'beverages': ['Tea', 'Cold Drinks'],
    'oils': ['Sunflower Oil', 'Mustard Oil', 'Groundnut Oil', 'Olive Oil'],
    'personal_care': ['Soap', 'Face Wash', 'Shampoo', 'Toothpaste', 'Toothbrush', 'Shaving Cream', 'Hair Oil'],
    'cleaning': ['Detergent'],
    'lentils': ['Dal', 'Toor Dal', 'Moong Dal', 'Chana Dal'],
    'dry_fruits_nuts': ['Dry Fruits', 'Almonds', 'Cashew Nuts', 'Dates & Raisins'],
    'pasta': ['Noodles', 'Pasta'],
    'confectionery': ['Chocolates', 'Sweets & Mithai'],
}

known_brands = [
    "Fortune", "Dabur", "Colgate", "Maggi", "Surf Excel", "Amul", 'Parle',
    "Sunfeast", "Good Day", "Marie Gold", "Lays", "Kurkure", "Bingo",
    "Haldirams", "Tata", "Britannia", "Haldiram's", "Mother Dairy",
    "Patanjali", "NestlÃ©️", "ITC", "Hindustan Unilever", "Godrej",
    "Bisleri", "Cadbury", "Vicco", "Frooti", "Kissan", "MTR", "Lijjat",
    "Nirma", "Boroline", "Everest", "MDH", "Bournvita", "Hajmola",
    "Lifebuoy", "Clinic Plus", "Parachute", "Fevicol", "Pidilite",
    "Santoor", "Vim", "Saffola", "Trust", "Mondelez", "Ananda",
    "AASHIRVAAD", "Uncle Chips", "Madhur", "Uttam", "Tata Tea",
    "Lipton", "Red Label", "Ariel", "Tide", "Dove", "Lux", "Pantene",
    "Head & Shoulders"
]

def extract_price(extracted_text):
    """
    Extract price from the extracted text using regex patterns.
    
    Args:
        extracted_text (str): Text to extract price from
    
    Returns:
        str: Extracted price or "Price Not Found"
    """
    price_patterns = [
        r"\$\d{1,3}(,\d{3})*(\.\d+)?",  
        r"RS.\d{1,3}(,\d{3})*(\.\d+)?",  
        r"MRP\s*Rs\.?\s*(\d+(?:\.\d{2})?)",
        r"(?:Price|MRP)[:.]?\s*(?:Rs\.?|₹)?\s*(\d+(?:\.\d{2})?)"  
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, extracted_text)
        if match:
            return match.group()  

    return "Price Not Found"

def detect_brand(extracted_text, known_brands):
    """
    Detect brand from extracted text using fuzzy matching.
    
    Args:
        extracted_text (str): Text to extract brand from
        known_brands (list): List of known brands to match against
    
    Returns:
        str: Detected brand or "Unknown"
    """
    best_match = process.extractOne(extracted_text, known_brands)
    if best_match and best_match[1] > 60:  # Adjust the score threshold as needed
        return best_match[0]
    return "Unknown"

def extract_product_details(extracted_text):
    """Extract product details from the extracted text."""
    details = {
        "Brand": "Not Available",
        "Expiry Date": "Not Available",
        "MRP": "Not Available",
        "Net Weight": "Not Available",
        "Manufacturer": "Not Available",
        "Storage Conditions": "Not Available",
        "Ingredients": "Not Available",
        "Nutritional Information": "Not Available"
    }

    # Define patterns as strings
    brand_pattern = r"NESTLE|NESCAFE"  # This should be a string
    expiry_pattern = r"BEST BEFORE\s*(\d+\s*MONTHS|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"  # This should be a string
    mrp_pattern = r"MRP\s*[:\s]*([\d\.]+)"  # Adjusted to capture MRP correctly
    net_weight_pattern = [
        r"Net Weight:\s*([\d\.]+g)",
        r"(\d+(\.\d+)?\s*(ml|l|g|kg|gm))",
        r"net\s*wt.*?(\d+(\.\d+)?\s*(ml|l|g|kg|gm))",  # Adjusted for better matching
    ]
    manufacturer_pattern = r"Mfq\.By:\s*([A-Za-z\s]+)"  # This should be a string
    storage_conditions_pattern = r"Store in a cool, dry and hygienic place"  # This should be a string

    # Extracting details using regex
    if re.search(brand_pattern, extracted_text, re.IGNORECASE):
        details["Brand"] = "Nestle"  # or the matched brand

    expiry_match = re.search(expiry_pattern, extracted_text, re.IGNORECASE)
    if expiry_match:
        details["Expiry Date"] = expiry_match.group(1)

    mrp_match = re.search(mrp_pattern, extracted_text, re.IGNORECASE)
    if mrp_match:
        details["MRP"] = mrp_match.group(1)

    # Handle net weight extraction
    for pattern in net_weight_pattern:
        net_weight_match = re.search(pattern, extracted_text, re.IGNORECASE)
        if net_weight_match:
            details["Net Weight"] = net_weight_match.group(1)
            break  # Exit loop after first match

    manufacturer_match = re.search(manufacturer_pattern, extracted_text, re.IGNORECASE)
    if manufacturer_match:
        details["Manufacturer"] = manufacturer_match.group(1)

    if re.search(storage_conditions_pattern, extracted_text, re.IGNORECASE):
        details["Storage Conditions"] = "Store in a cool, dry and hygienic place"

    # Extract ingredients and nutritional information if available
    ingredients_pattern = r"Ingredients:\s*(.*?)(?=NUTRITIONAL|$)"
    nutritional_pattern = r"NUTRITIONAL COMPASSO\s*(.*)"

    ingredients_match = re.search(ingredients_pattern, extracted_text, re.IGNORECASE | re.DOTALL)
    if ingredients_match:
        details["Ingredients"] = ingredients_match.group(1).strip()

    nutritional_match = re.search(nutritional_pattern, extracted_text, re.IGNORECASE)
    if nutritional_match:
        details["Nutritional Information"] = nutritional_match.group(1).strip()

    return details

# Text extraction class with error handling
class TextExtractor:
    def __init__(self, confidence_threshold=0.5):
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.easyocr_reader = Reader(['en'])
        self.confidence_threshold = confidence_threshold

    def extract_text(self, image_path):
        """Extract text from an image using PaddleOCR and EasyOCR with error handling."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # PaddleOCR extraction
            paddle_results = self.paddle_ocr.ocr(image, cls=True) or []
            paddle_text = [
                text[1][0]
                for line in paddle_results for text in line
                if len(text) > 1 and text[1][1] > self.confidence_threshold
            ]

            # EasyOCR extraction
            easyocr_results = self.easyocr_reader.readtext(image) or []
            easyocr_text = [
                text[1]
                for text in easyocr_results
                if text[2] > self.confidence_threshold
            ]
            
            # Combine results
            combined_text = list(set(paddle_text + easyocr_text))
            return combined_text
        except Exception as e:
            print(f"Text extraction error: {e}")
            traceback.print_exc()
            return []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load Dragon-Mistral 0.3 GGUF model and tokenizer
try:
    

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", 
                                               trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        # device_map="auto",  # Automatically distribute model across available devices
        torch_dtype=torch.float16,  # Use float16 for efficiency
        trust_remote_code=True
    )
    model.to(device)

    def generate_product_details(input_text):
        """Generate product details using the language model."""
        # Prepare the input
        # inputs = tokenizer(input_text, return_tensors="pt")
        # Example: Checking and moving inputs to the device
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate text
        outputs = model.generate(
            **inputs.to(device), 
            max_length=4700, 
            num_return_sequences=1, 
            temperature=0.6
        )
        
        # Decode the generated text
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result_text

    def validate_image(image_path):
        """Validate the image format and readability."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read or decode the image: {image_path}")
        return image

    def analyze_image(image_path):
        """Analyze the image to extract product details."""
        try:
            # Validate the input image
            validate_image(image_path)
            
            # Initialize TextExtractor
            extractor = TextExtractor()
            
            # Extract text
            extracted_text = extractor.extract_text(image_path)
            if not extracted_text:
                extracted_text = ["No text extracted"]

            # Join extracted text into a single string
            full_text = ' '.join(extracted_text)

            # Extract additional details
            detected_price = extract_price(full_text)
            detected_brand = detect_brand(full_text, known_brands)

            product_details = extract_product_details(full_text)

            # Prepare input for model
            model_input = f"""Extract comprehensive product details from the provided text, ensuring accurate organization and structure.

Input:
Text: {full_text}

Additional Context:

Detected Brand: {detected_brand}
Detected Price: {detected_price}
Task:
From the provided text, identify and extract the following details, if available:

Brand
Expiry Date
MRP (Maximum Retail Price)
Net Weight
Manufacturer
Storage Conditions
Ingredients
Nutritional Information

Output Format: 
{{
    "Brand": "Extracted Brand",
    "Expiry Date": "Extracted Expiry Date",
    "MRP": "Extracted MRP",
    "Net Weight": "Extracted Net Weight",
    "Manufacturer": "Extracted Manufacturer",
    "Storage Conditions": "Extracted Storage Conditions",
    "Ingredients": "Extracted Ingredients",
    "Nutritional Information": "Extracted Nutritional Information"
}}
If any detail is not found, indicate it as "Not Available".
"""

            # Generate product details
            result_text = generate_product_details(model_input)

            # Debugging: Print the raw output from the model
            print("Raw model output:", result_text)

            try:
                result_dict = json.loads(result_text)  # Use json.loads instead of eval
            except json.JSONDecodeError:
                print("Failed to decode JSON from the model output.")
                result_dict = {
                    "Brand": "Not Available",
                    "Expiry Date": "Not Available",
                    "MRP": "Not Available",
                    "Net Weight": "Not Available",
                    "Manufacturer": "Not Available",
                    "Storage Conditions": "Not Available",
                    "Ingredients": "Not Available",
                    "Nutritional Information": "Not Available"
                }

            return {
                "Filename": os.path.basename(image_path),
                "Extracted Text": full_text,
                "Detected Brand": detected_brand,
                "Detected Price": detected_price,
                "Details": product_details
            }
        except Exception as e:
            print(f"Analysis error: {e}")
            traceback.print_exc()
            return None

    # Test the pipeline on a sample image
    def main():
        sample_image_path = "/kaggle/input/nestle-image/nestle.jpeg"  # Replace with your image path
        result = analyze_image(sample_image_path)
        if result:
            print("Product Information:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print("Analysis failed.")

    if __name__ == "__main__":
        main()

except Exception as e:
    print(f"Model loading error: {e}")
    import traceback
    traceback.print_exc()