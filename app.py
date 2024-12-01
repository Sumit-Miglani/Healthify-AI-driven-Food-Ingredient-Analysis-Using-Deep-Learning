import os
import re
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'SUMIT'  # Set a secret key for session management

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def enhanced_preprocess_image(image):
    # Convert image to grayscale
    image = image.convert('L')
    # Apply a Gaussian blur to reduce noise
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Perform adaptive thresholding
    image = image.point(lambda x: 0 if x < 128 else 255, '1')
    # Resize the image to make text more legible for OCR using LANCZOS
    image = image.resize([2 * dim for dim in image.size], Image.Resampling.LANCZOS)
    return image

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    if image is None:
        return "Error: Could not open the image file."

    preprocessed_image = enhanced_preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed_image, lang='eng')
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def search_ingredient_data(ingredients, datasets):
    results = {name: [] for name in datasets.keys()}
    healthiness_scores = []
    vectorizer = TfidfVectorizer()
    for user_ingredient in ingredients:
        processed_user_ingredient = preprocess_text(user_ingredient)
        for name, dataset in datasets.items():
            dataset_ingredients = dataset['Ingredient'].apply(preprocess_text)
            ingredients_list = dataset_ingredients.tolist()
            ingredients_list.append(processed_user_ingredient)
            tfidf_matrix = vectorizer.fit_transform(ingredients_list)
            cosine_sim = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
            best_match_idx = cosine_sim.argmax()
            best_match_score = cosine_sim[0, best_match_idx]
            if best_match_score > 0.5:
                matched_row = dataset.iloc[best_match_idx]
                match_data = matched_row.drop(['Ingredient'], errors='ignore').to_dict()
                match_data['Ingredient'] = processed_user_ingredient
                results[name].append(match_data)
                if 'HealthinessScore' in matched_row and matched_row['HealthinessScore'] is not None:
                    healthiness_scores.append(matched_row['HealthinessScore'])
    overall_healthiness_score = np.mean(healthiness_scores) if healthiness_scores else 0  # Default to 0 if no scores
    return results, overall_healthiness_score

datasets = {
    'Healthy Ingredients': pd.read_csv('table1.csv'),
    'Unhealthy Ingredients': pd.read_csv('table2.csv'),
    'Food Additives': pd.read_csv('FoodSubstances.csv')
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            text = extract_text_from_image(file_path)
            ingredients = [i.strip() for i in text.split(',')]
            ingredient_data, overall_healthiness_score = search_ingredient_data(ingredients, datasets)
            
            # Store the results in the session
            session['ingredient_data'] = ingredient_data
            session['overall_healthiness_score'] = overall_healthiness_score
            
            # Redirect to the results page
            return redirect(url_for('show_results'))
    else:
        return render_template('upload.html')
    
@app.route('/capture', methods=['GET', 'POST'])

def capture():
    if request.method == 'POST':
        data = request.get_json()
        image_data = data['imageData']
        header, encoded = image_data.split(",", 1)
        data = base64.b64decode(encoded)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.png')
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.png')  # Path for the processed image
        extracted_text_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_text.txt')  # Path for the extracted text file
        
        with open(image_path, 'wb') as f:
            f.write(data)
        
        # Process the image with OCR
        image = Image.open(image_path)
        preprocessed_image = enhanced_preprocess_image(image)
        text = pytesseract.image_to_string(preprocessed_image, lang='eng')
        
        # Save the OCR-processed image
        preprocessed_image.save(processed_image_path)
        
        # Save the extracted text to a file
        with open(extracted_text_path, 'w') as text_file:
            text_file.write(text)
        
        # Process the extracted text for ingredients
        ingredients = [i.strip() for i in text.split(',')]
        ingredient_data, overall_healthiness_score = search_ingredient_data(ingredients, datasets)
        
        # Store the results in the session
        session['ingredient_data'] = ingredient_data
        session['overall_healthiness_score'] = overall_healthiness_score
        
        # Return a JSON response indicating success and redirect to the results page
        return jsonify({'status': 'image captured and processed successfully', 'redirect': url_for('show_results')})
    else:
        return render_template('capture.html')

@app.route('/results')
def show_results():
    # Retrieve the results from the session
    ingredient_data = session.get('ingredient_data', {})
    overall_healthiness_score = session.get('overall_healthiness_score', None)

    # Helper function to capitalize nouns based on a simple heuristic
    def capitalize_nouns(text):
        non_nouns = {"and", "or", "but", "nor", "so", "for", "yet", "after", "although", "as", "because", "before", 
                     "if", "once", "since", "though", "till", "unless", "until", "when", "where", "while", "in", 
                     "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", 
                     "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", 
                     "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", 
                     "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", 
                     "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", 
                     "can", "will", "just", "don", "should", "now"}
        return ' '.join(word if word.lower() in non_nouns else word.capitalize() for word in text.split())

    # Normalize and filter ingredient data
    healthy_ingredients_set = set(capitalize_nouns(entry['Ingredient']) for entry in ingredient_data.get('Healthy Ingredients', []))
    unhealthy_ingredients_set = set(capitalize_nouns(entry['Ingredient']) for entry in ingredient_data.get('Unhealthy Ingredients', []))

    # Filter 'Food Additives' based on 'Healthy Ingredients' and 'Unhealthy Ingredients'
    ingredient_data['Food Additives'] = [
        entry for entry in ingredient_data.get('Food Additives', []) 
        if capitalize_nouns(entry['Ingredient']) not in healthy_ingredients_set 
        and capitalize_nouns(entry['Ingredient']) not in unhealthy_ingredients_set
    ]

    # Filter 'Unhealthy Ingredients' based on 'Healthy Ingredients'
    ingredient_data['Unhealthy Ingredients'] = [
        entry for entry in ingredient_data.get('Unhealthy Ingredients', []) 
        if capitalize_nouns(entry['Ingredient']) not in healthy_ingredients_set
    ]

    for entry in ingredient_data.get('Food Additives', []):
        if 'Effects' in entry:
            effects_words = entry['Effects'].split()
            entry['Effects'] = ' '.join(effects_words[:4]).capitalize()

    # Generate HTML tables with proper formatting for 'Unhealthy Ingredients' and 'Food Additives'
    data_frames_html = {}
    for dataset_name in ['Healthy Ingredients', 'Unhealthy Ingredients', 'Food Additives']:
        data_list = ingredient_data.get(dataset_name, [])
        if not data_list:
            continue  # Skip datasets with no data
        
        # Create DataFrame from the data list
        df = pd.DataFrame(data_list)

        # Special column ordering for 'Unhealthy Ingredients'
        if dataset_name == 'Unhealthy Ingredients':
            df = df[['Ingredient', 'Nutrition', 'Health Risks/Disease Concerned']]

        # Capitalize nouns and format each entry properly
        df = df.applymap(lambda x: capitalize_nouns(x) if isinstance(x, str) else x)

        # Drop 'Healthiness Score' columns if present
        df = df.drop(columns=['Healthiness Score', 'HealthinessScore'], errors='ignore')

        # Convert the DataFrame to HTML
        data_frames_html[dataset_name] = df.to_html(classes='data', border=0, index=False)

    # Render the results template with the data
    return render_template('results.html',
                           data_frames_html=data_frames_html,
                           overall_healthiness_score=overall_healthiness_score)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
    
