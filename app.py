from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import joblib

app = Flask(__name__)
app.secret_key = 'super_secret_ai_key' 

# ==========================================
# 1. DATABASE SETUP
# ==========================================
def init_db():
    with sqlite3.connect('users.db') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)')
init_db()

# ==========================================
# 2. LOAD AI MODEL (Mac Keras 3 Bypass)
# ==========================================
model = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),
    BatchNormalization(),
    Dropout(0.1),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.05),
    Dense(19, activation='softmax')
])

model.load_weights('crop_prediction_model.h5')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

MOCK_PRICES = {
    'apple': '₹180/kg', 'banana': '₹60/kg', 'blackgram': '₹120/kg', 'chickpea': '₹110/kg', 
    'coconut': '₹40/piece', 'coffee': '₹350/kg', 'grapes': '₹100/kg', 'jute': '₹70/kg', 
    'kidneybeans': '₹140/kg', 'lentil': '₹90/kg', 'mango': '₹200/kg', 'mothbeans': '₹85/kg',
    'mungbean': '₹100/kg', 'muskmelon': '₹50/kg', 'orange': '₹80/kg', 'papaya': '₹45/kg', 
    'pigeonpeas': '₹115/kg', 'pomegranate': '₹160/kg', 'watermelon': '₹30/kg'
}

# ==========================================
# 3. ROUTES & AUTHENTICATION
# ==========================================
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('intro.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    # Explicitly use pbkdf2 to bypass the macOS scrypt error
    password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
    try:
        with sqlite3.connect('users.db') as conn:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        flash('Registration successful! Please log in.', 'success')
    except sqlite3.IntegrityError:
        flash('Username already exists.', 'error')
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    with sqlite3.connect('users.db') as conn:
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if user and check_password_hash(user[2], password):
        session['user_id'] = user[0]
        session['username'] = user[1]
        return redirect(url_for('dashboard'))
    flash('Invalid credentials.', 'error')
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'})
    try:
        data = request.form
        N, P, K = float(data['N']), float(data['P']), float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        urban_area_proximity = float(data['urban_area_proximity'])

        # GUARDRAIL: Extrapolation Error Prevention
        if not (0 <= N <= 250 and 0 <= P <= 250 and 0 <= K <= 250):
            return jsonify({'success': False, 'error': 'NPK values must be realistic (0 - 250).'})
        if not (0 <= temperature <= 60):
            return jsonify({'success': False, 'error': 'Temperature must be between 0°C and 60°C.'})
        if not (0 <= humidity <= 100):
            return jsonify({'success': False, 'error': 'Humidity must be a valid percentage (0% - 100%).'})
        if not (0 <= ph <= 14):
            return jsonify({'success': False, 'error': 'Soil pH must be on the standard scale (0 - 14).'})
        if not (0 <= urban_area_proximity <= 100):
            return jsonify({'success': False, 'error': 'Urban Proximity Index must be between 0 and 100.'})

        npk_sum = N + P + K
        
        features = pd.DataFrame([{
            'N': N, 'P': P, 'K': K,
            'temperature': temperature, 'humidity': humidity, 
            'ph': ph, 'urban_area_proximity': urban_area_proximity, 
            'NPK_sum': npk_sum
        }])

        scaled_features = scaler.transform(features)
        predictions = model.predict(scaled_features)
        
        predicted_class_idx = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        crop_name = le.inverse_transform([predicted_class_idx])[0]

        return jsonify({
            'success': True,
            'crop': crop_name.capitalize(),
            'confidence': f"{confidence:.2f}%",
            'price': MOCK_PRICES.get(crop_name, "N/A")
        })
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid input. Please enter numbers only.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)