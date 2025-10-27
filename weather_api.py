from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import json
import os

app = Flask(__name__)
CORS(app)

# Load weather dataset from CSV file
def load_weather_dataset():
    csv_file = 'weather_dataset_3500_records.csv'
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded dataset with {len(df)} records from '{csv_file}'")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            print("Generating new dataset...")
    else:
        print(f"⚠️  CSV file '{csv_file}' not found. Generating new dataset...")
    
    return generate_weather_dataset()

# Generate synthetic weather dataset as fallback
def generate_weather_dataset():
    np.random.seed(42)
    n_records = 3500
    
    cities = ['New York', 'London', 'Tokyo', 'Mumbai', 'Sydney', 'Paris', 'Berlin', 'Toronto', 'Dubai', 'Singapore']
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    
    print("🔄 Generating weather dataset...")
    
    # City base temperature adjustments
    city_temp_offset = {
        'New York': 0, 'London': -2, 'Tokyo': 2, 'Mumbai': 10,
        'Sydney': 5, 'Paris': -1, 'Berlin': -3, 'Toronto': -4,
        'Dubai': 15, 'Singapore': 12
    }
    
    data = {
        'record_id': range(1, n_records + 1),
        'city': np.random.choice(cities, n_records),
        'season': np.random.choice(seasons, n_records),
        'humidity': np.random.randint(30, 95, n_records),
        'pressure': np.random.randint(980, 1030, n_records),
        'wind_speed': np.random.randint(0, 50, n_records),
        'cloud_cover': np.random.randint(0, 100, n_records),
    }
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=5*365)
    dates = [start_date + timedelta(days=int(i * 5)) for i in range(n_records)]
    data['date'] = [d.strftime('%Y-%m-%d') for d in dates]
    
    # Generate temperature and conditions
    temps = []
    conditions = []
    precipitation = []
    feels_like = []
    uv_index = []
    visibility = []
    
    for i in range(n_records):
        # Base temperature by season
        season_temp = {
            'Winter': np.random.randint(-5, 15),
            'Spring': np.random.randint(10, 25),
            'Summer': np.random.randint(20, 40),
            'Fall': np.random.randint(5, 20)
        }[data['season'][i]]
        
        # Add city offset and correlations
        city_offset = city_temp_offset[data['city'][i]]
        temp = season_temp + city_offset
        temp += (data['humidity'][i] - 60) * 0.1
        temp += (data['wind_speed'][i] - 25) * 0.05
        temp += (data['pressure'][i] - 1000) * 0.02
        temp = round(temp + np.random.randn() * 3, 1)
        temp = max(-20, min(50, temp))
        temps.append(temp)
        
        # Weather condition
        if data['humidity'][i] > 80 and data['cloud_cover'][i] > 70 and temp < 35:
            condition = 'Rainy'
        elif data['cloud_cover'][i] > 85:
            condition = 'Cloudy'
        elif data['cloud_cover'][i] < 20:
            condition = 'Sunny'
        elif data['cloud_cover'][i] < 50:
            condition = 'Partly Cloudy'
        else:
            condition = np.random.choice(['Cloudy', 'Partly Cloudy'], p=[0.6, 0.4])
        conditions.append(condition)
        
        # Precipitation probability
        if condition == 'Rainy':
            precip = np.random.randint(60, 100)
        elif condition == 'Cloudy':
            precip = np.random.randint(20, 60)
        elif condition == 'Partly Cloudy':
            precip = np.random.randint(5, 30)
        else:
            precip = np.random.randint(0, 10)
        precipitation.append(precip)
        
        # Feels like temperature
        feel = temp
        if data['wind_speed'][i] > 20:
            feel -= (data['wind_speed'][i] - 20) * 0.2
        if data['humidity'][i] > 70 and temp > 20:
            feel += (data['humidity'][i] - 70) * 0.15
        feels_like.append(round(feel, 1))
        
        # UV index
        if condition == 'Sunny':
            uv = np.random.randint(7, 12)
        elif condition == 'Partly Cloudy':
            uv = np.random.randint(4, 8)
        elif condition == 'Cloudy':
            uv = np.random.randint(1, 5)
        else:
            uv = np.random.randint(0, 3)
        uv_index.append(uv)
        
        # Visibility
        if condition == 'Rainy':
            vis = np.random.randint(2, 8)
        elif condition == 'Cloudy':
            vis = np.random.randint(5, 15)
        else:
            vis = np.random.randint(10, 30)
        visibility.append(vis)
    
    data['temperature'] = temps
    data['feels_like_temperature'] = feels_like
    data['condition'] = conditions
    data['precipitation_probability'] = precipitation
    data['uv_index'] = uv_index
    data['visibility_km'] = visibility
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_file = 'weather_dataset_3500_records.csv'
    df.to_csv(csv_file, index=False)
    print(f"💾 Generated and saved dataset with {len(df)} records to '{csv_file}'")
    
    return df

# Initialize dataset
print("\n" + "="*60)
print("🌤️  WEATHER PREDICTION API - INITIALIZING")
print("="*60)

df = load_weather_dataset()

# Prepare encoders
le_city = LabelEncoder()
le_season = LabelEncoder()
le_condition = LabelEncoder()

df['city_encoded'] = le_city.fit_transform(df['city'])
df['season_encoded'] = le_season.fit_transform(df['season'])
df['condition_encoded'] = le_condition.fit_transform(df['condition'])

# Prepare features for training
X = df[['city_encoded', 'season_encoded', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']]
y_temp = df['temperature']
y_condition = df['condition_encoded']

# Split data for evaluation
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_cond_train, y_cond_test = train_test_split(X, y_condition, test_size=0.2, random_state=42)

# Train models
print("\n🤖 Training AI models...")
print("-" * 60)

# Temperature prediction model
temp_model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
temp_model.fit(X_train, y_temp_train)

# Condition prediction model
condition_model = GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42)
condition_model.fit(X_train, y_cond_train)

# Calculate accuracies
temp_train_score = temp_model.score(X_train, y_temp_train)
temp_test_score = temp_model.score(X_test, y_temp_test)
cond_train_score = condition_model.score(X_train, y_cond_train)
cond_test_score = condition_model.score(X_test, y_cond_test)

# Calculate RMSE for temperature
y_pred_temp = temp_model.predict(X_test)
temp_rmse = np.sqrt(mean_squared_error(y_temp_test, y_pred_temp))

print("✅ Model training completed!")
print(f"\n📊 Temperature Model:")
print(f"   Training Accuracy: {temp_train_score*100:.2f}%")
print(f"   Testing Accuracy:  {temp_test_score*100:.2f}%")
print(f"   RMSE: {temp_rmse:.2f}°C")

print(f"\n📊 Condition Model:")
print(f"   Training Accuracy: {cond_train_score*100:.2f}%")
print(f"   Testing Accuracy:  {cond_test_score*100:.2f}%")

print("\n" + "="*60)
print("✅ API READY - Server starting on http://localhost:5000")
print("="*60 + "\n")

@app.route('/api/weather', methods=['POST'])
def get_weather():
    data = request.json
    city = data.get('city', 'New York')
    
    # Get current season
    month = datetime.now().month
    if month in [12, 1, 2]:
        season = 'Winter'
    elif month in [3, 4, 5]:
        season = 'Spring'
    elif month in [6, 7, 8]:
        season = 'Summer'
    else:
        season = 'Fall'
    
    # Get historical data for the city
    city_data = df[df['city'] == city]
    
    if len(city_data) == 0:
        city = 'New York'
        city_data = df[df['city'] == city]
    
    # Get average values
    avg_humidity = int(city_data['humidity'].mean())
    avg_pressure = int(city_data['pressure'].mean())
    avg_wind = int(city_data['wind_speed'].mean())
    avg_cloud = int(city_data['cloud_cover'].mean())
    
    # Encode inputs
    try:
        city_enc = le_city.transform([city])[0]
    except:
        city_enc = 0
        city = 'New York'
    
    season_enc = le_season.transform([season])[0]
    
    # Make prediction
    X_pred = np.array([[city_enc, season_enc, avg_humidity, avg_pressure, avg_wind, avg_cloud]])
    
    predicted_temp = round(float(temp_model.predict(X_pred)[0]), 1)
    predicted_condition_enc = int(round(condition_model.predict(X_pred)[0]))
    predicted_condition_enc = max(0, min(len(le_condition.classes_) - 1, predicted_condition_enc))
    predicted_condition = le_condition.inverse_transform([predicted_condition_enc])[0]
    
    # Generate 5-day forecast
    forecast = []
    prev_temp = predicted_temp
    
    for i in range(1, 6):
        humidity = max(30, min(95, avg_humidity + np.random.randint(-10, 10)))
        wind = max(0, min(50, avg_wind + np.random.randint(-5, 5)))
        cloud = max(0, min(100, avg_cloud + np.random.randint(-15, 15)))
        
        X_forecast = np.array([[city_enc, season_enc, humidity, avg_pressure, wind, cloud]])
        
        day_temp = round(float(temp_model.predict(X_forecast)[0]), 1)
        day_condition_enc = int(round(condition_model.predict(X_forecast)[0]))
        day_condition_enc = max(0, min(len(le_condition.classes_) - 1, day_condition_enc))
        day_condition = le_condition.inverse_transform([day_condition_enc])[0]
        
        forecast.append({
            'day': (datetime.now() + timedelta(days=i)).strftime('%A'),
            'temperature': day_temp,
            'condition': day_condition,
            'humidity': int(humidity),
            'wind_speed': int(wind)
        })
        
        prev_temp = day_temp
    
    response = {
        'city': city,
        'current': {
            'temperature': predicted_temp,
            'condition': predicted_condition,
            'humidity': avg_humidity,
            'pressure': avg_pressure,
            'wind_speed': avg_wind,
            'cloud_cover': avg_cloud
        },
        'forecast': forecast,
        'dataset_size': len(df),
        'model_accuracy': round(temp_test_score * 100, 2)
    }
    
    return jsonify(response)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = {
        'total_records': len(df),
        'cities': df['city'].unique().tolist(),
        'date_range': {
            'start': df['date'].min() if 'date' in df.columns else 'N/A',
            'end': df['date'].max() if 'date' in df.columns else 'N/A'
        },
        'temperature_stats': {
            'mean': round(float(df['temperature'].mean()), 2),
            'min': round(float(df['temperature'].min()), 2),
            'max': round(float(df['temperature'].max()), 2),
            'std': round(float(df['temperature'].std()), 2)
        },
        'conditions': df['condition'].value_counts().to_dict(),
        'model_performance': {
            'temperature_model': {
                'train_accuracy': round(temp_train_score * 100, 2),
                'test_accuracy': round(temp_test_score * 100, 2),
                'rmse': round(float(temp_rmse), 2)
            },
            'condition_model': {
                'train_accuracy': round(cond_train_score * 100, 2),
                'test_accuracy': round(cond_test_score * 100, 2)
            }
        }
    }
    return jsonify(stats)

@app.route('/api/city/<city_name>', methods=['GET'])
def get_city_data(city_name):
    city_data = df[df['city'] == city_name]
    
    if len(city_data) == 0:
        return jsonify({'error': 'City not found'}), 404
    
    response = {
        'city': city_name,
        'total_records': len(city_data),
        'temperature': {
            'mean': round(float(city_data['temperature'].mean()), 2),
            'min': round(float(city_data['temperature'].min()), 2),
            'max': round(float(city_data['temperature'].max()), 2)
        },
        'conditions': city_data['condition'].value_counts().to_dict(),
        'avg_humidity': round(float(city_data['humidity'].mean()), 2),
        'avg_wind_speed': round(float(city_data['wind_speed'].mean()), 2)
    }
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Weather Prediction API',
        'version': '2.0',
        'endpoints': {
            'POST /api/weather': 'Get weather prediction for a city',
            'GET /api/stats': 'Get dataset statistics',
            'GET /api/city/<city_name>': 'Get city-specific data'
        },
        'status': 'online',
        'dataset_loaded': len(df) > 0,
        'total_records': len(df)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
