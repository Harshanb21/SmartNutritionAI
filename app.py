# ============================================================================
# FLASK WEB APPLICATION - Smart Nutrition & Exercise Companion
# Backend API with ML Integration
# ============================================================================

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# ============================================================================
# LOAD DATASETS AND INITIALIZE ML SYSTEM
# ============================================================================

class SmartFitnessML:
    """ML System for Web Application"""
    
    def __init__(self):
        """Initialize with all three datasets"""
        try:
            self.users_df = pd.read_csv('users.csv')
            self.activity_df = pd.read_csv('activity_calories.csv')
            self.indian_food_df = pd.read_csv('Indian_Food_Nutrition_Processed.csv')
            
            print(f"✅ Loaded {len(self.users_df)} user profiles")
            print(f"✅ Loaded {len(self.activity_df)} activities")
            print(f"✅ Loaded {len(self.indian_food_df)} Indian foods")
            
            self._preprocess_data()
            self._train_model()
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise
    
    def _preprocess_data(self):
        """Preprocess all datasets"""
        # User data preprocessing
        bmi_mapping = {'Normal': 25, 'Underweight': 23, 'Overweight': 27, 'Obese': 32}
        self.users_df['BMI_numeric'] = self.users_df['BMI Category'].map(bmi_mapping)
        
        self.ml_features = self.users_df[[
            'Age', 'Physical Activity Level', 'BMI_numeric', 
            'Daily Steps', 'Stress Level'
        ]].dropna()
        
        # Indian food preprocessing
        self.indian_food_df.columns = [
            'dish_name', 'calories', 'carbs', 'protein', 'fats', 
            'sugar', 'fiber', 'sodium', 'calcium', 'iron', 
            'vitamin_c', 'folate'
        ]
        
        print(f"✅ Preprocessed {len(self.ml_features)} user samples")
    
    def _train_model(self):
        """Train KNN model"""
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(self.ml_features)
        
        self.knn_model = NearestNeighbors(n_neighbors=4, algorithm='auto', metric='euclidean')
        self.knn_model.fit(features_scaled)
        
        print(f"✅ KNN model trained on {len(self.ml_features)} samples")
    
    def calculate_metrics(self, weight_kg, height_cm, age, gender, activity_level):
        """Calculate BMI, BMR, TDEE"""
        # BMR (Harris-Benedict)
        if gender.lower() == 'male':
            bmr = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)
        
        # TDEE
        multipliers = {'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 
                      'active': 1.725, 'very_active': 1.9}
        tdee = bmr * multipliers.get(activity_level, 1.55)
        
        # BMI
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        return {
            'bmr': round(bmr, 0),
            'tdee': round(tdee, 0),
            'bmi': round(bmi, 1)
        }
    
    def find_similar_users(self, age, activity_level, bmi, steps, stress):
        """Find 3 similar users using KNN"""
        user_features = np.array([[age, activity_level, bmi, steps, stress]])
        user_scaled = self.scaler.transform(user_features)
        
        distances, indices = self.knn_model.kneighbors(user_scaled, n_neighbors=4)
        
        similar_users = []
        for i in range(1, 4):
            idx = indices[0][i]
            similarity = max(0, 100 - (distances[0][i] * 20))
            
            similar_users.append({
                'similarity': round(similarity, 1),
                'age': int(self.ml_features.iloc[idx]['Age']),
                'activity': int(self.ml_features.iloc[idx]['Physical Activity Level']),
                'bmi': float(self.ml_features.iloc[idx]['BMI_numeric'])
            })
        
        return similar_users
    
    def recommend_exercises(self, weight_kg, fitness_level, goal):
        """Get top 5 exercises"""
        activity_data = self.activity_df.copy()
        activity_data['calories_per_kg'] = activity_data['Calories per kg']
        
        def intensity(cal_per_kg):
            return 'Light' if cal_per_kg < 1.0 else ('Moderate' if cal_per_kg < 2.0 else 'Vigorous')
        
        activity_data['intensity'] = activity_data['calories_per_kg'].apply(intensity)
        
        # Filter by fitness level
        if fitness_level == 'beginner':
            filtered = activity_data[activity_data['intensity'].isin(['Light', 'Moderate'])]
        elif fitness_level == 'intermediate':
            filtered = activity_data[activity_data['intensity'].isin(['Moderate', 'Vigorous'])]
        else:
            filtered = activity_data[activity_data['intensity'] == 'Vigorous']
        
        filtered = filtered.copy()
        filtered['calories_30min'] = (filtered['calories_per_kg'] * weight_kg * 0.5).round(0)
        
        if goal == 'weight_loss':
            filtered = filtered.sort_values('calories_30min', ascending=False)
        else:
            filtered = filtered.sort_values('calories_30min', ascending=True)
        
        exercises = []
        for _, row in filtered.head(5).iterrows():
            exercises.append({
                'name': row['Activity, Exercise or Sport (1 hour)'],
                'intensity': row['intensity'],
                'calories': int(row['calories_30min'])
            })
        
        return exercises
    
    def generate_meal_plan(self, target_calories, goal):
        """Generate meal plan from Indian food database"""
        breakfast_target = target_calories * 0.25
        lunch_target = target_calories * 0.35
        dinner_target = target_calories * 0.30
        snack_target = target_calories * 0.10
        
        def find_meal(target, keywords):
            candidates = self.indian_food_df[
                self.indian_food_df['dish_name'].str.contains('|'.join(keywords), case=False, na=False)
            ].copy()
            
            if len(candidates) == 0:
                candidates = self.indian_food_df.copy()
            
            candidates['diff'] = abs(candidates['calories'] - target)
            best = candidates.nsmallest(1, 'diff').iloc[0]
            
            return {
                'name': best['dish_name'],
                'calories': round(best['calories'], 1),
                'protein': round(best['protein'], 1),
                'carbs': round(best['carbs'], 1),
                'fats': round(best['fats'], 1)
            }
        
        breakfast = find_meal(breakfast_target, ['chai', 'coffee', 'dosa', 'idli', 'paratha', 'upma'])
        lunch = find_meal(lunch_target, ['rice', 'dal', 'curry', 'roti', 'biryani', 'pulao'])
        dinner = find_meal(dinner_target, ['paneer', 'chicken', 'fish', 'curry', 'masala'])
        snacks = find_meal(snack_target, ['lassi', 'samosa', 'pakora', 'chaat', 'juice'])
        
        total_cal = breakfast['calories'] + lunch['calories'] + dinner['calories'] + snacks['calories']
        total_protein = breakfast['protein'] + lunch['protein'] + dinner['protein'] + snacks['protein']
        
        return {
            'breakfast': breakfast,
            'lunch': lunch,
            'dinner': dinner,
            'snacks': snacks,
            'totals': {
                'calories': round(total_cal, 1),
                'protein': round(total_protein, 1)
            }
        }
    
    def generate_recommendation(self, user_data):
        """Complete recommendation generation"""
        # Calculate metrics
        metrics = self.calculate_metrics(
            user_data['weight_kg'],
            user_data['height_cm'],
            user_data['age'],
            user_data['gender'],
            user_data['activity_level']
        )
        
        # Target calories
        if user_data['goal'] == 'weight_loss':
            target_calories = metrics['tdee'] * 0.8
        elif user_data['goal'] == 'muscle_gain':
            target_calories = metrics['tdee'] * 1.1
        else:
            target_calories = metrics['tdee']
        
        # Find similar users
        similar_users = self.find_similar_users(
            user_data['age'],
            user_data.get('activity_minutes', 60),
            metrics['bmi'],
            user_data.get('daily_steps', 7000),
            user_data.get('stress_level', 5)
        )
        
        # Get exercises
        exercises = self.recommend_exercises(
            user_data['weight_kg'],
            user_data['fitness_level'],
            user_data['goal']
        )
        
        # Generate meal plan
        meal_plan = self.generate_meal_plan(target_calories, user_data['goal'])
        
        # Confidence
        avg_similarity = np.mean([u['similarity'] for u in similar_users])
        confidence = round(min(95, 70 + avg_similarity * 0.3), 1)
        
        return {
            'metrics': {
                'bmi': metrics['bmi'],
                'bmr': metrics['bmr'],
                'tdee': metrics['tdee'],
                'target_calories': round(target_calories, 0)
            },
            'similar_users': similar_users,
            'exercises': exercises,
            'meal_plan': meal_plan,
            'confidence': confidence
        }

# Initialize ML system globally
print("Initializing ML System...")
ml_system = SmartFitnessML()
print("✅ System Ready!")

# ============================================================================
# FLASK ROUTES (API ENDPOINTS)
# ============================================================================

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendation():
    """API endpoint for generating recommendations"""
    try:
        user_data = request.json
        
        # Validate input
        required_fields = ['age', 'weight_kg', 'height_cm', 'gender', 
                          'activity_level', 'fitness_level', 'goal']
        
        for field in required_fields:
            if field not in user_data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Generate recommendations
        recommendations = ml_system.generate_recommendation(user_data)
        
        return jsonify(recommendations)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'users': len(ml_system.users_df),
        'activities': len(ml_system.activity_df),
        'foods': len(ml_system.indian_food_df)
    })

@app.route('/api/foods/search', methods=['GET'])
def search_foods():
    """Search Indian foods by name"""
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    results = ml_system.indian_food_df[
        ml_system.indian_food_df['dish_name'].str.contains(query, case=False, na=False)
    ].head(10)
    
    foods = []
    for _, row in results.iterrows():
        foods.append({
            'name': row['dish_name'],
            'calories': round(row['calories'], 1),
            'protein': round(row['protein'], 1)
        })
    
    return jsonify(foods)

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 STARTING FLASK WEB SERVER")
    print("=" * 70)
    print("\n📊 Datasets loaded:")
    print(f"   - Users: {len(ml_system.users_df)}")
    print(f"   - Activities: {len(ml_system.activity_df)}")
    print(f"   - Indian Foods: {len(ml_system.indian_food_df)}")
    print(f"\n🌐 Server running at: http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
