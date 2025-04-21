import json
import pandas as pd
import joblib
import os
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import traceback
from rest_framework import generics
from .models import *
from .serializers import SkillSerializer, JobSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from django.contrib.auth.models import User
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework import viewsets, filters
from .tasks import scrape_jobs_task
from django_filters.rest_framework import DjangoFilterBackend
from .scraper import scrape_brighter_monday
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.http import HttpResponse
from .models import UserAssessment
from ai_model.predict import identify_skill_gaps
from ai_model.resources import get_learning_resources
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
from utils import generate_dynamic_learning_links

# Load the career dataset
def load_career_dataset():
    """Load the career dataset with descriptions, required skills, and industry types"""
    try:
        # Update this path to where your career dataset is stored
        career_dataset_path = os.path.join("C:/Users/SHIRAH/Desktop/Test/career_matching/career_match/career_data.csv")
        career_df = pd.read_csv(career_dataset_path, encoding='latin1')
        return career_df
    except Exception as e:
        print(f"Error loading career dataset: {str(e)}")
        # Return a minimal dataset if the file can't be loaded
        return pd.DataFrame({
            'career_name': [],
            'description': [],
            'required_skills': [],
            'industry_type': []
        })

# Load the ML model and encoders
def load_models():
    """Load all the trained models and encoders"""
    model_dir = os.path.join(settings.BASE_DIR, 'matching/model')
    
    model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    skills_encoder = joblib.load(os.path.join(model_dir, "skills_encoder.pkl"))
    interests_encoder = joblib.load(os.path.join(model_dir, "interests_encoder.pkl"))
    education_encoder = joblib.load(os.path.join(model_dir, "education_encoder.pkl"))
    target_encoder = joblib.load(os.path.join(model_dir, "target_encoder.pkl"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    
    return model, skills_encoder, interests_encoder, education_encoder,  target_encoder, feature_names

# Preprocess user input for prediction
def preprocess_input(user_input, skills_encoder, interests_encoder, education_encoder, feature_names):
    """Convert user input to the format expected by the model"""
    # Extract user input
    age = user_input.get('age', 25)
    education = user_input.get('education', "bachelor's")
    skills = [skill.lower().strip() for skill in user_input.get('skills', [])]
    interests = [interest.lower().strip() for interest in user_input.get('interests', [])]

    # Encode skills
    skills_df = pd.DataFrame(columns=skills_encoder.classes_)
    skills_df.loc[0] = 0
    for skill in skills:
        if skill in skills_encoder.classes_:
            skills_df.loc[0, skill] = 1
        else:
            print(f"⚠️ Unknown skill: {skill}")  # Handle unknown skills

    # Encode interests
    interests_df = pd.DataFrame(columns=interests_encoder.classes_)
    interests_df.loc[0] = 0
    for interest in interests:
        if interest in interests_encoder.classes_:
            interests_df.loc[0, interest] = 1
        else:
            print(f"⚠️ Unknown interest: {interest}")  # Handle unknown interests

    print(f"Available education levels: {education_encoder.classes_}")
    education_cleaned = education.strip()
    education_value = education_encoder.transform([education_cleaned])[0]
    education_df = pd.DataFrame([[education_value]], columns=['education_encoded'])

    # Combine features
    X = pd.concat([skills_df.reset_index(drop=True),
                    interests_df.reset_index(drop=True),
                    education_df.reset_index(drop=True)], axis=1)

    # Remove duplicate columns in X
    X = X.loc[:, ~X.columns.duplicated()]

    print("Prediction DataFrame Columns:")
    print(X.columns.tolist())

    print("Feature Names From Training:")
    print(feature_names)
    missing_cols = set(feature_names) - set(X.columns)
    extra_cols = set(X.columns) - set(feature_names)

    if missing_cols:
        print("⚠️ Missing columns in prediction data:", missing_cols)
    if extra_cols:
        print("⚠️ Extra columns in prediction data:", extra_cols)

    X = X.reindex(columns=feature_names, fill_value=0)

    # Final check
    assert list(X.columns) == feature_names, "❌ Feature order mismatch!"

    return X


# Get career details from the dataset
def get_career_details(career_name, career_df):
    """Get description, required skills, and industry type for a career"""
    # Find the career in the dataset (case-insensitive)
    print(career_df.columns)

    career_df['career_name'] = career_df['career_name'].astype(str)

    career_row = career_df[career_df['career_name'].str.lower() == career_name.lower()]
    
    if not career_row.empty:
        # Get the first matching row
        row = career_row.iloc[0]
        
        # Parse required skills if they're stored as a string
        required_skills = row.get('required_skills', '')
        if isinstance(required_skills, str):
            try:
                # Try to parse as JSON
                required_skills = json.loads(required_skills)
            except:
                # If that fails, split by comma
                required_skills = [skill.strip() for skill in required_skills.split(',')]
        
        return {
            'description': row.get('description', 'No description available'),
            'required_skills': required_skills if isinstance(required_skills, list) else [],
            'industry_type': row.get('industry_type', 'Not specified')
        }
    else:
        # Return default values if career not found
        return {
            'description': 'No description available',
            'required_skills': [],
            'industry_type': 'Not specified'
        }

@csrf_exempt
def predict_career(request):
    """API endpoint to predict careers based on user input"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    try:
        # Parse JSON data from request
        data = json.loads(request.body)
        
        # Load models and career dataset
        model, skills_encoder, interests_encoder, education_encoder, target_encoder, feature_names = load_models()
        career_df = load_career_dataset()
        
        # Preprocess input
        X = preprocess_input(data, skills_encoder, interests_encoder, education_encoder, feature_names)
        
        # Make prediction
        probabilities = model.predict_proba(X)
        
        # Get top 3 predictions
        top_n = 3
        top_indices = np.argsort(probabilities[0])[::-1][:top_n]
        top_probabilities = probabilities[0][top_indices]
        
        # Convert indices to career names
        top_careers = target_encoder.inverse_transform(top_indices)
        
        # Create results with career details
        recommendations = []
        for i, (career, probability) in enumerate(zip(top_careers, top_probabilities)):
            # Get career details from dataset
            career_details = get_career_details(career, career_df)
            
            # Find matching skills and interests for explanation
            matching_skills = [skill for skill in data.get('skills', []) 
                              if skill.lower() in skills_encoder.classes_]
            matching_interests = [interest for interest in data.get('interests', []) 
                                 if interest.lower() in interests_encoder.classes_]
            
            # Create explanation
            explanation = {
                "skills": matching_skills[:3],  # Top 3 matching skills
                "interests": matching_interests[:3],  # Top 3 matching interests
                "education_match": True  # Simplified for this example
            }
            
            recommendations.append({
                "title": career,
                "matchScore": round(float(probability) * 1000), 
                "description": career_details['description'],
                "requiredSkills": career_details['required_skills'],
                "industryType": career_details['industry_type'],
                "explanation": explanation
            })
        
        return JsonResponse({'recommendations': recommendations})
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)
    

class JobViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Job.objects.all()
    serializer_class = JobSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['job_type', 'location', 'company']
    search_fields = ['title', 'description', 'company', 'location']
    ordering_fields = ['posted_date', 'created_at', 'title']
    ordering = ['-created_at']

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def trigger_scraping(request):
    task = scrape_jobs_task.delay()
    return Response({"message": "Job scraping started", "task_id": task.id})


@api_view(["GET"])
def get_brighter_monday_jobs(request):
    jobs = Job.objects.all().order_by('posted_date')  # Optional: limit e.g., .[:20]
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data)


@api_view(['POST'])
def register_user(request):
    username = request.data.get('username')
    password = request.data.get('password')
    if User.objects.filter(username=username).exists():
        return Response({'error': 'Username already taken'}, status=400)
    
    user = User.objects.create_user(username=username, password=password)
    return Response({'message': 'User registered successfully'})


def home(request):
    """
    A simple home endpoint to welcome users to the system.
    """
    return HttpResponse("Welcome to the E-Career Guidance System!")
    
@csrf_exempt
def recommend_careers(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_skills = data.get('skills', [])

        # 💡 Your logic to recommend careers based on `user_skills`
        careers = ["Software Developer", "Data Analyst"]  # Replace with actual logic

        return JsonResponse({"careers": careers})
    return JsonResponse({"error": "Invalid request method"}, status=400)

@api_view(['POST'])
def recommend_learning(request):
    missing_skills = request.data.get("missing_skills", [])
    skills_to_improve = request.data.get("skills_to_improve", [])
    all_skills = set(missing_skills + skills_to_improve)

    resources = []
    for skill in all_skills:
        links = generate_dynamic_learning_links(skill)
        for link in links:
            resources.append({
                "skill": skill,
                "site": link["site"],
                "resource": link["url"]
            })

    return Response({"resources": resources})
    
def process_skills(user_skills):
    """
    Process the user's skills and return analysis results.
    """
    if not user_skills or not isinstance(user_skills, list):
        return {"error": "Invalid or missing 'skills' list."}

    # Step 1: Analyze skills and predict gaps
    skill_analysis = identify_skill_gaps(user_skills)
    
    # Debugging: print the skill analysis
    print("Skill Analysis:", skill_analysis)

    # Step 2: Extract skills by category
    missing_skills = [
        skill.split(": ")[1]
        for skill in skill_analysis if skill.startswith("Missing skill")
    ]
    
    skills_to_improve = [
        skill.split(": ")[1]
        for skill in skill_analysis if skill.startswith("Skill to improve")
    ]
    
    strong_skills = [
        skill.split(": ")[1]
        for skill in skill_analysis if skill.startswith("Strong skill")
    ]

    # Step 3: Fetch learning recommendations
    learning_recommendations = get_learning_resources(missing_skills + skills_to_improve)

    # Step 4: Calculate percentage scores by skill type
    # This is a placeholder - implement your actual percentage calculation logic
    percentage_scores = calculate_percentage_scores(user_skills)

    return {
        "percentage_scores": percentage_scores,
        "strong_skills": strong_skills,
        "skills_to_improve": skills_to_improve,
        "missing_skills": missing_skills,
        "learning_recommendations": learning_recommendations
    }


def calculate_percentage_scores(user_skills):
    """
    Calculate percentage scores by skill type.
    Implement your actual calculation logic from your ML model here.
    """
    # This is a placeholder - replace with your actual calculation
    skill_types = {
        "Technical Skills": [],
        "Soft Skills": [],
        "Management Skills": [],
        "Analytical": [],
        "Creative": []
    }
    
    # Group skills by type
    for skill in user_skills:
        skill_type = skill.get("type", "Technical Skills")
        if skill_type in skill_types:
            skill_types[skill_type].append(skill)
    
    # Calculate average score for each type
    percentage_scores = {}
    for skill_type, skills in skill_types.items():
        if skills:
            avg_score = sum(skill.get("score", 0) for skill in skills) / len(skills)
            # Convert to percentage (assuming score is 1-5)
            percentage = (avg_score / 5) * 100
            percentage_scores[skill_type] = f"{percentage:.1f}%"
        else:
            percentage_scores[skill_type] = "0.0%"
    
    return percentage_scores


@api_view(['POST'])
@permission_classes([AllowAny])  # Allow any client to access this endpoint
@csrf_exempt  # Disable CSRF for this API endpoint
def submit_assessment(request):
    """
    Process skill assessment data and return formatted results for the frontend.
    """
    try:
        # Get skills data from request
        user_skills = request.data.get('skills', [])
        
        # Process the skills
        result = process_skills(user_skills)
        
        if "error" in result:
            return Response({"error": result["error"]}, status=400)
        
        # Format the response for the frontend
        response_data = {
            "percentageScores": result["percentage_scores"],
            "strongSkills": result["strong_skills"],
            "skillsToImprove": [
                {
                    "skill": skill,
                    "course": result["learning_recommendations"].get(skill, {}).get("course", "No course available"),
                    "link": result["learning_recommendations"].get(skill, {}).get("link", "")
                }
                for skill in result["skills_to_improve"]
            ],
            "missingSkills": [
                {
                    "skill": skill,
                    "course": result["learning_recommendations"].get(skill, {}).get("course", "No course available"),
                    "link": result["learning_recommendations"].get(skill, {}).get("link", "")
                }
                for skill in result["missing_skills"]
            ]
        }
        
        # Save assessment to database if needed
        UserAssessment.objects.create(
            user=request.user if request.user.is_authenticated else None,
            skills=user_skills,
            results=response_data
        )
        
        return Response(response_data)

    except Exception as e:
        print(f"Error processing assessment: {str(e)}")
        return Response({"error": str(e)}, status=500)



@api_view(['GET'])
@permission_classes([AllowAny])
def get_skills(request):
    """
    Return a list of predefined skills for the assessment.
    """
    # You can replace this with skills from your database
    skills = [
        {"id": 1, "name": "Python Programming", "type": "Technical Skills"},
        {"id": 2, "name": "Data Analysis", "type": "Technical Skills"},
        {"id": 3, "name": "Communication", "type": "Soft Skills"},
        {"id": 4, "name": "Project Management", "type": "Management Skills"},
        {"id": 5, "name": "Problem Solving", "type": "Analytical"},
        {"id": 6, "name": "Creative Thinking", "type": "Creative"},
        {"id": 7, "name": "JavaScript", "type": "Technical Skills"},
        {"id": 8, "name": "Leadership", "type": "Management Skills"},
        {"id": 9, "name": "Critical Thinking", "type": "Analytical"},
        {"id": 10, "name": "Design Thinking", "type": "Creative"},
    ]
    
    return Response(skills)


