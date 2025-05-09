import json
import pandas as pd
import joblib
import os
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import traceback
from django.contrib.sessions.models import Session
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib.auth import logout
from rest_framework import generics
from rest_framework.authtoken.models import Token
from django.utils.decorators import method_decorator
from django.views import View
from .models import UserActivity, PredictionHistory, AIModelPerformance
from .models import *
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
from rest_framework.response import Response
import json
from django.views.decorators.csrf import csrf_exempt
from utils import generate_dynamic_learning_links
from .serializers import UserSerializer, InterestSerializer, EducationSerializer, JobSerializer, ProfileSerializer, SkillSerializer, ProfileSerializer
from django.contrib.auth import get_user_model
from rest_framework.views import APIView
from rest_framework import generics, status, permissions
from django.contrib.auth import authenticate
import jwt
from datetime import datetime, timedelta
from rest_framework.authtoken.models import Token
from django.db.models import Avg, Count, Sum, Max, Min
from django.utils.timezone import now
import csv

User = get_user_model()


from django.views.generic import TemplateView


class HomeView(TemplateView):
    template_name = "home/home.html"
    
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

          # SAVE PREDICTION HISTORY
        PredictionHistory.objects.create(
            session_id=data.get('session_id', 'unknown'),  # Assume frontend sends a session_id
            user_input=data,
            predicted_careers=[rec['title'] for rec in recommendations],
            confidence_scores=[rec['matchScore'] for rec in recommendations]
        )

        # SAVE AI PERFORMANCE
        for score in top_probabilities:
            AIModelPerformance.objects.create(
                prediction_success=True,
                confidence_score=float(score)
            )
        
        return JsonResponse({'recommendations': recommendations})
    
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class TrackUserActivity(View):
    def post(self, request, *args, **kwargs):
        data = json.loads(request.body)

        UserActivity.objects.create(
            session_id=data.get('session_id', 'unknown'),
            event_type=data.get('event_type', 'unknown'),
            event_data=data.get('event_data', {})
        )
        return JsonResponse({'message': 'Activity tracked successfully'})

@csrf_exempt
def dashboard_summary(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Only GET allowed'}, status=405)

    total_predictions = PredictionHistory.objects.count()
    avg_confidence = AIModelPerformance.objects.aggregate(Avg('confidence_score'))['confidence_score__avg']
    user_events = UserActivity.objects.values('event_type').annotate(count=Count('id'))

    summary = {
        "total_predictions": total_predictions,
        "average_confidence": avg_confidence,
        "user_event_counts": list(user_events)
    }
    return JsonResponse(summary)

@csrf_exempt
def export_prediction_report(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Only GET allowed'}, status=405)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction_history.csv"'

    writer = csv.writer(response)
    writer.writerow(['User Session', 'Predicted Career', 'Confidence Score', 'Timestamp'])

    for record in PredictionHistory.objects.all():
        writer.writerow([record.user_session, record.predicted_career, record.confidence_score, record.timestamp])

    return response

@csrf_exempt
def user_engagement_summary(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Only GET allowed'}, status=405)

    today = now().date()
    week_ago = today - timedelta(days=7)

    daily_users = UserActivity.objects.filter(created_at__date=today).count()
    weekly_users = UserActivity.objects.filter(created_at__date__gte=week_ago).count()

    return JsonResponse({
        "daily_active_users": daily_users,
        "weekly_active_users": weekly_users,
    })


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

# Token helper
def get_token_from_request(request):
    token = request.COOKIES.get('auth_token')
    if token:
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    return None

class SignUpView(generics.CreateAPIView):
    serializer_class = UserSerializer
    permission_classes = [AllowAny]  
    queryset = User.objects.all()

    def create(self, request, *args, **kwargs):
        print("SIGNUP DATA RECEIVED:", request.data)  # 🔍 Print data
        return super().create(request, *args, **kwargs)


from django.contrib.auth.hashers import check_password
@method_decorator(csrf_exempt, name='dispatch')
class SignInView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        print("SIGNIN DATA:", request.data)
        email = request.data.get('email')
        password = request.data.get('password')

        if not email or not password:
            return Response({'error': 'Email and password are required'}, status=status.HTTP_400_BAD_REQUEST)

        # Use `username=email` so it works with your custom EmailBackend
        user = authenticate(request, username=email, password=password)

        if user is None:
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_400_BAD_REQUEST)

        token, _ = Token.objects.get_or_create(user=user)

        response = Response({'message': 'Login successful'})
        response.set_cookie(
            key='auth_token',
            value=token.key,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite='Lax',
            max_age=7 * 24 * 60 * 60,
            path='/',
        )
        print("Token set in cookie:", token.key)
        return response


class UserInterestsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        interests = Interest.objects.filter(user=request.user)
        serializer = InterestSerializer(interests, many=True)
        return Response(serializer.data)

    def post(self, request):
        # Clear existing interests and add new ones
        Interest.objects.filter(user=request.user).delete()
        interests_data = request.data.get('interests', [])
        for interest in interests_data:
            Interest.objects.create(
                user=request.user,
                name=interest['name'],
                category=interest.get('category', 'Personal')
            )
        return Response({"message": "Interests updated successfully."})

@api_view(["GET", "POST", "PUT"])
@permission_classes([IsAuthenticated])
def education(request):
    if request.method == "GET":
        # Return all education entries for current user
        educations = Education.objects.filter(user=request.user)
        serializer = EducationSerializer(educations, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    elif request.method in ["POST", "PUT"]:
        # Clear existing entries (if PUT or POST is treated as update)
        Education.objects.filter(user=request.user).delete()

        education_data = request.data.get("education", [])

        if not isinstance(education_data, list):
            return Response({"error": "Expected a list of education entries under 'education' key."}, status=400)

        for edu in education_data:
            if not isinstance(edu, dict):
                continue  # skip invalid entry

            year_range = edu.get("year", " - ")
            start_year, end_year = ("", "")
            if " - " in year_range:
                start_year, end_year = year_range.split(" - ")

            Education.objects.create(
                user=request.user,
                institution=edu.get("institution"),
                degree=edu.get("degree"),
                field=edu.get("field"),
                start_year=start_year.strip(),
                end_year=end_year.strip(),
                description=edu.get("description")
            )

        return Response({"message": "Education updated successfully."}, status=status.HTTP_200_OK)
    

class UserSkillsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        skills = Skill.objects.filter(user=request.user)
        serializer = SkillSerializer(skills, many=True)
        return Response(serializer.data)

    def post(self, request):
        # Clear old skills and add new ones
        Skill.objects.filter(user=request.user).delete()
        skills_data = request.data.get('skills', [])

        for skill in skills_data:
            Skill.objects.create(
                user=request.user,
                name=skill.get('name', ''),
                level=skill.get('level', 'Intermediate'),
                description=skill.get('description', '')
            )

        return Response({"message": "Skills updated successfully."}, status=status.HTTP_200_OK)

class PersonalView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        profile, created = Profile.objects.get_or_create(user=request.user)
        serializer = ProfileSerializer(profile)
        return Response(serializer.data)

    def post(self, request):
        profile, created = Profile.objects.get_or_create(user=request.user)
        serializer = ProfileSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        profile, _ = Profile.objects.get_or_create(user=request.user)
        serializer = ProfileSerializer(profile)
        return Response(serializer.data)

    def post(self, request):
        profile, _ = Profile.objects.get_or_create(user=request.user)
        serializer = ProfileSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def ensure_user_profile(request):
    user = request.user
    profile, created = Profile.objects.get_or_create(user=user)
    return Response({"profile_id": profile.id, "created": created})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user(request):
    serializer = UserSerializer(request.user)
    return Response(serializer.data)

@csrf_exempt
@require_POST
def logout_view(request):
    try:
        user = request.user
        print("Logging out user:", user)

        # Clear token
        if user.is_authenticated:
            Token.objects.filter(user=user).delete()

        logout(request)  # Clear session too

        response = JsonResponse({"message": "Logged out successfully."})
        response.delete_cookie("auth_token")
        return response
    except Exception as e:
        print("Logout error:", str(e))
        return JsonResponse({"error": "Logout failed."}, status=500)

from rest_framework.authtoken.models import Token

@api_view(['POST'])
def verify_token(request):
    token_key = request.data.get('token')

    if not token_key:
        return Response({'error': 'No token provided'}, status=400)

    try:
        token = Token.objects.get(key=token_key)
        user = token.user

        return Response({
            'valid': True,
            'user': {
                'id': str(user.id),
                'username': user.username,
                'email': user.email,
                'role': getattr(user, 'role', 'user'),
            }
        })
    except Token.DoesNotExist:
        return Response({'error': 'Invalid token'}, status=401)
    
@ensure_csrf_cookie
def get_csrf_token(request):
    return JsonResponse({"detail": "CSRF cookie set"})


from rest_framework import viewsets, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from .models import Profile, Skill, Interest, Education
from .serializers import (
    ProfileSerializer, SkillSerializer, InterestSerializer, 
    EducationSerializer, PersonalInfoSerializer
)

class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Custom permission to only allow owners of a profile to edit it.
    """
    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions are only allowed to the owner
        if hasattr(obj, 'user'):
            return obj.user == request.user
        elif hasattr(obj, 'profile'):
            return obj.profile.user == request.user
        return False

class ProfileView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
    
    def get(self, request):
        """Get the current user's profile"""
        try:
            profile = Profile.objects.get(user=request.user)
            serializer = ProfileSerializer(profile)
            return Response(serializer.data)
        except Profile.DoesNotExist:
            # Create a new profile if it doesn't exist
            profile = Profile.objects.create(user=request.user, name=request.user.username)
            serializer = ProfileSerializer(profile)
            return Response(serializer.data)
    
    def patch(self, request):
        """Update the current user's profile"""
        profile = get_object_or_404(Profile, user=request.user)
        serializer = ProfileSerializer(profile, data=request.data, partial=True)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def ensure_profile(request):
    """Ensure the user has a profile, creating one if needed"""
    profile, created = Profile.objects.get_or_create(
        user=request.user,
        defaults={'name': request.user.username}
    )
    return Response({'profile_id': profile.id, 'created': created})

class PersonalInfoView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
    
    def get(self, request):
        profile = get_object_or_404(Profile, user=request.user)
        serializer = PersonalInfoSerializer(profile)
        return Response(serializer.data)
    
    def patch(self, request):
        profile = get_object_or_404(Profile, user=request.user)
        
        # Log the received data for debugging
        print(f"Received personal info data: {request.data}")
        
        serializer = PersonalInfoSerializer(profile, data=request.data, partial=True)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        # Log validation errors
        print(f"Validation errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ProfileHeaderView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
    
    def patch(self, request):
        profile = get_object_or_404(Profile, user=request.user)
        
        # Only update name, title, and bio fields
        data = {
            'name': request.data.get('name', profile.name),
            'title': request.data.get('title', profile.title),
            'bio': request.data.get('bio', profile.bio),
        }
        
        serializer = ProfileSerializer(profile, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# Update the SkillsView class to match the ProfileSkill model
class SkillsView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
    
    def get(self, request):
        profile = get_object_or_404(Profile, user=request.user)
        skills = ProfileSkill.objects.filter(profile=profile)
        serializer = SkillSerializer(skills, many=True)
        return Response(serializer.data)
    
    def put(self, request):
        profile = get_object_or_404(Profile, user=request.user)
        
        # Clear existing skills
        ProfileSkill.objects.filter(profile=profile).delete()
        
        # Add new skills
        skills_data = request.data.get('skills', [])
        
        # Log the received data for debugging
        print(f"Received skills data: {skills_data}")
        
        # Handle both array of strings and array of objects
        for skill_data in skills_data:
            if isinstance(skill_data, dict):
                ProfileSkill.objects.create(
                    profile=profile, 
                    name=skill_data.get('name', ''),
                    level=skill_data.get('level', 'Intermediate')
                )
            elif isinstance(skill_data, str):
                ProfileSkill.objects.create(profile=profile, name=skill_data, level='Intermediate')
        
        # Return updated skills
        skills = ProfileSkill.objects.filter(profile=profile)
        serializer = SkillSerializer(skills, many=True)
        return Response(serializer.data)

class InterestsView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
    
    def get(self, request):
        profile = get_object_or_404(Profile, user=request.user)
        interests = Interest.objects.filter(profile=profile)
        serializer = InterestSerializer(interests, many=True)
        return Response(serializer.data)
    
    def put(self, request):
        profile = get_object_or_404(Profile, user=request.user)
        
        # Clear existing interests
        Interest.objects.filter(profile=profile).delete()
        
        # Add new interests
        interests_data = request.data.get('interests', [])
        
        # Handle both array of strings and array of objects
        for interest_data in interests_data:
            if isinstance(interest_data, dict):
                Interest.objects.create(
                    profile=profile, 
                    name=interest_data.get('name', ''),
                    category=interest_data.get('category', 'Personal')
                )
            elif isinstance(interest_data, str):
                Interest.objects.create(profile=profile, name=interest_data, category='Personal')
        
        # Return updated interests
        interests = Interest.objects.filter(profile=profile)
        serializer = InterestSerializer(interests, many=True)
        return Response(serializer.data)

class EducationView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
    
    def get(self, request):
        profile = get_object_or_404(Profile, user=request.user)
        education = Education.objects.filter(profile=profile)
        serializer = EducationSerializer(education, many=True)
        return Response(serializer.data)
    
    def put(self, request):
        profile = get_object_or_404(Profile, user=request.user)
        
        # Clear existing education entries
        Education.objects.filter(profile=profile).delete()
        
        # Add new education entries
        education_data = request.data.get('education', [])
        for edu_data in education_data:
            Education.objects.create(profile=profile, **edu_data)
        
        # Return updated education
        education = Education.objects.filter(profile=profile)
        serializer = EducationSerializer(education, many=True)
        return Response(serializer.data)
