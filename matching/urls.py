from .views import register_user, get_brighter_monday_jobs
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .models import *
from .serializers import *
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import SignUpView, verify_token, get_user_skills_assessment, GetLatestCareerPredictionView, GetJobListingsForCareerView, SignInView, get_csrf_token, logout_view, ensure_profile, current_user, TrackUserActivity, dashboard_summary, user_engagement_summary, export_prediction_report
from .views import ProfileView, ProfileHeaderView, PersonalInfoView, SkillsView, InterestsView, EducationView, SaveCareerPredictionView
from .views import SignUpView, verify_token, SignInView, get_csrf_token, logout_view, ensure_profile, UserProfileView, current_user, UserSkillsView, education, UserInterestsView, TrackUserActivity, dashboard_summary, user_engagement_summary, export_prediction_report
from .views import ProfileView, ProfileHeaderView, PersonalInfoView, SkillsView, InterestsView, EducationView
from .views import HomeView

urlpatterns = [
    path("", HomeView.as_view(), name="api-home"),
    path('trigger-scraping/', views.trigger_scraping, name='trigger-scraping'),
    path('jobs/', get_brighter_monday_jobs, name='brightermonday-jobs'),
    path('register/', register_user, name='register'),
    path('login/', TokenObtainPairView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('submit-assessment/', views.submit_assessment, name='submit_assessment'),
    path('get-skills/', views.get_skills, name='get_skills'),
    path('recommend-careers/', views.recommend_careers, name='recommend-careers'),
    path('recommend-learning/', views.recommend_learning, name='recommend-learning'),
    path('', views.home, name='home'),
    path('predict/', views.predict_career, name='predict_career'),
    path('signup/', SignUpView.as_view(), name='signup'),
    path('signin/', SignInView.as_view(), name='signin'),
    path('track_activity/', TrackUserActivity.as_view(), name='track_activity'),
    path('dashboard_summary/', dashboard_summary, name='dashboard_summary'),
    path('export_prediction_report/', export_prediction_report, name='export_prediction_report'),
    path('user_engagement_summary/', user_engagement_summary, name='user_engagement_summary'),
    path('me/', current_user, name='current_user'),
    path("logout/", logout_view, name="logout"),
    path('verify-token', views.verify_token, name='verify-token'),
    path("csrf/", get_csrf_token),
    path('ensure-profile/', views.ensure_profile, name='ensure-profile'),
    path('profile/', views.ProfileView.as_view(), name='profile'),
    path('header/', views.ProfileHeaderView.as_view(), name='profile-header'),
    path('personal-info/', views.PersonalInfoView.as_view(), name='personal-info'),
    path('skills/', views.SkillsView.as_view(), name='skills'),
    path('interests/', views.InterestsView.as_view(), name='interests'),
    path('education/', views.EducationView.as_view(), name='education'),
    path('save/', SaveCareerPredictionView.as_view(), name='save-career-prediction'),
    path('latest/', GetLatestCareerPredictionView.as_view(), name='get-latest-prediction'),
    path('job/<str:career_title>/', GetJobListingsForCareerView.as_view(), name='get-job-listings'),
    path('skills/assessment/', get_user_skills_assessment, name='get-skills-assessment'),
    
   
]
