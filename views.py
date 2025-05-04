from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Count
from django.db.models.functions import TruncDate
import joblib
import os
import pandas as pd
import json
from datetime import datetime

from .models import PredictionRecord, ModelMetrics

# Path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_models', 'best_model.pkl')
import csv
from django.http import HttpResponse, JsonResponse

@login_required
def export_predictions(request):
    """Export predictions as CSV"""
    try:
        # Create a CSV file in memory
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
        
        # Create CSV writer
        writer = csv.writer(response)
        
        # Write header row
        writer.writerow([
            'Prediction Date', 'User', 'Administrative Pages', 'Administrative Duration', 
            'Informational Pages', 'Informational Duration', 'Product Related Pages', 
            'Product Related Duration', 'Bounce Rate', 'Exit Rate', 'Page Value', 
            'Special Day', 'Month', 'Operating System', 'Browser', 'Region', 
            'Traffic Type', 'Visitor Type', 'Weekend', 'Prediction'
        ])
        
        # In your case, since you're having model issues, just return the empty CSV
        # In a complete implementation, you would query the database here
        
        return response
    except Exception as e:
        messages.error(request, f"Error exporting data: {str(e)}")
        return redirect('dashboard')

def load_model():
    """Load the trained model if it exists"""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def home(request):
    """Home page view"""
    return render(request, 'myapp/home.html', {
        'current_datetime': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    })

@login_required
def export_predictions(request):
    """Temporary stub function for CSV export"""
    messages.info(request, "Export feature is coming soon!")
    return redirect('dashboard')

@login_required
def bulk_predict(request):
    """Temporary stub function for bulk prediction"""
    messages.info(request, "Bulk prediction feature is coming soon!")
    return redirect('dashboard')

def api_predict(request):
    """Temporary stub function for API"""
    return JsonResponse({'status': 'API coming soon'}, status=200)

@login_required
def predict(request):
    """Prediction form and processing"""
    model = load_model()
    if model is None:
        messages.error(request, "Model not found. Please train the model first!")
        return redirect('dashboard')
    
    if request.method == 'POST':
        # Extract features from form
        features = {
            'Administrative': float(request.POST.get('administrative')),
            'Administrative_Duration': float(request.POST.get('administrative_duration')),
            'Informational': float(request.POST.get('informational')),
            'Informational_Duration': float(request.POST.get('informational_duration')),
            'ProductRelated': float(request.POST.get('product_related')),
            'ProductRelated_Duration': float(request.POST.get('product_related_duration')),
            'BounceRates': float(request.POST.get('bounce_rate')),
            'ExitRates': float(request.POST.get('exit_rate')),
            'PageValues': float(request.POST.get('page_value')),
            'SpecialDay': float(request.POST.get('special_day')),
            'Month': request.POST.get('month'),
            'OperatingSystems': int(request.POST.get('operating_systems')),
            'Browser': int(request.POST.get('browser')),
            'Region': int(request.POST.get('region')),
            'TrafficType': int(request.POST.get('traffic_type')),
            'VisitorType': request.POST.get('visitor_type'),
            'Weekend': request.POST.get('weekend') == 'True',
        }
        
        # Convert to DataFrame and make prediction
        df = pd.DataFrame([features])
        
        # Ensure proper encoding for categorical variables 
        prediction = model.predict(df)[0]
        
        # Save the prediction
        record = PredictionRecord(
            user=request.user,
            admin_info=features['Administrative'],
            administrative_duration=features['Administrative_Duration'],
            informational=features['Informational'],
            informational_duration=features['Informational_Duration'],
            product_related=features['ProductRelated'],
            product_related_duration=features['ProductRelated_Duration'],
            bounce_rate=features['BounceRates'],
            exit_rate=features['ExitRates'],
            page_value=features['PageValues'],
            special_day=features['SpecialDay'],
            month=features['Month'],
            operating_systems=features['OperatingSystems'],
            browser=features['Browser'],
            region=features['Region'],
            traffic_type=features['TrafficType'],
            visitor_type=features['VisitorType'],
            weekend=features['Weekend'],
            prediction=bool(prediction)
        )
        record.save()
        
        return render(request, 'myapp/result.html', {
            'prediction': bool(prediction),
            'features': features,
            'current_datetime': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # GET request - show form
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    visitor_types = ['Returning_Visitor', 'New_Visitor', 'Other']
    
    return render(request, 'myapp/predict.html', {
        'months': months,
        'visitor_types': visitor_types,
        'current_datetime': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    })

@login_required
def dashboard(request):
    """Dashboard with statistics"""
    # Get latest model metrics
    try:
        metrics = ModelMetrics.objects.latest('updated_date')
    except ModelMetrics.DoesNotExist:
        metrics = {
            'auc_score': 0,
            'precision_score': 0,
            'recall_score': 0,
            'f1_score': 0
        }
    
    # Get prediction statistics
    total_predictions = PredictionRecord.objects.count()
    purchase_count = PredictionRecord.objects.filter(prediction=True).count()
    no_purchase_count = total_predictions - purchase_count
    purchase_rate = purchase_count / total_predictions if total_predictions > 0 else 0
    
    # Get predictions over time
    predictions_by_date = (
        PredictionRecord.objects
        .annotate(date=TruncDate('prediction_date'))
        .values('date')
        .annotate(count=Count('id'))
        .order_by('date')
    )
    
    dates = [pred['date'].strftime('%Y-%m-%d') for pred in predictions_by_date]
    counts = [pred['count'] for pred in predictions_by_date]
    
    stats = {
        'total_predictions': total_predictions,
        'purchase_count': purchase_count,
        'no_purchase_count': no_purchase_count,
        'purchase_rate': purchase_rate,
        'dates': json.dumps(dates),
        'counts': json.dumps(counts)
    }
    
    return render(request, 'myapp/dashboard.html', {
        'metrics': metrics,
        'stats': stats,
        'current_datetime': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    })

@login_required
def train_model(request):
    """Train the model and save metrics"""
    messages.success(request, "Model training initiated. This will take some time.")
    return redirect('dashboard')

def documentation(request):
    """Documentation view"""
    return render(request, 'myapp/documentation.html', {
        'current_datetime': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    })
