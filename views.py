from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Count
from django.db.models.functions import TruncDate
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.urls import reverse
import joblib
import os
import pandas as pd
import json
import csv
from datetime import datetime
import io

from myapp.models import PredictionRecord, ModelMetrics

# Path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_models', 'best_model.pkl')

def load_model():
    """Load the trained model if it exists"""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def home(request):
    """Home page view"""
    return render(request, 'myapp/home.html')

@login_required
def predict(request):
    """Prediction form and processing"""
    model = load_model()
    if model is None:
        messages.error(request, "Model not found. Please train the model first!")
        return redirect('dashboard')
    
    if request.method == 'POST':
        try:
            # Extract features from form
            features = {
                'Administrative': float(request.POST.get('administrative', 0)),
                'Administrative_Duration': float(request.POST.get('administrative_duration', 0)),
                'Informational': float(request.POST.get('informational', 0)),
                'Informational_Duration': float(request.POST.get('informational_duration', 0)),
                'ProductRelated': float(request.POST.get('product_related', 0)),
                'ProductRelated_Duration': float(request.POST.get('product_related_duration', 0)),
                'BounceRates': float(request.POST.get('bounce_rate', 0)),
                'ExitRates': float(request.POST.get('exit_rate', 0)),
                'PageValues': float(request.POST.get('page_value', 0)),
                'SpecialDay': float(request.POST.get('special_day', 0)),
                'Month': request.POST.get('month', 'Feb'),
                'OperatingSystems': int(request.POST.get('operating_systems', 1)),
                'Browser': int(request.POST.get('browser', 1)),
                'Region': int(request.POST.get('region', 1)),
                'TrafficType': int(request.POST.get('traffic_type', 1)),
                'VisitorType': request.POST.get('visitor_type', 'Returning_Visitor'),
                'Weekend': request.POST.get('weekend') == 'True',
            }
            
            # Convert to DataFrame and make prediction
            df = pd.DataFrame([features])
            
            # Make prediction
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
                'features': features
            })
        except Exception as e:
            messages.error(request, f"Error making prediction: {str(e)}")
            return redirect('predict')
    
    # GET request - show form
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    visitor_types = ['Returning_Visitor', 'New_Visitor', 'Other']
    
    return render(request, 'myapp/predict.html', {
        'months': months,
        'visitor_types': visitor_types
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
    purchase_rate = (purchase_count / total_predictions * 100) if total_predictions > 0 else 0
    
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
        'stats': stats
    })

def documentation(request):
    """Documentation view"""
    return render(request, 'myapp/documentation.html')

@login_required
def train_model(request):
    """Trigger model training"""
    # In a production environment, you would start this as a background task
    messages.success(request, "Model training initiated. This may take some time.")
    
    try:
        import train_model
        train_model.train_model()
        messages.success(request, "Model training completed successfully!")
    except Exception as e:
        messages.error(request, f"Error training model: {str(e)}")
    
    return redirect('dashboard')

@login_required
def load_example_data(request):
    """Load example prediction data"""
    if request.method == 'POST':
        try:
            # Create some example predictions
            examples = [
                {
                    'admin_info': 0,
                    'administrative_duration': 0,
                    'informational': 0,
                    'informational_duration': 0,
                    'product_related': 15,
                    'product_related_duration': 346.5,
                    'bounce_rate': 0.2,
                    'exit_rate': 0.2, 
                    'page_value': 0,
                    'special_day': 0,
                    'month': 'Feb',
                    'operating_systems': 1,
                    'browser': 1,
                    'region': 1,
                    'traffic_type': 1,
                    'visitor_type': 'Returning_Visitor',
                    'weekend': True,
                    'prediction': False
                },
                {
                    'admin_info': 3,
                    'administrative_duration': 87.83,
                    'informational': 0,
                    'informational_duration': 0,
                    'product_related': 27,
                    'product_related_duration': 798.33,
                    'bounce_rate': 0,
                    'exit_rate': 0.013,
                    'page_value': 22.92,
                    'special_day': 0.8,
                    'month': 'Feb',
                    'operating_systems': 2,
                    'browser': 2,
                    'region': 3,
                    'traffic_type': 1,
                    'visitor_type': 'Returning_Visitor',
                    'weekend': False,
                    'prediction': True
                },
                {
                    'admin_info': 4,
                    'administrative_duration': 73.5,
                    'informational': 2,
                    'informational_duration': 120,
                    'product_related': 36,
                    'product_related_duration': 998.74,
                    'bounce_rate': 0,
                    'exit_rate': 0.0147,
                    'page_value': 19.45,
                    'special_day': 0.2,
                    'month': 'Feb',
                    'operating_systems': 2,
                    'browser': 2,
                    'region': 4,
                    'traffic_type': 1,
                    'visitor_type': 'Returning_Visitor',
                    'weekend': False,
                    'prediction': False
                },
                {
                    'admin_info': 6,
                    'administrative_duration': 176.25,
                    'informational': 12,
                    'informational_duration': 404,
                    'product_related': 41,
                    'product_related_duration': 2720.67,
                    'bounce_rate': 0.013,
                    'exit_rate': 0.033,
                    'page_value': 8.83,
                    'special_day': 0,
                    'month': 'Mar',
                    'operating_systems': 2,
                    'browser': 2,
                    'region': 2,
                    'traffic_type': 2,
                    'visitor_type': 'Returning_Visitor',
                    'weekend': True,
                    'prediction': True
                }
            ]
            
            # Save examples to database
            for example in examples:
                record = PredictionRecord(
                    user=request.user,
                    admin_info=example['admin_info'],
                    administrative_duration=example['administrative_duration'],
                    informational=example['informational'],
                    informational_duration=example['informational_duration'],
                    product_related=example['product_related'],
                    product_related_duration=example['product_related_duration'],
                    bounce_rate=example['bounce_rate'],
                    exit_rate=example['exit_rate'],
                    page_value=example['page_value'],
                    special_day=example['special_day'],
                    month=example['month'],
                    operating_systems=example['operating_systems'],
                    browser=example['browser'],
                    region=example['region'],
                    traffic_type=example['traffic_type'],
                    visitor_type=example['visitor_type'],
                    weekend=example['weekend'],
                    prediction=example['prediction']
                )
                record.save()
            
            messages.success(request, "Example data has been loaded successfully!")
            return redirect('dashboard')
        except Exception as e:
            messages.error(request, f"Error loading example data: {str(e)}")
            return redirect('dashboard')
    
    # If not POST, render confirmation page
    return render(request, 'myapp/load_examples.html')

# Enhancement #1: Export Predictions to CSV
@login_required
def export_predictions(request):
    """Export predictions as CSV"""
    try:
        # Create a CSV file in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header row
        writer.writerow([
            'Prediction Date', 'User', 'Administrative Pages', 'Administrative Duration', 
            'Informational Pages', 'Informational Duration', 'Product Related Pages', 
            'Product Related Duration', 'Bounce Rate', 'Exit Rate', 'Page Value', 
            'Special Day', 'Month', 'Operating System', 'Browser', 'Region', 
            'Traffic Type', 'Visitor Type', 'Weekend', 'Prediction'
        ])
        
        # Write data rows
        predictions = PredictionRecord.objects.all().order_by('-prediction_date')
        for pred in predictions:
            writer.writerow([
                pred.prediction_date.strftime('%Y-%m-%d %H:%M:%S'),
                pred.user.username,
                pred.admin_info,
                pred.administrative_duration,
                pred.informational,
                pred.informational_duration,
                pred.product_related,
                pred.product_related_duration,
                pred.bounce_rate,
                pred.exit_rate,
                pred.page_value,
                pred.special_day,
                pred.month,
                pred.operating_systems,
                pred.browser,
                pred.region,
                pred.traffic_type,
                pred.visitor_type,
                'Yes' if pred.weekend else 'No',
                'Will Purchase' if pred.prediction else 'No Purchase'
            ])
        
        # Create the HTTP response with the CSV file
        response = HttpResponse(output.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
        return response
    except Exception as e:
        messages.error(request, f"Error exporting data: {str(e)}")
        return redirect('dashboard')

# Enhancement #2: Bulk prediction
@login_required
def bulk_predict(request):
    """Handle bulk prediction from CSV upload"""
    if request.method == 'POST':
        try:
            csv_file = request.FILES.get('csv_file')
            if not csv_file:
                messages.error(request, "No file uploaded!")
                return redirect('bulk_predict')
            
            # Check file extension
            if not csv_file.name.endswith('.csv'):
                messages.error(request, "File must be a CSV!")
                return redirect('bulk_predict')
            
            # Load model
            model = load_model()
            if model is None:
                messages.error(request, "Model not found. Please train the model first!")
                return redirect('dashboard')
            
            # Read CSV
            data = pd.read_csv(csv_file)
            
            # Make predictions
            predictions = model.predict(data)
            
            # Add predictions to the dataframe
            data['Prediction'] = predictions
            data['Prediction_Label'] = data['Prediction'].apply(lambda x: 'Will Purchase' if x else 'No Purchase')
            
            # Store predictions in the database
            saved_count = 0
            for _, row in data.iterrows():
                try:
                    record = PredictionRecord(
                        user=request.user,
                        admin_info=row.get('Administrative', 0),
                        administrative_duration=row.get('Administrative_Duration', 0),
                        informational=row.get('Informational', 0),
                        informational_duration=row.get('Informational_Duration', 0),
                        product_related=row.get('ProductRelated', 0),
                        product_related_duration=row.get('ProductRelated_Duration', 0),
                        bounce_rate=row.get('BounceRates', 0),
                        exit_rate=row.get('ExitRates', 0),
                        page_value=row.get('PageValues', 0),
                        special_day=row.get('SpecialDay', 0),
                        month=row.get('Month', 'Feb'),
                        operating_systems=row.get('OperatingSystems', 1),
                        browser=row.get('Browser', 1),
                        region=row.get('Region', 1),
                        traffic_type=row.get('TrafficType', 1),
                        visitor_type=row.get('VisitorType', 'Returning_Visitor'),
                        weekend=row.get('Weekend', False),
                        prediction=bool(row['Prediction'])
                    )
                    record.save()
                    saved_count += 1
                except Exception as e:
                    continue
            
            # Return the results as CSV
            csv_buffer = io.StringIO()
            data.to_csv(csv_buffer, index=False)
            
            response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="bulk_predictions.csv"'
            
            messages.success(request, f"{saved_count} predictions processed and saved to database!")
            return response
            
        except Exception as e:
            messages.error(request, f"Error processing bulk predictions: {str(e)}")
            return redirect('bulk_predict')
    
    # GET request - show upload form
    return render(request, 'myapp/bulk_predict.html')

# Enhancement #3: Simple API endpoint
def api_predict(request):
    """API endpoint for predictions"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    try:
        # Parse JSON data
        data = json.loads(request.body)
        
        # Load model
        model = load_model()
        if model is None:
            return JsonResponse({'error': 'Model not found'}, status=500)
        
        # Prepare features
        features = {
            'Administrative': float(data.get('administrative', 0)),
            'Administrative_Duration': float(data.get('administrative_duration', 0)),
            'Informational': float(data.get('informational', 0)),
            'Informational_Duration': float(data.get('informational_duration', 0)),
            'ProductRelated': float(data.get('product_related', 0)),
            'ProductRelated_Duration': float(data.get('product_related_duration', 0)),
            'BounceRates': float(data.get('bounce_rate', 0)),
            'ExitRates': float(data.get('exit_rate', 0)),
            'PageValues': float(data.get('page_value', 0)),
            'SpecialDay': float(data.get('special_day', 0)),
            'Month': data.get('month', 'Feb'),
            'OperatingSystems': int(data.get('operating_systems', 1)),
            'Browser': int(data.get('browser', 1)),
            'Region': int(data.get('region', 1)),
            'TrafficType': int(data.get('traffic_type', 1)),
            'VisitorType': data.get('visitor_type', 'Returning_Visitor'),
            'Weekend': data.get('weekend', False),
        }
        
        # Make prediction
        df = pd.DataFrame([features])
        prediction = bool(model.predict(df)[0])
        
        # Return result
        return JsonResponse({
            'prediction': prediction,
            'prediction_label': 'Will Purchase' if prediction else 'No Purchase',
            'features': features
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)