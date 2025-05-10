from django.contrib import admin
from .models import PredictionRecord, ModelMetrics

@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ('user', 'prediction', 'prediction_date')
    list_filter = ('prediction', 'prediction_date', 'visitor_type', 'month')
    search_fields = ('user__username',)

@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = ('name', 'auc_score', 'precision_score', 'recall_score', 'f1_score', 'updated_date')
    
