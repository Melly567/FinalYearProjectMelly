from django.db import models
from django.contrib.auth.models import User

class PredictionRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    admin_info = models.FloatField()
    administrative_duration = models.FloatField()
    informational = models.FloatField()
    informational_duration = models.FloatField()
    product_related = models.FloatField()
    product_related_duration = models.FloatField()
    bounce_rate = models.FloatField()
    exit_rate = models.FloatField()
    page_value = models.FloatField()
    special_day = models.FloatField()
    month = models.CharField(max_length=3)
    operating_systems = models.IntegerField()
    browser = models.IntegerField()
    region = models.IntegerField()
    traffic_type = models.IntegerField()
    visitor_type = models.CharField(max_length=20)
    weekend = models.BooleanField()
    prediction = models.BooleanField()
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.user} on {self.prediction_date}"

class ModelMetrics(models.Model):
    name = models.CharField(max_length=100)
    auc_score = models.FloatField()
    precision_score = models.FloatField()
    recall_score = models.FloatField()
    f1_score = models.FloatField()
    updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.updated_date}"
    