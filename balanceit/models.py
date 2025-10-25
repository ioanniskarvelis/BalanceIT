from django.contrib.auth.models import AbstractUser
from django.db import models
from django.db.models.deletion import CASCADE

# Create your models here.

class User(AbstractUser):
    phone = models.CharField(max_length=10, blank=True)
    company = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"{self.username}"
    

class Dataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name}"

class Classification(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='classifications')
    created_at = models.DateTimeField(auto_now_add=True)

    # Training parameters
    target_column = models.CharField(max_length=100)
    
    # Sampler parameters
    sampler = models.CharField(max_length=100)
    sampler_parameters = models.JSONField(default=dict)

    # Model parameters
    model = models.CharField(max_length=100)
    model_parameters = models.JSONField(default=dict)

    # Evaluation metrics
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    roc_auc = models.FloatField()
    gmean = models.FloatField()
    mcc = models.FloatField()

    def __str__(self):
        return f"Classification for {self.dataset.name} on {self.created_at}"