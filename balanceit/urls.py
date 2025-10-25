from django.urls import path
from . import views
from .views import ClassifyWizard
urlpatterns = [
    path('', views.home, name='home'),
    path('analyze/', views.analyze, name='analyze'),
    path('classify/', ClassifyWizard.as_view(), name='classify'),
    path('train/', views.training, name='training'),
    path('update-steps/', views.update_steps, name='update_steps'),
    path('results/', views.results, name='results'),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("register/", views.register, name="register"),
    path("my_datasets/", views.my_datasets, name="my_datasets"),
    path("view_dataset/<int:dataset_id>", views.view_dataset, name="view_dataset"),
    path("delete_dataset/<int:dataset_id>", views.delete_dataset, name="delete_dataset"),
    path("my_classifications/", views.my_classifications, name="my_classifications"),
    path("view_classification/<int:classification_id>", views.view_classification, name="view_classification"),
    path("delete_classification/<int:classification_id>", views.delete_classification, name="delete_classification"),
    path('contact/', views.contact, name='contact'),
]