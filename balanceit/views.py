from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import render
from django.db import IntegrityError
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from .models import User, Dataset, Classification
import pandas as pd
from balanceit.analyze import plot_dataset
import logging
from formtools.wizard.views import SessionWizardView
from . import forms
from urllib.parse import urlencode
import json
from website import settings
import os
from .script import clean_text
import nltk
from nltk.corpus import stopwords
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from .script import get_sampler, get_model, compute_metrics

with open(os.path.join(settings.BASE_DIR, 'static', 'classification.json')) as f:
    ML_DATA = json.load(f)

class ClassifyWizard(SessionWizardView):
    form_list = [
                forms.DatasetSelectionForm, 
                forms.ColumnSelectionForm, 
                forms.SamplerSelectionForm, 
                forms.SamplerParametersForm,
                forms.ModelSelectionForm,
                forms.ModelParametersForm
                ]
    template_name = "balanceit/classify.html"

    def get_form_kwargs(self, step):
        kwargs = super().get_form_kwargs(step)
        if step == '0':
            kwargs['user'] = self.request.user
        elif step == '1':
            dataset_pk = self.get_cleaned_data_for_step('0')['dataset'].pk
            kwargs['dataset'] = Dataset.objects.get(pk=dataset_pk)
        elif step == '3':  # SamplerParametersForm
            sampler_data = self.get_cleaned_data_for_step('2')
            if sampler_data:
                kwargs['sampler'] = sampler_data['sampler']
        elif step == '5':  # ModelParametersForm
            model_data = self.get_cleaned_data_for_step('4')
            if model_data:
                kwargs['model'] = model_data['model']
        return kwargs

    def done(self, form_list, **kwargs):
        # Process the final form submission
        dataset = form_list[0].cleaned_data['dataset']
        target_column = form_list[1].cleaned_data['target_column']
        selected_sampler = form_list[2].cleaned_data['sampler']
        sampler_parameters = form_list[3].cleaned_data
        selected_model = form_list[4].cleaned_data['model']
        model_parameters = form_list[5].cleaned_data

        # Get the selected sampler details
        selected_sampler_data = next((sampler for sampler in ML_DATA['samplers'] if sampler['value'] == selected_sampler), None)
        selected_model_data = next((model for model in ML_DATA['models'] if model['value'] == selected_model), None)

        context = {
            'dataset': dataset,
            'target_column': target_column,
            'sampler': selected_sampler_data['name'] if selected_sampler_data else 'Unknown',
            'sampler_parameters': sampler_parameters,
            'model': selected_model_data['name'] if selected_model_data else 'Unknown',
            'model_parameters': model_parameters
        }

        return render(self.request, 'balanceit/preview.html', {'context': context ,'query': urlencode(context)})

# Create your views here.
logger = logging.getLogger(__name__)

@csrf_exempt
def update_steps(request):
    global STEPS, DF, TFIDF_VECTORS, VECTORIZER
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X_RESAMPLED, Y_RESAMPLED
    global MODEL, PREDICTIONS, Y_SCORE
    
    step_index = int(request.POST.get('step', 0))
    
    if step_index < len(STEPS):
        if step_index == 0:
            # Load dataset
            dataset_name = request.POST.get('dataset')
            dataset = Dataset.objects.filter(name=dataset_name).first()
            DF = pd.read_csv(dataset.file)
            STEPS[0]["status"] = "completed"

            row_count = len(DF)
            STEPS[0]["output"] = f"Dataset loaded successfully. {row_count} were loaded."
        elif step_index == 1:
            # Download stopwords
            nltk.download('stopwords')
            STEPS[1]["status"] = "completed"
            STEPS[1]["output"] = ""
        elif step_index == 2:
            # Clean text columns
            target = request.POST.get('target')
            stop_words = set(stopwords.words('english'))
            DF = DF.apply(lambda col: col.apply(lambda x: clean_text(x, stop_words)) if col.name != target else col)
            STEPS[2]["status"] = "completed"
            STEPS[2]["output"] = "Removed stopwords and applied text cleaning functions."
        elif step_index == 3:
            # Vectorize text data
            VECTORIZER = TfidfVectorizer()
            TFIDF_VECTORS = VECTORIZER.fit_transform(DF['text'].tolist())
            STEPS[3]["status"] = "completed"
            STEPS[3]["output"] = f"Text data vectorized successfully. Vocabulary size: {len(VECTORIZER.vocabulary_)}"
        elif step_index == 4:
            # Split data into train and test sets
            target = request.POST.get('target')
            X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
                TFIDF_VECTORS, DF[target], test_size=0.2, random_state=42
            )
            STEPS[4]["status"] = "completed"

            train_rows = X_TRAIN.shape[0]
            test_rows = X_TEST.shape[0]
            STEPS[4]["output"] = f"Training set: {train_rows} rows, Test set: {test_rows} rows."
        elif step_index == 5:
            sampler_name = request.POST.get('sampler')
            sampler_parameters = request.POST.get('sampler_parameters')

            sampler = get_sampler(sampler_name, sampler_parameters)            
            if sampler is None:
                X_RESAMPLED, Y_RESAMPLED = X_TRAIN, Y_TRAIN
                STEPS[5]["output"] = f"No resampling applied as no sampler was selected."
                STEPS[5]["status"] = "completed"
            else:
                X_RESAMPLED, Y_RESAMPLED = sampler.fit_resample(X_TRAIN, Y_TRAIN)
                STEPS[5]["output"] = f"Resampling applied using {sampler_name}."
                STEPS[5]["status"] = "completed"
        elif step_index == 6:
            model_name = request.POST.get('model')
            model_parameters = request.POST.get('model_parameters')

            MODEL = get_model(model_name, model_parameters)
            MODEL.fit(X_RESAMPLED, Y_RESAMPLED)
            STEPS[6]["output"] = f"Model trained successfully using {model_name}."
            STEPS[6]["status"] = "completed"
        elif step_index == 7:

            PREDICTIONS = MODEL.predict(X_TEST)
            Y_SCORE = MODEL.predict_proba(X_TEST)
            
            metrics = compute_metrics(Y_TEST, PREDICTIONS, Y_SCORE)

            dataset = Dataset.objects.filter(name=request.POST.get('dataset')).first()
            classification = Classification(dataset=dataset, target_column=request.POST.get('target'), 
                                            sampler=request.POST.get('sampler'), sampler_parameters=request.POST.get('sampler_parameters'),
                                            model=request.POST.get('model'), model_parameters=request.POST.get('model_parameters'),
                                            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
                                            f1_score=metrics['f1'], roc_auc=metrics['roc_auc'], gmean=metrics['gmean'], mcc=metrics['mcc'])
            classification.save()

            STEPS[7]["output"] += f"\nClassification results saved to database with ID: {classification.id}"
            STEPS[7]["status"] = "completed"
        
        step_index += 1

    completed = all(step["status"] == "completed" for step in STEPS)
    response_data = {
        "steps": STEPS, 
        "completed": completed, 
        "next_step": step_index
    }
    
    if completed and DF is not None:
        response_data["final_output"] = "All steps completed successfully!"

    return JsonResponse(response_data)

STEPS = []
DF = None
VECTORIZER = None
TFIDF_VECTORS = None
X_TRAIN = None
X_TEST = None
Y_TRAIN = None
Y_TEST = None
X_RESAMPLED = None
Y_RESAMPLED = None
MODEL = None
PREDICTIONS = None
Y_SCORE = None

def training(request):
    global STEPS, DF

    dataset = request.GET.get('dataset')
    target = request.GET.get('target_column')
    sampler = request.GET.get('sampler')
    sampler_parameters = request.GET.get('sampler_parameters')
    model = request.GET.get('model')
    model_parameters = request.GET.get('model_parameters')

    STEPS = [
        {"message": "Loading dataset", "status": "pending", "output": ""},
        {"message": "Downloading stopwords", "status": "pending", "output": ""},
        {"message": "Cleaning text columns", "status": "pending", "output": ""},
        {"message": "Vectorizing text data", "status": "pending", "output": ""},
        {"message": "Splitting data into train and test sets", "status": "pending", "output": ""},
        {"message": "Applying resampling for balancing", "status": "pending", "output": ""},
        {"message": "Training the model", "status": "pending", "output": ""},
        {"message": "Making predictions and computing metrics", "status": "pending", "output": ""}
    ]

    DF = None

    context = {
        'dataset': dataset,
        'target': target,
        'sampler': sampler,
        'sampler_parameters': sampler_parameters,
        'model': model,
        'model_parameters': model_parameters,
    }

    return render(request, 'balanceit/training.html', context)

def home(request):
    return render(request, 'balanceit/home.html')

def login_view(request):
    if request.method == "POST":
        print(request.POST)
        # Attempt to sign user in
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)

        # Check if authentication successful
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse("home"))
        else:
            return render(request, "balanceit/login.html", {
                "message": "Invalid username and/or password."
            })
    else:
        return render(request, "balanceit/login.html")

def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse("home"))

def register(request):
    if request.method == "POST":

        username = request.POST["floating_username"]
        email = request.POST["floating_email"]

        # Ensure password matches confirmation
        password = request.POST["floating_password"]
        confirmation = request.POST["repeat_password"]
        if password != confirmation:
            return render(request, "balanceit/register.html", {
                "message": "Passwords must match."
            })

        first_name = request.POST["floating_first_name"]
        last_name = request.POST["floating_last_name"]
        phone = request.POST["floating_phone"]
        company = request.POST["floating_company"]

        # Attempt to create new user
        try:
            user = User.objects.create_user(username=username, email=email, 
                                            password=password, first_name=first_name, 
                                            last_name=last_name, phone=phone, company=company)
            user.save()
        except IntegrityError:
            return render(request, "balanceit/register.html", {
                "message": "Username already taken."
            })
        login(request, user)
        return HttpResponseRedirect(reverse("home"))
    else:
        return render(request, "balanceit/register.html")

def analyze(request):
    if request.method == 'POST' and request.FILES.get('dataset', None):
        dataset = request.FILES['dataset']
        try:
            df = pd.read_csv(dataset)
            if not Dataset.objects.filter(user=request.user, name=dataset.name).exists():
                my_dataset = Dataset(user=request.user, name=dataset.name, file=dataset)
                my_dataset.save()
            return render(request, 'balanceit/results.html', {'analytics': plot_dataset(df, df['label']), 
                                                              'filename': dataset.name, 
                                                              'df_head': df.head().to_dict(orient='records')
                                                            })
        except Exception as e:
            logger.error("Error processing dataset", exc_info=True)
            return HttpResponse(f"Error processing dataset: {str(e)}")
    elif request.method == 'POST' and request.POST.get('dataset', None):
        try:
            filename = request.POST.get('dataset', None)
            dataset = Dataset.objects.get(user=request.user, file=filename)
            df = pd.read_csv(dataset.file)
            return render(request, 'balanceit/results.html', {'analytics': plot_dataset(df, df['label']), 
                                                              'filename': dataset.name, 
                                                              'df_head': df.head().to_dict(orient='records')
                                                            })
        except Exception as e:
            logger.error("Error processing dataset", exc_info=True)
            return HttpResponse(f"Error processing dataset: {str(e)}")
    return render(request, 'balanceit/analyze.html')

def results(request):
    dataframe_html = request.session.get('dataframe', None)
    if dataframe_html:
        return render(request, 'balanceit/results.html', {'dataframe': dataframe_html})
    return HttpResponse("No results to display.")

@login_required
def my_datasets(request):
    if request.method == 'POST':
        dataset = request.FILES['dataset']

        try:
            df = pd.read_csv(dataset)
            my_dataset = Dataset(user=request.user, name=dataset.name, file=dataset)
            my_dataset.save()
            return render(request, 'balanceit/datasets.html', {'datasets': Dataset.objects.filter(user=request.user)})
        except Exception as e:
            logger.error("Error processing dataset", exc_info=True)
            return HttpResponse(f"Error processing dataset: {str(e)}")
    else:
        return render(request, 'balanceit/datasets.html', {'datasets': Dataset.objects.filter(user=request.user)})

@login_required
def view_dataset(request, dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)
    df = pd.read_csv(dataset.file)
    headers = df.columns.tolist()

    df_dict = df.to_dict(orient='records')

    paginator = Paginator(df_dict, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'balanceit/view_dataset.html', 
                  {
                    'page_obj': page_obj, 
                   'headers': headers, 
                   'name': dataset.name
                })

@login_required
def delete_dataset(request, dataset_id):
    dataset = Dataset.objects.get(id=dataset_id)
    dataset.delete()
    return HttpResponseRedirect(reverse('my_datasets'))

@login_required
def my_classifications(request):
    return render(request, 'balanceit/classifications.html', {'classifications': Classification.objects.filter(dataset__user=request.user)})

@login_required
def view_classification(request, classification_id):
    classification = Classification.objects.get(id=classification_id)
    return render(request, 'balanceit/view_classification.html', {'classification': classification})

@login_required
def delete_classification(request, classification_id):
    classification = Classification.objects.get(id=classification_id)
    classification.delete()
    return HttpResponseRedirect(reverse('my_classifications'))

def contact(request):
    return render(request, 'balanceit/contact.html')