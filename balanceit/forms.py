from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit
from .models import Dataset
import json
from website import settings
import os

with open(os.path.join(settings.BASE_DIR, 'static', 'classification.json')) as f:
    ML_DATA = json.load(f)


class DatasetSelectionForm(forms.Form):
    dataset = forms.ModelChoiceField(queryset=Dataset.objects.none(), empty_label=None)

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.layout = Layout('dataset', Submit('next', 'Next'))
        if user:
            self.fields['dataset'].queryset = Dataset.objects.filter(user=user)

class ColumnSelectionForm(forms.Form):
    target_column = forms.ChoiceField(choices=[])

    def __init__(self, *args, **kwargs):
        dataset = kwargs.pop('dataset', None)
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.layout = Layout('target_column', Submit('submit', 'Submit'))
        if dataset:
            # Assuming your dataset file is CSV
            import pandas as pd
            df = pd.read_csv(dataset.file.path)
            self.fields['target_column'].choices = [(col, col) for col in df.columns]

class SamplerSelectionForm(forms.Form):
    sampler = forms.ChoiceField(
        choices=[(sampler['value'], sampler['name']) for sampler in ML_DATA['samplers']],
        widget=forms.Select,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.layout = Layout(
            'sampler',
            Submit('submit', 'Submit')
        )

class SamplerParametersForm(forms.Form):
    def __init__(self, *args, **kwargs):
        sampler = kwargs.pop('sampler', None)
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False

        if sampler:
            sampler_data = next((s for s in ML_DATA['samplers'] if s['value'] == sampler), None)
            if sampler_data:
                for param in sampler_data['parameters']:
                    field_name = param['name']
                    field_type = param['type']
                    field_default = param['default']
                    
                    if field_type == 'float':
                        self.fields[field_name] = forms.FloatField(initial=field_default, required=False)
                    elif field_type == 'int':
                        self.fields[field_name] = forms.IntegerField(initial=field_default, required=False)
                    elif field_type == 'str':
                        self.fields[field_name] = forms.CharField(initial=field_default, required=False)
                    elif field_type == 'choice':
                        choices = [(choice, choice) for choice in param['choices']]
                        self.fields[field_name] = forms.ChoiceField(
                            choices=choices,
                            initial=field_default,
                            required=False,
                            widget=forms.Select
                        )
                    
                    self.fields[field_name].help_text = param['description']

        self.helper.layout = Layout(
            *self.fields,
            Submit('submit', 'Submit', css_class='mt-4 bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded')
        )

class ModelSelectionForm(forms.Form):
    model = forms.ChoiceField(
        choices=[(model['value'], model['name']) for model in ML_DATA['models']],
        widget=forms.Select(attrs={'class': 'form-select mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50'}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.layout = Layout(
            'model',
            Submit('next', 'Next', css_class='mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded')
        )

class ModelParametersForm(forms.Form):
    def __init__(self, *args, **kwargs):
        model = kwargs.pop('model', None)
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False

        if model:
            model_data = next((m for m in ML_DATA['models'] if m['value'] == model), None)
            if model_data:
                for param in model_data['parameters']:
                    field_name = param['name']
                    field_type = param['type']
                    field_default = param['default']
                    
                    if field_type == 'float':
                        self.fields[field_name] = forms.FloatField(initial=field_default, required=False)
                    elif field_type == 'int':
                        self.fields[field_name] = forms.IntegerField(initial=field_default, required=False)
                    elif field_type == 'str':
                        self.fields[field_name] = forms.CharField(initial=field_default, required=False)
                    elif field_type == 'bool':
                        self.fields[field_name] = forms.BooleanField(initial=field_default, required=False)
                    elif field_type == 'choice':
                        choices = [(choice, choice) for choice in param['choices']]
                        self.fields[field_name] = forms.ChoiceField(
                            choices=choices,
                            initial=field_default,
                            required=False,
                            widget=forms.Select(attrs={'class': 'form-select mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50'})
                        )
                    
                    self.fields[field_name].help_text = param['description']

        self.helper.layout = Layout(
            *self.fields,
            Submit('submit', 'Submit', css_class='mt-4 bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded')
        )
        