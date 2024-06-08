from django import forms
from .models import Video, UserProfile
from django.contrib.auth.forms import UserCreationForm

class UserSignUpForm(UserCreationForm):
    address = forms.CharField(widget=forms.TextInput)
    phone = forms.CharField(widget=forms.TextInput)
    age = forms.IntegerField()
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'maxlength': 30}))
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput(attrs={'maxlength': 30}))

    class Meta(UserCreationForm.Meta):
        fields = UserCreationForm.Meta.fields + ('address', 'phone', 'age')

    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.save()
            UserProfile.objects.create(user=user, address=self.cleaned_data['address'], phone=self.cleaned_data['phone'], age=self.cleaned_data['age'])
        return user


class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['video']