# Generated by Django 4.1.2 on 2024-02-24 21:01

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0008_customuser'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='video',
            name='size',
        ),
        migrations.DeleteModel(
            name='CustomUser',
        ),
    ]