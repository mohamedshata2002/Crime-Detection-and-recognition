# Generated by Django 4.1.2 on 2024-03-30 02:12

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0012_video_duration_video_size'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='video',
            name='duration',
        ),
        migrations.RemoveField(
            model_name='video',
            name='size',
        ),
    ]
