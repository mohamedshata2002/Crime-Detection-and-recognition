# Generated by Django 4.1.2 on 2024-02-24 02:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0005_remove_video_size_remove_video_title_video_address'),
    ]

    operations = [
        migrations.AlterField(
            model_name='video',
            name='address',
            field=models.CharField(default='NO ADDR', max_length=255),
        ),
    ]
