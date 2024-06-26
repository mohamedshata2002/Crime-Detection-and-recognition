# Generated by Django 4.1.2 on 2024-02-24 11:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0006_alter_video_address'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='video',
            name='address',
        ),
        migrations.AddField(
            model_name='video',
            name='size',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True),
        ),
        migrations.AddField(
            model_name='video',
            name='title',
            field=models.CharField(default='NoTitle', max_length=100),
            preserve_default=False,
        ),
    ]
