# Generated by Django 5.0.6 on 2025-03-03 18:14

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SMApp', '0002_rename_items_item'),
    ]

    operations = [
        migrations.CreateModel(
            name='Floor',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('length', models.IntegerField()),
                ('width', models.IntegerField()),
                ('data', models.JSONField(default=dict)),
            ],
        ),
        migrations.CreateModel(
            name='Shelf',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cell_index', models.IntegerField()),
                ('item_ids', models.JSONField(default=list)),
                ('item_names', models.JSONField(default=list)),
                ('mode', models.CharField(max_length=20)),
                ('floor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='SMApp.floor')),
            ],
        ),
    ]
