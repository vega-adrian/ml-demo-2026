import os
from google.cloud import bigquery


if os.getenv('ENV', 'local') == 'local':
    from dotenv import load_dotenv
    load_dotenv()


client = bigquery.Client()


# Create table
table_schema = [
    bigquery.SchemaField(name='partition_date', field_type='DATE'),
    bigquery.SchemaField(name='unique_id', field_type='STRING'),
    bigquery.SchemaField(
        name='input_features',
        field_type='RECORD',
        fields=[
            bigquery.SchemaField(name='sepal_length', field_type='FLOAT'),
            bigquery.SchemaField(name='sepal_width', field_type='FLOAT'),
            bigquery.SchemaField(name='petal_length', field_type='FLOAT'),
            bigquery.SchemaField(name='petal_width', field_type='FLOAT'),
        ]
    ),
    bigquery.SchemaField(
        name='output_results',
        field_type='RECORD',
        fields=[
            bigquery.SchemaField(name='class_idx', field_type='INTEGER'),
            bigquery.SchemaField(name='class_name', field_type='STRING'),
        ]
    ),
    bigquery.SchemaField(name='insertion_timestamp', field_type='TIMESTAMP'),
]

dataset_reference = bigquery.DatasetReference(
    project=os.getenv('PROJECT_ID'),
    dataset_id=os.getenv('DATASET_ID'),
)
table_ref = bigquery.TableReference(
    dataset_ref=dataset_reference,
    table_id=os.getenv('TABLE_ID'),
)
table = bigquery.Table(table_ref, table_schema)
table.time_partitioning = bigquery.TimePartitioning(field='partition_date')


client.create_table(table, exists_ok=True)