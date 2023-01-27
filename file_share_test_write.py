import boto3

s3 = boto3.resource('s3')

with open('data_test.txt', 'r') as f:
    data = f.read()

s3.Bucket('tiny-torch').put_object(Key='artifact_test/data_test.txt', Body=data)

bp = 0
