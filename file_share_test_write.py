import boto3

s3 = boto3.resource('s3')

with open('artifact_test/data_test.txt', 'w') as f:
    f.write('fake data')
    s3.Bucket('tiny-torch').put_object(Key='data_test.txt', Body=f)
