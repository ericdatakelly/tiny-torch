import boto3

s3 = boto3.client('s3')
with open('data_test_download.txt', 'wb') as f:
    s3.download_fileobj('tiny-torch', 'artifact_test/data_test.txt', f)
