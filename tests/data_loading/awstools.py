# demo for upload and access data in S3
# requirements:
# * install aws client
# * install boto3
import boto3
from botocore.exceptions import ClientError
import os
import io
import pickle
import base64

PROJECT_BUCKET = "cdi-lab-workspaces"

class S3Handler:
    def __init__(self):
        self.client_s3 = boto3.client("s3")

    def put_file(self, path, newname=None, bucket=PROJECT_BUCKET):
        """Upload a file to an S3 bucket
        :param path: path of file to upload, e.g., 'current_directory/progress.json'
        :param bucket: Bucket to upload to
        :param newname: S3 object name. 'user_id_hased/session_id_hased/progress.json'
                        If not specified then file_name in path is used
        :return: True if file was uploaded, else False
        upload same name file will replace previous file
        """
        # If S3 object_name was not specified, use file_name in path/file_name
        if newname is None:
            newname = os.path.basename(path)
        # Upload the file
        # client_s3 = boto3.client('s3')
        try:
            response = self.client_s3.upload_file(path, bucket, newname)
        except ClientError as e:
            print(e)
            return False
        return True
    
    def fetch_file(self, path, newname=None, bucket=PROJECT_BUCKET):
        """
        path: str, e.g., 'user_id_hased/session_id_hased/params.json'
        bucket: str
        newname: str, new file name, e.g., 'parameters'
        dowload file from s3 to current directory
        """
        # If S3 object_name was not specified, use file_name
        if newname is None:
            # e.g., params.json in 'user_id_hased/session_id_hased/params.json'
            newname = os.path.basename(path)
        # download the file
        try:
            response = self.client_s3.download_file(bucket, path, newname)
        except ClientError as e:
            print(e)
            return False
        return True
        
    def open_file(self, path, bucket=PROJECT_BUCKET):
        """
        path: str, e.g., 'user_id_hased/session_id_hased/params.json'
        bucket: str
        open file from s3 without downloading
        usage example: 
            with s3_handler.open_file('test_user_1/test_session_1/report.json') as infile: 
                l2 = json.load(infile)
        """
        # download the file
        try:
            # remove read() so that can pickle
            res_data_to_read = self.client_s3.get_object(Bucket=bucket, Key=path)['Body'] #.read()
        except ClientError as e:
            print(e)
            return None
        return res_data_to_read
    
    def open_model(self, path, bucket=PROJECT_BUCKET):
        """
        example application:
        with s3_handler.open_model('test_user_1/test_session_1/model.pkl') as infile: 
            model = joblib.load(infile)
        """
        obj = self.client_s3.get_object(Bucket=bucket, Key=path)
        bytestream = io.BytesIO(obj["Body"].read())
        return bytestream

    def get_filenames(self, path='', bucket=PROJECT_BUCKET):
        """
        path: str, e.g., 'test_user_1/test_session_1', if blank '', 
                all files under all folders bucket will be returned, but not include folder name
        return: list of file names in a bucket, 
                when path='', or there are subfolders under the path
                may include duplicates filenames saved in different folders
                also '' if the folder if empty
        """
        response = self.client_s3.list_objects_v2(
                                            Bucket=bucket,
                                            Prefix=path)
        filenames = [os.path.split(content['Key'])[-1] for content in response.get('Contents', [])]
        return filenames
    
    def delete_file(self, path, bucket=PROJECT_BUCKET):
        """delete file in s3 bucket
        path: str, file path, e.g., 'invoices/January.pdf'
        return: bool
        """
        try:
            response = self.client_s3.delete_object(
                Bucket=bucket,
                Key=path
            )
        except ClientError as e:
            print(e)
            return False
        return True

    def load_pkl_from_s3(self, user_id_session_id, filename='routes.pkl'):
        """ 
        user_id_session_id: e.g., 'user_id/session_id/'
        filename: str, routes.pkl, routes_anal.pkl, chiral_rxns.pkl
        """
        # defualt  bucket=USER_BUCKET
        files_search_folder = self.get_filenames(path=user_id_session_id)
        if filename not in files_search_folder:
            raise FileNotFoundError('{} data not found.'.format('routes'))
        else:
            path_routes = os.path.join(user_id_session_id, filename)
            res_data = self.open_file(path=path_routes)
            res = pickle.load(res_data)
        return res
    
    def get_secret(self, secret_name, region_name='ap-southeast-1'):
        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        try:
            response = client.get_secret_value(SecretId=secret_name)
        except ClientError as e:
            # Handle error accordingly
            raise e
        else:
            # Decrypts secret using the associated KMS key.
            if 'SecretString' in response:
                return response['SecretString']
            else:
                return base64.b64decode(response['SecretBinary'])