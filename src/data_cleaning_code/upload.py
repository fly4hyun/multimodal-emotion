from minio import Minio # 7.1.14
from minio.commonconfig import CopySource
import os
import shutil
import zipfile


if __name__ == '__main__':
    """
    공식 문서 : https://min.io/docs/minio/linux/developers/python/API.html
    """

    # web 서버주소는 192.168.2.139:6732
    # API web 서버주소는 192.168.2.139:6731
    minioClient = Minio(endpoint="192.168.2.139:6731",
                        access_key="hee",
                        secret_key="!Wkdal8d192b@",
                        secure=False)

    # local에 있는 파일을 minio로 다운로드

    minioClient.fput_object(bucket_name="aaai-emotion", # 파일을 저장한 버킷 선택
                            object_name="video/studio.zip", # 버킷내 파일을 저장 할 경로
                            file_path="../a.ed_code_and_data/studio.zip" # 올리고 싶은 파일
                            )
    
    
    
    # minio에 있는 파일을 local로 다운로드
    minioClient.fget_object(bucket_name="aaai-emotion", # 파일을 불러올  버킷 선택
                            object_name="video/studio.zip",# 버킷내 파일을 불러올 할 경로
                            file_path="./studio.zip" # 저장할 경로 선택
                            )