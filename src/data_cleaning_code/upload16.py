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
    


    list_name = []

    zip_file = zipfile.ZipFile('./Emotion_and_speaker_style/audio16.zip', "w")  # "w": write 모드
    for file in os.listdir('./Emotion_and_speaker_style/audio16'):
        if file.endswith('.wav'):
            zip_file.write(os.path.join('./Emotion_and_speaker_style/audio16', file), compress_type=zipfile.ZIP_DEFLATED)

    zip_file.close()
    list_name.append('audio16.zip')
    shutil.rmtree('./Emotion_and_speaker_style/audio16')

    
    
    
    
    for name in list_name:
        minioClient.fput_object(bucket_name="aaai-emotion", # 파일을 저장한 버킷 선택
                                object_name="audio/emotion_and_speaker_style/" + name, # 버킷내 파일을 저장 할 경로
                                file_path="./Emotion_and_speaker_style/" + name # 올리고 싶은 파일
                                )
        
        
    minioClient.fget_object(bucket_name="aaai-emotion", # 파일을 불러올  버킷 선택
                    object_name="audio/multi_modal_audio/audio.zip", 
                    file_path="./multi_modal_audio/audio/audio.zip" # 저장할 경로 선택
                    )
    

    zipfile.ZipFile("./multi_modal_audio/audio/audio.zip").extractall()
    
    os.remove("./multi_modal_audio/audio/audio.zip")