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
    

    
    list1 = os.listdir('./Emotion_and_speaker_style/audio')
    list_name = []
    i = 0
    name_i = 0
    for name in list1:
        if i == 52000:
            
            zip_file = zipfile.ZipFile('./Emotion_and_speaker_style/audio' + str(name_i) + ".zip", "w")  # "w": write 모드
            for file in os.listdir('./Emotion_and_speaker_style/audio' + str(name_i)):
                if file.endswith('.wav'):
                    zip_file.write(os.path.join('./Emotion_and_speaker_style/audio' + str(name_i), file), compress_type=zipfile.ZIP_DEFLATED)

            zip_file.close()
            list_name.append('audio' + str(name_i) + ".zip")
            shutil.rmtree('./Emotion_and_speaker_style/audio' + str(name_i))
            name_i = name_i + 1
            i = 0
        
        if i == 0:
            os.mkdir('./Emotion_and_speaker_style/audio' + str(name_i))
        i = i + 1
        shutil.copyfile('./Emotion_and_speaker_style/audio/' + name, './Emotion_and_speaker_style/audio' + str(name_i) + '/' + name)
        
    
    for name in list_name:
        minioClient.fput_object(bucket_name="aaai-emotion", # 파일을 저장한 버킷 선택
                                object_name="audio/emotion_and_speaker_style/" + name, # 버킷내 파일을 저장 할 경로
                                file_path="./Emotion_and_speaker_style/" + name # 올리고 싶은 파일
                                )