# from minio import Minio

# #minioClient = Minio("127.0.0.1:9000", access_key="URP7lNBmNT890JTQ", secret_key="DnZ9KAyZXxTomuYi1T22JClm3ojRIuhN")#, secure=False)

# minioClient = Minio("192.168.2.139:6731", access_key="hee", secret_key="!Wkdal8d192b@")#, secure=False)


# minioClient.fget_object('aaai-hee', 'aaai-hee/emotion/audio/novel_audio_emotion/Training.zip', './Training.zip')








from minio import Minio # 7.1.14

if __name__ == '__main__':

    minioClient = Minio(endpoint="192.168.2.139:6731",
    access_key="hee",
    secret_key="!Wkdal8d192b@",
    secure=False)
    
    get_list = minioClient.list_objects(bucket_name="aaai-emotion",
                                        prefix = "audio/emotion_and_speaker_style/Validation", 
                                        # prefix="test_folder/", # 버킷내 특정 prefix만 검색시 사용(prefix는 버킷내 폴더와 같은 의미)
                                        recursive=True # default는 false, false시 버킷의 최상위에 있는 폴더와 파일만 검색, True일 때는 하위 모든 파일 검색 가능
                                        )
    
    for i in get_list:

        print(i.object_name[43:], i.size)
        minioClient.fget_object(bucket_name="aaai-emotion", # 파일을 불러올  버킷 선택
                            object_name="audio/emotion_and_speaker_style/Validation" + i.object_name[42:], 
                            file_path="./Validation/" + i.object_name[43:] # 저장할 경로 선택
                            )

    get_list = minioClient.list_objects(bucket_name="aaai-emotion",
                                        prefix = "audio/emotion_and_speaker_style/Training", 
                                        # prefix="test_folder/", # 버킷내 특정 prefix만 검색시 사용(prefix는 버킷내 폴더와 같은 의미)
                                        recursive=True # default는 false, false시 버킷의 최상위에 있는 폴더와 파일만 검색, True일 때는 하위 모든 파일 검색 가능
                                        )

    for i in get_list:
        print(i.object_name[41:], i.size)
        minioClient.fget_object(bucket_name="aaai-emotion", # 파일을 불러올  버킷 선택
                            object_name="audio/emotion_and_speaker_style/Training" + i.object_name[40:], 
                            file_path="./Training/" + i.object_name[41:] # 저장할 경로 선택
                            )



