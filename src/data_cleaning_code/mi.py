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

    # 작은 파일로 테스트
    minioClient.fget_object(bucket_name="aaai-hee",
    object_name="emotion/audio/novel_audio_emotion/Validation/validation_label.json",
    file_path="./validation_label.json")

    # 실제 받고 싶은 파일
    minioClient.fget_object(bucket_name="aaai-hee",
    object_name="emotion/audio/novel_audio_emotion/Training.zip",
    file_path="./Training.zip")
