
# Python==3.9

from minio import Minio # 7.1.14
from minio.commonconfig import CopySource

if __name__ == '__main__':
    """
    공식 문서 : https://min.io/docs/minio/linux/developers/python/API.html
    """

    # web 서버주소는 192.168.2.139:6732
    # API web 서버주소는 192.168.2.139:6731
    minioClient = Minio(endpoint="192.168.2.139:6731",
                        access_key="[부여받은 ID]",
                        secret_key="[패스워드]",
                        secure=False)

    # local에 있는 파일을 minio로 다운로드
    minioClient.fput_object(bucket_name="aaai-emotion", # 파일을 저장한 버킷 선택
                            object_name="test_folder/superset_testfile.csv", # 버킷내 파일을 저장 할 경로
                            file_path="./superset_testfile.csv" # 올리고 싶은 파일
                            )

    # minio에 있는 파일을 local로 다운로드
    minioClient.fget_object(bucket_name="aaai-emotion", # 파일을 불러올  버킷 선택
                            object_name="test_folder/superset_testfile.csv",# 버킷내 파일을 불러올 할 경로
                            file_path="./superset_testfile2.csv" # 저장할 경로 선택
                            )

    # minio의 버킷 내 파일 검색
    get_list = minioClient.list_objects(bucket_name="aaai-emotion",
                                        # prefix="test_folder/", # 버킷내 특정 prefix만 검색시 사용(prefix는 버킷내 폴더와 같은 의미)
                                        recursive=True # default는 false, false시 버킷의 최상위에 있는 폴더와 파일만 검색, True일 때는 하위 모든 파일 검색 가능
                                        )

    # get_list는 리스트들이 담겨있는 object
    for i in get_list:
        print("파일 이름 :", i.object_name)
        print("파일 사이즈:", i.size)

    # 파일 복제
    result = minioClient.copy_object(bucket_name="aaai-emotion",  # 붙여놓을 버킷
                                     object_name="superset_testfile.csv",  # 버킷내 붙여놓을 파일 이름
                                     source=CopySource("aaai-emotion", "test_folder/superset_testfile.csv")
                                     # 복사할 대상 (버킷, 파일이름)
                                     )

    # 파일 삭제
    minioClient.remove_object(bucket_name="aaai-emotion",
                              object_name="superset_testfile.csv")
    # 다중 파일 삭제
    minioClient.remove_objects(bucket_name="aaai-emotion",
                               delete_object_list=["superset_testfile.csv"])


##