

import os
import shutil
import zipfile

list1 = os.listdir('./audio')

i = 0
name_i = 0
for name in list1:
    if i == 52000:
        name_i = name_i + 1
        i = 0
    
    if i == 0:
        os.mkdir('./audio' + str(name_i))
    i = i + 1
    shutil.copyfile('./audio/' + name, './audio' + str(name_i) + '/' + name)
    
    
for i in range(5):
    
    zip_file = zipfile.ZipFile('./audio' + str(i) + ".zip", "w")  # "w": write 모드
    for file in os.listdir('./audio' + str(i)):
        if file.endswith('.wav'):
            zip_file.write(os.path.join('./audio' + str(i), file), compress_type=zipfile.ZIP_DEFLATED)

    zip_file.close()
    
# print(len(list1))


# asdfasdfasdf

# list1 = os.listdir('./audio')
# for name1 in list1:#N
#     list2 = os.listdir('./audio/' + name1)
#     for name2 in list2:#01
#         list3 = os.listdir('./audio/' + name1 + '/' + name2)
        
#         for name3 in list3:
            
#             shutil.move('./audio/' + name1 + '/' + name2 + '/' + name3, './audios/' + name3)
            
            
        

