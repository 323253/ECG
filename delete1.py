import os

def delete_files_with_keyword(folder_path, keyword, file_extension):
    # 遍历文件夹中的文件
    for file_name in os.listdir(folder_path):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)

        # 判断文件名是否包含关键字且以指定扩展名结尾
        if keyword in file_name and file_name.endswith(file_extension):
            # 删除文件
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

# 指定文件夹路径、关键字和文件扩展名
folder_path = r'D:\毕设数据'
keyword = 'HeartBeats'
file_extension = '.mat'

# 删除包含关键字的 .m 文件
delete_files_with_keyword(folder_path, keyword, file_extension)
