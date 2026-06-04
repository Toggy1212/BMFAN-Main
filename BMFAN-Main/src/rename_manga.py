import os

def rename_files_in_folder(folder_path):
    """
    批量修改指定文件夹内所有文件的名字
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # 如果文件名包含LRBI，进行替换
            if 'LRBI' in filename:
                # 获取文件名中的基名部分，去除'_LRBI_x2', '_LRBI_x3', '_LRBI_x4'
                new_name = filename.replace("_LRBI_x2", "x2").replace("_LRBI_x3", "x3").replace("_LRBI_x4", "x4")
                # 重命名文件
                old_file = os.path.join(folder_path, filename)
                new_file = os.path.join(folder_path, new_name)
                os.rename(old_file, new_file)
                print(f"文件 {filename} 重命名为 {new_name}")

def main():
    # 定义根目录，遍历 X2、X3、X4 文件夹
    root_dir = "/data/sun820851/EDSR-PyTorch-master/dataset/benchmark/Manga109/LR_bicubic"
    
    # 定义要处理的倍率文件夹（X2, X3, X4）
    scales = ['X2', 'X3', 'X4']

    # 遍历每个倍率文件夹进行重命名
    for scale in scales:
        folder_path = os.path.join(root_dir, scale)
        if os.path.isdir(folder_path):
            print(f"正在处理文件夹：{folder_path}")
            rename_files_in_folder(folder_path)
        else:
            print(f"文件夹 {folder_path} 不存在，跳过")

if __name__ == '__main__':
    main()
