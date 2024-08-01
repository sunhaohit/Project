import os


def delete_missing_files(target_folder, reference_folder):
    """
    有些模型没有点云数据，但是上一步在npz2V.py中把全部文件都转换为VE矩阵了。所以需要删除一些
    :param target_folder:
    :param reference_folder:
    :return:
    """
    # 获取参考文件夹中所有文件的名称（不包括子文件夹中的文件）
    reference_files = set(os.listdir(reference_folder))

    # 遍历目标文件夹中的所有文件和文件夹
    for item in os.listdir(target_folder):
        # 构造目标文件夹中每个文件的完整路径
        target_path = os.path.join(target_folder, item)

        # 排除子文件夹，只处理文件
        if os.path.isfile(target_path):
            # 如果文件不在参考文件夹中，则删除它

            if item[:-4] + ".ply" not in reference_files:
                os.remove(target_path)
                print(f"Deleted: {target_path}")


def delete_missing_files2(target_folder, reference_folder):
    """
    有些模型部件的数量超过8个了，在上一步在npz2V.py中没有转换。所以需要把它对应的点云原始数据删除
    :param target_folder:
    :param reference_folder:
    :return:
    """
    # 获取参考文件夹中所有文件的名称（不包括子文件夹中的文件）
    reference_files = set(os.listdir(reference_folder))

    # 遍历目标文件夹中的所有文件和文件夹
    for item in os.listdir(target_folder):
        # 构造目标文件夹中每个文件的完整路径
        target_path = os.path.join(target_folder, item)

        # 排除子文件夹，只处理文件
        if os.path.isfile(target_path):
            # 如果文件不在参考文件夹中，则删除它

            if item[:-4] + ".pkl" not in reference_files:
                os.remove(target_path)
                print(f"Deleted: {target_path}")


target_folder = 'VE'
reference_folder = 'pointcloud'
delete_missing_files(target_folder, reference_folder)

target_folder = 'pointcloud'
reference_folder = 'VE'
delete_missing_files2(target_folder, reference_folder)
