import glob
import shutil
import os


def transfer(name):
    # sideOfface = input("enter side of Face: ")
    # side_list = os.listdir("Dataset/" + name)
    directories = [d for d in os.listdir(
        "Dataset/" + name + "/") if os.path.isdir(os.path.join("Dataset/" + name + "/", d))]
    # print(directories)
    # print(d)
    # side_list = [d for d in side_list if os.path.isdir(d)]
    # print(directories)
    for sub_dir in directories:
        src_dir = "Dataset/" + name + "/" + sub_dir + "/"
        # print(src_dir)
        dst_dir = "Dataset/" + name + "/"
        # print(name)
        for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
            print(jpgfile)
    #         # print()
            shutil.copy(jpgfile, dst_dir)
        shutil.rmtree("Dataset/" + name + "/" + sub_dir + "/")

    # for sub_dir in side_list:
    #     print(sub_dir)
    # os.rmdir("Dataset/" + name + "/" + sub_dir + "/")


dir_list = os.listdir("Dataset/")
print(dir_list)
for d in dir_list:
    transfer(d)
# print(d)
