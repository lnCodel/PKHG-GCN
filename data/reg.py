##将 3min向pre配准。1.获取文件路径；2.读取数据，数据格式为 ants.core.ants_image.ANTsImage；3.进行配准，方法为Affine；4.保存配准结果。
import multiprocessing

import ants
import os
import shutil
from multiprocessing import Pool
from functools import partial
path = r"/home/lining/Data/skull"
sa_path = r"/home/lining/Data/Registration"
move_path = r'/home/lining/Data/ASPECT_MNI2_masked.nii.gz'  ##fix

def registration_to_MNI(args, prefix="SyNCC", flag=False,additional_files = None):
    file, out_base = args
    file_name = os.path.basename(file)
    if os.path.exists(out_base):
        test = os.path.join(out_base, f'{prefix}_{file_name}')
        if os.path.exists(test):
            return
        else:
            pass
    else:
        os.mkdir(out_base)
    moving_path = r'/home/lining/Data/ASPECT_MNI2_masked.nii.gz'
    print(file_name)
    fixed_path = file

    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)

    if flag:
        pat_id = moving_path.split(os.path.sep)[-2]
        out_dir = os.path.join(out_base, pat_id)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = out_base


    mni_file_path = os.path.join(out_dir, f'{prefix}_{file_name}')
    mat_path = os.path.join(out_dir, 'Affine_MNI.mat')

    print(f'Registering {fixed} to MNI...')
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform=prefix)
    print('Done!')
    ants.image_write(mytx['warpedmovout'], mni_file_path)
    shutil.copyfile(mytx['fwdtransforms'][-1], mat_path)

    if additional_files is not None:
        for f in additional_files:
            moving = ants.image_read(f)
            file_name = os.path.basename(f)
            mni_file_path = os.path.join(out_dir, f'{prefix}_{file_name}')
            print(f'Registering {f} to MNI...')
            if file_name == 'VessTerritory_f3d.nii.gz':
                warped_img = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'],
                                                   interpolator="genericLabel")
            else:
                warped_img = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'],
                                                   interpolator="linear")
            ants.image_write(warped_img, mni_file_path)
    elif additional_files == None:

        warped_img = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'], interpolator="genericLabel")
        ants.image_write(warped_img, mni_file_path)



if __name__ == '__main__':
        print(multiprocessing.cpu_count())
        p = Pool(8)
        pat_list = os.listdir(path)
        pat_list1 = list(map(partial(os.path.join, path), pat_list))
        for i in pat_list1:
            print(i)
        feature_list = [os.path.join(sa_path,i.replace(".nii.gz","")) for i in pat_list]
        for i in feature_list:
            print(i)
        p.map(registration_to_MNI, zip(pat_list1, feature_list))
        p.close()
        p.join()





