import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import pandas as pd 

def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(90),
        T.RandomCrop(256, padding=32),
        T.ToTensor(),
        T.Normalize((0.456, 0.456, 0.456), (0.224, 0.224, 0.224))
    ])
def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.456, 0.456, 0.456), (0.224, 0.224, 0.224))
    ])


def imglist_generation(csv_path, patient_id_path, target_percent): # split datasets by DICOM patient IDs

    big_df = pd.read_csv(patient_id_path)
    name_df = big_df[["ID", "PatientID"]]
    name_df.set_index("PatientID", inplace = True)
    name_list_df = pd.read_csv(csv_path)
    name_list_df = name_list_df["0"]
    name_list = name_list_df.values.tolist()


    train_names1, train_names2= train_test_split(name_list, test_size = 0.2, random_state=7, shuffle=True)
    train_imgs1 = []
    train_imgs2 = []
    for i in range(len(train_names1)):
        train_imgs1.append(name_df.loc[train_names1[i]].values.tolist())
    for i in range(len(train_names2)):
        train_imgs2.append(name_df.loc[train_names2[i]].values.tolist())
    train_imgs1 = [item for sublist in train_imgs1 for item in sublist]
    train_imgs2 = [item for sublist in train_imgs2 for item in sublist]

    train_imgs = train_imgs1
    val_imgs = train_imgs2

    return train_imgs, val_imgs
