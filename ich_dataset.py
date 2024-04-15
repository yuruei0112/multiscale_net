import torch 

class ICHDataset(Dataset):
    def __init__(self, imgs,img_dir, img_size, csv_path ,mode="train", transforms= None):
        super().__init__()
        self.imgs = imgs
        self.mode = mode
        self.transforms = transforms
        self.img_dir = img_dir
        self.img_size = img_size 
        self.img_df = pd.read_csv(csv_path)
    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        image_name = str(image_name)[2:14]
        path = os.path.join(self.img_dir, image_name)
        img = Image.open(path)
        img = img.resize((self.img_size, self.img_size))

        if self.mode == "train":
            label = self.img_df.loc[[image_name]]
            label = torch.tensor(label.values, dtype = torch.float32)
            img = self.transforms(img)

            return  img, label
        elif self.mode == "val":
            label = self.img_df.loc[[image_name]]
            label = torch.tensor(label.values, dtype = torch.float32)
            img = self.transforms(img)

            return img, label, image_name

    def __len__(self):
        return len(self.imgs)