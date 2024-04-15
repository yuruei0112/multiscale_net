from sklearn.metrics import roc_auc_score
import pandas as pd
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
import argparse

from .utils import get_train_transform, get_val_transform,imglist_generation
from .ich_dataset import ICHDataset 
from .model import Multiscale_Net

def parse_args():
    parser = argparse.ArgumentParser(description='Multiscale_net for ICH classification')

    # basic settings
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val-batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_scheduler', type=bool, default=False)
    parser.add_argument('--lr_min', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--early-stop-limit', type=int, default=10)
    parser.add_argument('--opt', default= torch.optim.Adam)
    parser.add_argument('--img-dir', type=str)

    parser.add_argument('--csv-path', type=str, default='./train_df.csv') # labels of imgs
    
    parser.add_argument('--train-patient-csv', type=str, default='./train_patient_names.csv') # patient IDs for training
    parser.add_argument('--patient-correspodence-csv', type=str, default='./stage_2_sort_by_PatientID_and_zposition.csv')                             

    parser.add_argument('--img-dir', default='./ich_dir', type=str, help='ich_dir')
    parser.add_argument('--in-ch', type=int, default=3)
    parser.add_argument('--nb-classes', type=int, default=6)
    parser.add_argument('--save-model-path', type=str, default='./checkpoints') 

    args = parser.parse_args()
    return args

def val_one_epoch(val_data_loader ,best_val_loss, chance, score_with_best_loss, model, device):
    global_trues = []
    global_pred = []
    global_file_name = []
    epoch_loss = []
    start_time = time.time()
    all_trues = []
    all_y_preds = []
    epoch_score = []

    for images, labels, image_name in val_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 6))
        global_file_name.extend(image_name)

        preds = model(images)

        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)


        preds = preds.cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        global_trues.extend(labels)
        global_pred.extend(preds)

    end_time = time.time()
    total_time = end_time - start_time

    epoch_loss = np.mean(epoch_loss)
    total_acc = np.mean(total_acc)
    any_acc = np.mean(any_acc)
    epidural_acc = np.mean(epidural_acc)
    subdural_acc = np.mean(subdural_acc)
    subarachnoid_acc = np.mean(subarachnoid_acc)
    intraparanchymal_acc = np.mean(intraparanchymal_acc)
    intraventricular_acc = np.mean(intraventricular_acc)
    for i in range(6):
        epoch_score.append(roc_auc_score(np.array(all_trues)[:, i], np.array(all_y_preds)[:, i]))

    val_logs["loss"].append(epoch_loss)
    val_logs["time"].append(total_time)
    val_logs["ROC_AUC SCORE"].append(epoch_score)

    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(args.checkpoints, "dw_multicscale_3block_"+'.pth'))
        df_1 = pd.DataFrame(global_file_name)
        df_2 = pd.DataFrame(global_trues)
        df_3 = pd.DataFrame(global_pred)
        df_all = pd.concat([df_1, df_2, df_3], axis=1)
        df_all.to_csv(os.path.join(args.checkpoints,'.csv'))
        chance = 0
        score_with_best_loss = epoch_score
    elif epoch_loss > best_val_loss:
        chance += 1
    return epoch_loss, total_time, best_val_loss, chance, epoch_score, score_with_best_loss, global_file_name, global_trues, global_pred


def train_one_epoch(train_data_loader,model, optimizer,device):
    epoch_loss = []

    start_time = time.time()
    all_trues = []
    all_y_preds = []
    epoch_score = []

    for images, labels in train_data_loader:
        images = images.to(device) #??
        labels = labels.to(device)
        labels = labels.reshape((labels.shape[0], 6))

        optimizer.zero_grad()
        preds =model(images)

        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)

        all_trues.extend(trues)
        all_y_preds.extend(y_pred)

        _loss.backward()
        optimizer.step()

    end_time = time.time()
    total_time = end_time - start_time

    epoch_loss = np.mean(epoch_loss)
    for i in range(6):
        epoch_score.append(roc_auc_score(np.array(all_trues)[:, i], np.array(all_y_preds)[:, i]))


    train_logs["loss"].append(epoch_loss)
    train_logs["time"].append(total_time)
    train_logs["ROC_AUC SCORE"].append(epoch_score)
    return epoch_loss,  total_time, epoch_score

train_logs = {"loss" : [], "time" : [], "ROC_AUC SCORE" : []}
val_logs = {"loss" : [], "time" : [], "ROC_AUC SCORE" : []}


def main(args):
    
    train_imgs, val_imgs = imglist_generation(args.train_patient_csv, args.paitent_correspondence_csv ,0.9)

    train_dataset = ICHDataset(train_imgs,ars.img_dir, args.csv_path, args.img_size ,mode="train", transforms=get_train_transform())
    val_dataset = ICHDataset(val_imgs, ars.img_dir, args.csv_path,args.img_size,mode="val", transforms=get_val_transform())    

    train_data_loader = DataLoader(
        dataset= train_dataset,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle= True,
        drop_last= True
    )
    val_data_loader= DataLoader(
        dataset=val_dataset,
        num_workers=4,
        batch_size=args.val_batch_size,
        shuffle=True
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = Multiscale_Net(in_ch = args.in_ch, out_ch = args.out_ch).to(device)
    summary(model)

    optimizer = args.opt(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.BCELoss()


    limit=args.early_stop_limit
    score_with_best_loss = []
    chance=0
    best_val_loss = 1
    k=epochs+1


    for epoch in range(epochs):
    
        model.train()
        loss, _time , score= train_one_epoch(train_data_loader,model,optimizer,device)
        if (epoch+1)%5 == 0:
            print("\nTraining")
            print("Epoch{}".format(epoch+1))
            print("Loss:{}".format(round(loss, 4)))
            print("ROC_AUC SCORE :{}".format(np.around(np.array(score), 4)))
            print("Time :{}".format(round(_time, 4)))
        elif chance == limit:
            print("\nTraining")
            print("Epoch{}".format(epoch + 1))
            print("Loss:{}".format(round(loss, 4)))
            print("ROC_AUC SCORE :{}".format(np.around(np.array(score), 4)))
            print("Time :{}".format(round(_time, 4)))
        #validation
        model.eval()
        with torch.no_grad():
            loss, _time, best_val_loss, chance, score, score_with_best_loss, global_file_name, global_trues, global_pred= val_one_epoch(val_data_loader, best_val_loss, chance, score_with_best_loss,model,device)
            if (epoch + 1) % 5 == 0:
                print("\nValidation")
                print("Epoch{}".format(epoch + 1))
                print("Loss:{}".format(round(loss, 4)))
                print("ROC_AUC SCORE :{}".format(np.around(np.array(score), 4)))
                print("Time :{}".format(round(_time, 4)))
            elif chance == limit:
                print("\nValidation")
                print("Epoch{}".format(epoch + 1))
                print("Loss:{}".format(round(loss, 4)))
                print("ROC_AUC SCORE :{}".format(np.around(np.array(score), 4)))
                print("Time :{}".format(round(_time, 4)))
            
        if chance == limit:
            k = epoch+2
            break

    


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
