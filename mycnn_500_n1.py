import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import random

import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import cv2
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ExampleNet import ExampleNet
from ExampleNet import CNN
from ExampleNet import CNN_500_part
torch.set_printoptions(profile='full')
# my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
# my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)
# tensor_x = torch.Tensor(my_x) # transform to torch tensor
# tensor_y = torch.Tensor(my_y)
bmname = 'food500_p0.5_m10_p1'
#_c0 and _c0test;;;_c0train and _c0ttest
def read_directory(directory_name):
    array_of_img = []  # this if for store all of the image data
    files = os.listdir(r"./"+directory_name)
    files.sort(key=lambda x: int(x[:-4]))
    #files.sort()
    for filename in files:
        img = cv2.imread(directory_name + "/" + filename,-1)
        array_of_img.append(img)
        #print(img)
    array_of_img=np.array(array_of_img)
    return array_of_img


def pre_data(bmname):
    prefix = os.path.join('cnn_data_dw_1', bmname,bmname)
    filename_classlabel = prefix + '_graph_labels_class.txt'
    graph_label = []
    label_vals = []
    with open(filename_classlabel, 'r') as f1:
        for line in f1:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_label.append(val)
    #label_map_to_int = {val: i for i, val in enumerate(label_vals)}  #将标签变为按顺序的从0-n
    #graph_label = np.array([label_map_to_int[l] for l in graph_label])
    tensor_label = torch.Tensor(graph_label).long()

    #adj_data = np.load(prefix+'_adj.npy')
    org_img_folder = prefix + '_adjdata'
    adj_data = read_directory(org_img_folder)
    tensor_data = torch.Tensor(adj_data).short()
    return tensor_data, tensor_label

#my_dataset = TensorDataset(tensor_data,tensor_label) # create your datset
#my_dataloader = DataLoader(my_dataset) # create your dataloader
#dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=4, shuffle=True)

# class0= {0: 0, 1: 1, 3: 2, 5: 3, 6: 4, 202: 5, 203: 6, 229: 7, 230: 8, 231: 9, 326: 10, 327: 11, 328: 12, 329: 13, 330: 14, 345: 15, 346: 16, 347: 17, 348: 18, 368: 19, 399: 20, 400: 21, 402: 22, 403: 23, 404: 24, 415: 25, 450: 26, 451: 27, 471: 28}
# class1= {2: 0, 4: 1, 10: 2, 11: 3, 12: 4, 24: 5, 25: 6, 26: 7, 27: 8, 28: 9, 43: 10, 54: 11, 65: 12, 66: 13, 92: 14, 93: 15, 94: 16, 95: 17, 96: 18, 98: 19, 105: 20, 172: 21, 174: 22, 179: 23, 180: 24, 181: 25, 182: 26, 183: 27, 184: 28, 186: 29, 188: 30, 205: 31, 207: 32, 208: 33, 239: 34, 240: 35, 243: 36, 244: 37, 245: 38, 246: 39, 249: 40, 257: 41, 260: 42, 261: 43, 262: 44, 274: 45, 284: 46, 286: 47, 287: 48, 289: 49, 290: 50, 291: 51, 295: 52, 298: 53, 324: 54, 361: 55, 363: 56, 382: 57, 390: 58, 393: 59, 398: 60, 409: 61, 410: 62, 411: 63, 413: 64, 423: 65, 424: 66, 425: 67, 426: 68, 428: 69, 429: 70, 430: 71, 435: 72, 455: 73, 472: 74, 474: 75, 478: 76, 479: 77}
# class2= {226: 0, 372: 1, 487: 2}
# class3= {13: 0, 14: 1, 15: 2, 16: 3, 18: 4, 19: 5, 20: 6, 21: 7, 22: 8, 23: 9, 85: 10, 191: 11, 192: 12, 193: 13, 194: 14, 383: 15, 394: 16, 449: 17}
# class4= {17: 0, 77: 1, 79: 2, 110: 3, 111: 4, 161: 5, 162: 6, 163: 7, 164: 8, 166: 9, 167: 10, 168: 11, 169: 12, 171: 13, 173: 14, 247: 15, 252: 16, 297: 17, 369: 18, 370: 19, 391: 20, 416: 21, 417: 22, 418: 23, 419: 24, 458: 25, 459: 26, 473: 27, 486: 28, 490: 29}
# class5= {187: 0, 320: 1, 321: 2, 414: 3, 491: 4}
# class6= {31: 0, 47: 1, 51: 2, 52: 3, 53: 4, 55: 5, 56: 6, 57: 7, 58: 8, 59: 9, 60: 10, 61: 11, 62: 12, 63: 13, 64: 14, 67: 15, 68: 16, 69: 17, 70: 18, 71: 19, 72: 20, 73: 21, 80: 22, 82: 23, 83: 24, 97: 25, 104: 26, 107: 27, 108: 28, 109: 29, 134: 30, 141: 31, 165: 32, 170: 33, 175: 34, 176: 35, 177: 36, 178: 37, 209: 38, 223: 39, 225: 40, 227: 41, 232: 42, 233: 43, 234: 44, 235: 45, 236: 46, 241: 47, 258: 48, 259: 49, 263: 50, 265: 51, 268: 52, 271: 53, 272: 54, 273: 55, 275: 56, 276: 57, 300: 58, 301: 59, 335: 60, 353: 61, 364: 62, 374: 63, 384: 64, 385: 65, 386: 66, 461: 67, 470: 68, 475: 69, 485: 70, 495: 71}
# class7= {211: 0, 212: 1, 213: 2, 214: 3, 215: 4, 444: 5}
# class8= {38: 0, 112: 1, 113: 2, 116: 3, 118: 4, 119: 5, 120: 6, 121: 7, 122: 8, 124: 9, 126: 10, 127: 11, 128: 12, 129: 13, 130: 14, 131: 15, 132: 16, 135: 17, 136: 18, 137: 19, 139: 20, 145: 21, 146: 22, 147: 23, 148: 24, 149: 25, 150: 26, 151: 27, 152: 28, 153: 29, 154: 30, 155: 31, 156: 32, 157: 33, 158: 34, 159: 35, 160: 36, 189: 37, 216: 38, 217: 39, 218: 40, 254: 41, 255: 42, 277: 43, 278: 44, 288: 45, 314: 46, 317: 47, 318: 48, 319: 49, 343: 50, 344: 51, 373: 52, 397: 53, 405: 54, 406: 55, 407: 56, 437: 57, 462: 58, 463: 59, 464: 60, 465: 61, 480: 62, 483: 63, 488: 64, 489: 65, 492: 66, 494: 67, 496: 68, 497: 69, 498: 70, 499: 71}
# class9= {39: 0, 40: 1, 41: 2, 42: 3, 44: 4, 45: 5, 114: 6, 115: 7, 219: 8, 220: 9, 221: 10, 222: 11, 322: 12}
# class10= {33: 0, 34: 1, 35: 2, 36: 3, 37: 4, 46: 5, 48: 6, 49: 7, 50: 8, 75: 9, 76: 10, 78: 11, 308: 12, 309: 13, 310: 14, 311: 15, 312: 16, 313: 17, 336: 18, 337: 19, 338: 20, 339: 21, 378: 22, 379: 23, 387: 24, 388: 25, 389: 26, 396: 27, 412: 28, 432: 29, 469: 30, 481: 31}
# class11= {7: 0, 8: 1, 9: 2, 29: 3, 30: 4, 32: 5, 81: 6, 84: 7, 86: 8, 87: 9, 88: 10, 89: 11, 90: 12, 91: 13, 106: 14, 117: 15, 123: 16, 125: 17, 133: 18, 138: 19, 140: 20, 142: 21, 143: 22, 144: 23, 185: 24, 190: 25, 195: 26, 196: 27, 197: 28, 198: 29, 199: 30, 200: 31, 201: 32, 206: 33, 210: 34, 224: 35, 228: 36, 237: 37, 238: 38, 242: 39, 250: 40, 251: 41, 253: 42, 266: 43, 279: 44, 280: 45, 285: 46, 296: 47, 307: 48, 316: 49, 323: 50, 331: 51, 332: 52, 333: 53, 334: 54, 340: 55, 341: 56, 342: 57, 351: 58, 352: 59, 354: 60, 355: 61, 356: 62, 357: 63, 358: 64, 359: 65, 360: 66, 362: 67, 365: 68, 366: 69, 375: 70, 381: 71, 392: 72, 401: 73, 408: 74, 422: 75, 427: 76, 431: 77, 433: 78, 434: 79, 436: 80, 438: 81, 439: 82, 440: 83, 441: 84, 442: 85, 443: 86, 445: 87, 446: 88, 447: 89, 448: 90, 456: 91, 457: 92, 460: 93, 466: 94, 467: 95, 468: 96, 477: 97, 482: 98, 484: 99}
# class12= {74: 0, 99: 1, 100: 2, 101: 3, 102: 4, 103: 5, 204: 6, 248: 7, 256: 8, 264: 9, 267: 10, 269: 11, 270: 12, 281: 13, 282: 14, 283: 15, 292: 16, 293: 17, 294: 18, 299: 19, 302: 20, 303: 21, 304: 22, 305: 23, 306: 24, 315: 25, 325: 26, 349: 27, 350: 28, 367: 29, 371: 30, 376: 31, 377: 32, 380: 33, 395: 34, 420: 35, 421: 36, 452: 37, 453: 38, 454: 39, 476: 40, 493: 41}
class0= {33: 0, 34: 1, 35: 2, 36: 3, 37: 4, 38: 5, 49: 6, 112: 7, 113: 8, 116: 9, 118: 10, 119: 11, 120: 12, 121: 13, 122: 14, 124: 15, 126: 16, 127: 17, 129: 18, 130: 19, 131: 20, 132: 21, 135: 22, 136: 23, 137: 24, 139: 25, 145: 26, 146: 27, 147: 28, 148: 29, 149: 30, 150: 31, 151: 32, 152: 33, 153: 34, 154: 35, 155: 36, 156: 37, 157: 38, 158: 39, 159: 40, 160: 41, 189: 42, 216: 43, 217: 44, 218: 45, 254: 46, 255: 47, 277: 48, 278: 49, 288: 50, 300: 51, 301: 52, 308: 53, 309: 54, 311: 55, 312: 56, 313: 57, 314: 58, 317: 59, 318: 60, 319: 61, 336: 62, 337: 63, 338: 64, 339: 65, 343: 66, 344: 67, 373: 68, 378: 69, 379: 70, 380: 71, 388: 72, 389: 73, 397: 74, 399: 75, 400: 76, 404: 77, 405: 78, 406: 79, 407: 80, 412: 81, 437: 82, 462: 83, 463: 84, 464: 85, 465: 86, 480: 87, 483: 88, 488: 89, 489: 90, 492: 91, 494: 92, 496: 93, 497: 94, 498: 95, 499: 96}
class1= {13: 0, 14: 1, 15: 2, 16: 3, 17: 4, 18: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 64: 11, 69: 12, 74: 13, 77: 14, 79: 15, 85: 16, 99: 17, 102: 18, 110: 19, 111: 20, 134: 21, 138: 22, 161: 23, 162: 24, 163: 25, 164: 26, 167: 27, 168: 28, 169: 29, 171: 30, 173: 31, 191: 32, 192: 33, 193: 34, 194: 35, 198: 36, 204: 37, 247: 38, 252: 39, 256: 40, 259: 41, 264: 42, 267: 43, 269: 44, 270: 45, 272: 46, 281: 47, 282: 48, 283: 49, 292: 50, 293: 51, 294: 52, 299: 53, 302: 54, 303: 55, 304: 56, 305: 57, 306: 58, 315: 59, 323: 60, 325: 61, 349: 62, 350: 63, 367: 64, 369: 65, 370: 66, 371: 67, 376: 68, 377: 69, 383: 70, 394: 71, 401: 72, 416: 73, 417: 74, 418: 75, 419: 76, 420: 77, 421: 78, 449: 79, 452: 80, 453: 81, 454: 82, 458: 83, 459: 84, 473: 85, 475: 86, 476: 87, 486: 88, 490: 89, 493: 90}
class2= {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 31: 6, 43: 7, 46: 8, 47: 9, 48: 10, 50: 11, 55: 12, 57: 13, 58: 14, 70: 15, 71: 16, 75: 17, 76: 18, 78: 19, 172: 20, 180: 21, 183: 22, 188: 23, 202: 24, 203: 25, 207: 26, 223: 27, 225: 28, 226: 29, 227: 30, 229: 31, 230: 32, 231: 33, 236: 34, 239: 35, 244: 36, 245: 37, 249: 38, 310: 39, 326: 40, 327: 41, 328: 42, 329: 43, 330: 44, 345: 45, 346: 46, 347: 47, 348: 48, 363: 49, 368: 50, 372: 51, 382: 52, 384: 53, 387: 54, 396: 55, 398: 56, 402: 57, 403: 58, 413: 59, 415: 60, 432: 61, 435: 62, 450: 63, 451: 64, 455: 65, 469: 66, 471: 67, 481: 68, 487: 69}
class3= {3: 0, 7: 1, 9: 2, 11: 3, 12: 4, 24: 5, 25: 6, 26: 7, 27: 8, 28: 9, 32: 10, 51: 11, 52: 12, 53: 13, 54: 14, 56: 15, 59: 16, 60: 17, 61: 18, 62: 19, 63: 20, 65: 21, 66: 22, 67: 23, 68: 24, 72: 25, 73: 26, 80: 27, 82: 28, 83: 29, 88: 30, 89: 31, 92: 32, 93: 33, 94: 34, 95: 35, 96: 36, 97: 37, 98: 38, 100: 39, 101: 40, 103: 41, 104: 42, 105: 43, 107: 44, 108: 45, 109: 46, 144: 47, 165: 48, 166: 49, 170: 50, 174: 51, 175: 52, 176: 53, 177: 54, 178: 55, 181: 56, 182: 57, 184: 58, 186: 59, 205: 60, 206: 61, 208: 62, 211: 63, 212: 64, 213: 65, 214: 66, 215: 67, 224: 68, 228: 69, 232: 70, 233: 71, 234: 72, 235: 73, 237: 74, 238: 75, 240: 76, 241: 77, 246: 78, 258: 79, 260: 80, 261: 81, 262: 82, 263: 83, 265: 84, 268: 85, 271: 86, 273: 87, 274: 88, 275: 89, 276: 90, 279: 91, 280: 92, 284: 93, 286: 94, 287: 95, 290: 96, 291: 97, 296: 98, 297: 99, 298: 100, 316: 101, 324: 102, 331: 103, 332: 104, 333: 105, 335: 106, 352: 107, 353: 108, 356: 109, 361: 110, 362: 111, 364: 112, 374: 113, 381: 114, 385: 115, 386: 116, 390: 117, 393: 118, 395: 119, 408: 120, 409: 121, 410: 122, 411: 123, 423: 124, 424: 125, 425: 126, 426: 127, 427: 128, 428: 129, 429: 130, 430: 131, 436: 132, 440: 133, 447: 134, 460: 135, 461: 136, 468: 137, 470: 138, 472: 139, 474: 140, 478: 141, 479: 142, 482: 143, 485: 144, 495: 145}
class4= {8: 0, 10: 1, 29: 2, 30: 3, 39: 4, 40: 5, 41: 6, 42: 7, 44: 8, 45: 9, 81: 10, 84: 11, 86: 12, 87: 13, 90: 14, 91: 15, 106: 16, 114: 17, 115: 18, 117: 19, 123: 20, 125: 21, 128: 22, 133: 23, 140: 24, 141: 25, 142: 26, 143: 27, 179: 28, 185: 29, 187: 30, 190: 31, 195: 32, 196: 33, 197: 34, 199: 35, 200: 36, 201: 37, 209: 38, 210: 39, 219: 40, 220: 41, 221: 42, 222: 43, 242: 44, 243: 45, 248: 46, 250: 47, 251: 48, 253: 49, 257: 50, 266: 51, 285: 52, 289: 53, 295: 54, 307: 55, 320: 56, 321: 57, 322: 58, 334: 59, 340: 60, 341: 61, 342: 62, 351: 63, 354: 64, 355: 65, 357: 66, 358: 67, 359: 68, 360: 69, 365: 70, 366: 71, 375: 72, 391: 73, 392: 74, 414: 75, 422: 76, 431: 77, 433: 78, 434: 79, 438: 80, 439: 81, 441: 82, 442: 83, 443: 84, 444: 85, 445: 86, 446: 87, 448: 88, 456: 89, 457: 90, 466: 91, 467: 92, 477: 93, 484: 94, 491: 95}
class0_reverse = {value: i for i, value in class0.items()}
class1_reverse = {value: i for i, value in class1.items()}
class2_reverse = {value: i for i, value in class2.items()}
class3_reverse = {value: i for i, value in class3.items()}
class4_reverse = {value: i for i, value in class4.items()}
# class5_reverse = {value: i for i, value in class5.items()}
# class6_reverse = {value: i for i, value in class6.items()}
# class7_reverse = {value: i for i, value in class7.items()}
# class8_reverse = {value: i for i, value in class8.items()}
# class9_reverse = {value: i for i, value in class9.items()}
# class10_reverse = {value: i for i, value in class10.items()}
# class11_reverse = {value: i for i, value in class11.items()}
# class12_reverse = {value: i for i, value in class12.items()}
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 10

    batch_size = 20

    #net = ExampleNet().to(device)
    net = CNN_500_part(5).to(device)
    criterion = nn.MSELoss(reduce = None, size_average = None)
    cross = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    train_data, train_label = pre_data(bmname+'_train')
    train_data = torch.unsqueeze(train_data, dim=1).type(torch.FloatTensor)
    train_dataset = TensorDataset(train_data, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data, test_label = pre_data(bmname+'_test')
    #test_data, test_label = pre_data('BA300_SI_m5_test')
    test_data = torch.unsqueeze(test_data, dim=1).type(torch.FloatTensor)
    test_dataset = TensorDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for i in range(epochs):

        for j, (input, target) in enumerate(train_dataloader):
            input = input.to(device)
            output = net(input)
            target = torch.zeros(target.size(0), 5).scatter_(1, target.view(-1, 1), 1).to(device)
            _, indices = torch.max(target, 1)
            #loss = criterion(output, target)
            loss = cross(output,indices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                losses.append(loss.float().detach().numpy())
                print("[epochs - {0} - {1}/{2}]loss: {3}".format(i, j, len(train_dataloader), loss.float()))
                #plt.clf()
                #plt.plot(losses)
                #plt.savefig(bmname+'0_loss.jpg')
                #plt.pause(0.01)
        with torch.no_grad():
            net.eval()
            correct = 0.
            total = 0.
            labels= []
            preds=[]
            for input, target in test_dataloader:
                input, target = input.to(device), target.to(device)
                output = net(input)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
                accuracy = correct.float() / total
                labels.append(target.long().numpy())
                preds.append(predicted.long().numpy())
            net.train()
            print("[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))

            # labels = np.hstack(labels)
            # preds = np.hstack(preds)
            # real_labels = []
            # for n in labels:
            #     real_labels.append(class2_reverse[n])
            # real_preds = []
            # for m in preds:
            #     real_preds.append(class2_reverse[m])
            # read_dic = np.load("food500_short_path.npy", allow_pickle=True).item()
            # distance_pred = []
            # count = 0
            # for w in range(len(labels)):
            #     a = read_dic[real_labels[w]][real_preds[w]]
            #     distance_pred.append(a)
            #     if labels[w] == preds[w]:
            #         count = count + 1
            # print('perdict accuracy:', count / len(labels))
            # result_dis = {}
            # ave_class = 0
            # for q in set(distance_pred):
            #     result_dis[q] = distance_pred.count(q)
            #     ave_class = q * result_dis[q] + ave_class
            # print('GNN distance:', ave_class / len(labels))
        torch.save(net.state_dict(), "models/"+bmname+"_p1.pth")
    para = sum([np.prod(list(p.size())) for p in net.parameters()])
    print(para)
    print('Model {} : params: {:4f}M'.format(net._get_name(), para * 4 / 1000 / 1000))
if __name__ == "__main__":
    main()


