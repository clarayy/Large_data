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
from ExampleNet import CNN_500
from ExampleNet import CNN_500_part
torch.set_printoptions(profile='full')
# my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
# my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)
# tensor_x = torch.Tensor(my_x) # transform to torch tensor
# tensor_y = torch.Tensor(my_y)
bmname = 'food500_p0.5_m10_p1'
#bmname1 = 'food500_SI_A1_m5'
#_c0 and _c0test;;;_c0train and _c0ttest
def read_directory(directory_name):
    array_of_img = []  # this if for store all of the image data
    files = os.listdir(r"./"+directory_name)
    files.sort(key=lambda x:int(x[:-4]))                  ###########一切都是有原因的！！！顺序错了导致对应labels错了导致结果不好
    #files.sort()                                                 # 单个100-199是对的，但和0-99组合在一起就不对了
    for filename in files:
        img = cv2.imread(directory_name + "/" + filename,-1)
        array_of_img.append(img)
        #print(img)
    array_of_img=np.array(array_of_img)
    return array_of_img


def pre_data(bmname):
    prefix = os.path.join('cnn_data_dw_1', bmname,bmname)
    #图像的第一级分类标签
    filename_classlabel = prefix + '_graph_labels_class.txt'
    graph_label_class = []
    label_vals = []
    with open(filename_classlabel, 'r') as f1:
        for line in f1:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_label_class.append(val)
    # label_map_to_int = {val: i for i, val in enumerate(label_vals)}  #将标签变为按顺序的从0-n
    # graph_label = np.array([label_map_to_int[l] for l in graph_label])
    tensor_label_class = torch.Tensor(graph_label_class).long()
    #图像的adjdata
    #adj_data = np.load(prefix+'_adj.npy')
    org_img_folder = prefix + '_adjdata'
    adj_data = read_directory(org_img_folder)
    tensor_data = torch.Tensor(adj_data).short()
    # #图像的jc标签
    # filename_jclabel = prefix + '_jordancenter.txt'
    # jc = []
    # with open(filename_jclabel, 'r') as f2:
    #     for line in f2:
    #         line = line.strip("\n")
    #         val = int(line)
    #         jc.append(val)
    # label_map_to_int = {val: i for i, val in enumerate(label_vals)}  #将标签变为按顺序的从0-n
    # graph_label = np.array([label_map_to_int[l] for l in graph_label])
    #tensor_jc = torch.Tensor(jc).long()
    #图像的真实标签
    filename_label = prefix + '_graph_labels.txt'
    graph_label = []
    label_vals = []
    with open(filename_label, 'r') as f3:
        for line in f3:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_label.append(val)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}  #将标签变为按顺序的从0-n
    #graph_label = np.array([label_map_to_int[l] for l in graph_label])       #单个类300-399时得到对应0-99
    graph_label = np.array([class_zong[l] for l in graph_label])              #5个类组合在一起时分别对应到0-99
    tensor_label = torch.Tensor(graph_label).long()
    return tensor_data, tensor_label_class, tensor_label

# class0 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60, 61: 61, 62: 62, 63: 63, 64: 64, 65: 65, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70, 71: 71, 72: 72, 73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80, 81: 81, 82: 82, 83: 83, 84: 84, 85: 85, 86: 86, 87: 87, 88: 88, 89: 89, 90: 90, 91: 91, 92: 92, 93: 93, 94: 94, 95: 95, 96: 96, 97: 97, 98: 98, 99: 99}
# class1 = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 105: 5, 106: 6, 107: 7, 108: 8, 109: 9, 110: 10, 111: 11, 112: 12, 113: 13, 114: 14, 115: 15, 116: 16, 117: 17, 118: 18, 119: 19, 120: 20, 121: 21, 122: 22, 123: 23, 124: 24, 125: 25, 126: 26, 127: 27, 128: 28, 129: 29, 130: 30, 131: 31, 132: 32, 133: 33, 134: 34, 135: 35, 136: 36, 137: 37, 138: 38, 139: 39, 140: 40, 141: 41, 142: 42, 143: 43, 144: 44, 145: 45, 146: 46, 147: 47, 148: 48, 149: 49, 150: 50, 151: 51, 152: 52, 153: 53, 154: 54, 155: 55, 156: 56, 157: 57, 158: 58, 159: 59, 160: 60, 161: 61, 162: 62, 163: 63, 164: 64, 165: 65, 166: 66, 167: 67, 168: 68, 169: 69, 170: 70, 171: 71, 172: 72, 173: 73, 174: 74, 175: 75, 176: 76, 177: 77, 178: 78, 179: 79, 180: 80, 181: 81, 182: 82, 183: 83, 184: 84, 185: 85, 186: 86, 187: 87, 188: 88, 189: 89, 190: 90, 191: 91, 192: 92, 193: 93, 194: 94, 195: 95, 196: 96, 197: 97, 198: 98, 199: 99}
# class2 = {200: 0, 201: 1, 202: 2, 203: 3, 204: 4, 205: 5, 206: 6, 207: 7, 208: 8, 209: 9, 210: 10, 211: 11, 212: 12, 213: 13, 214: 14, 215: 15, 216: 16, 217: 17, 218: 18, 219: 19, 220: 20, 221: 21, 222: 22, 223: 23, 224: 24, 225: 25, 226: 26, 227: 27, 228: 28, 229: 29, 230: 30, 231: 31, 232: 32, 233: 33, 234: 34, 235: 35, 236: 36, 237: 37, 238: 38, 239: 39, 240: 40, 241: 41, 242: 42, 243: 43, 244: 44, 245: 45, 246: 46, 247: 47, 248: 48, 249: 49, 250: 50, 251: 51, 252: 52, 253: 53, 254: 54, 255: 55, 256: 56, 257: 57, 258: 58, 259: 59, 260: 60, 261: 61, 262: 62, 263: 63, 264: 64, 265: 65, 266: 66, 267: 67, 268: 68, 269: 69, 270: 70, 271: 71, 272: 72, 273: 73, 274: 74, 275: 75, 276: 76, 277: 77, 278: 78, 279: 79, 280: 80, 281: 81, 282: 82, 283: 83, 284: 84, 285: 85, 286: 86, 287: 87, 288: 88, 289: 89, 290: 90, 291: 91, 292: 92, 293: 93, 294: 94, 295: 95, 296: 96, 297: 97, 298: 98, 299: 99}
# class3 = {300: 0, 301: 1, 302: 2, 303: 3, 304: 4, 305: 5, 306: 6, 307: 7, 308: 8, 309: 9, 310: 10, 311: 11, 312: 12, 313: 13, 314: 14, 315: 15, 316: 16, 317: 17, 318: 18, 319: 19, 320: 20, 321: 21, 322: 22, 323: 23, 324: 24, 325: 25, 326: 26, 327: 27, 328: 28, 329: 29, 330: 30, 331: 31, 332: 32, 333: 33, 334: 34, 335: 35, 336: 36, 337: 37, 338: 38, 339: 39, 340: 40, 341: 41, 342: 42, 343: 43, 344: 44, 345: 45, 346: 46, 347: 47, 348: 48, 349: 49, 350: 50, 351: 51, 352: 52, 353: 53, 354: 54, 355: 55, 356: 56, 357: 57, 358: 58, 359: 59, 360: 60, 361: 61, 362: 62, 363: 63, 364: 64, 365: 65, 366: 66, 367: 67, 368: 68, 369: 69, 370: 70, 371: 71, 372: 72, 373: 73, 374: 74, 375: 75, 376: 76, 377: 77, 378: 78, 379: 79, 380: 80, 381: 81, 382: 82, 383: 83, 384: 84, 385: 85, 386: 86, 387: 87, 388: 88, 389: 89, 390: 90, 391: 91, 392: 92, 393: 93, 394: 94, 395: 95, 396: 96, 397: 97, 398: 98, 399: 99}
# class4 = {400: 0, 401: 1, 402: 2, 403: 3, 404: 4, 405: 5, 406: 6, 407: 7, 408: 8, 409: 9, 410: 10, 411: 11, 412: 12, 413: 13, 414: 14, 415: 15, 416: 16, 417: 17, 418: 18, 419: 19, 420: 20, 421: 21, 422: 22, 423: 23, 424: 24, 425: 25, 426: 26, 427: 27, 428: 28, 429: 29, 430: 30, 431: 31, 432: 32, 433: 33, 434: 34, 435: 35, 436: 36, 437: 37, 438: 38, 439: 39, 440: 40, 441: 41, 442: 42, 443: 43, 444: 44, 445: 45, 446: 46, 447: 47, 448: 48, 449: 49, 450: 50, 451: 51, 452: 52, 453: 53, 454: 54, 455: 55, 456: 56, 457: 57, 458: 58, 459: 59, 460: 60, 461: 61, 462: 62, 463: 63, 464: 64, 465: 65, 466: 66, 467: 67, 468: 68, 469: 69, 470: 70, 471: 71, 472: 72, 473: 73, 474: 74, 475: 75, 476: 76, 477: 77, 478: 78, 479: 79, 480: 80, 481: 81, 482: 82, 483: 83, 484: 84, 485: 85, 486: 86, 487: 87, 488: 88, 489: 89, 490: 90, 491: 91, 492: 92, 493: 93, 494: 94, 495: 95, 496: 96, 497: 97, 498: 98, 499: 99}
# class_zong = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50, 51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60, 61: 61, 62: 62, 63: 63, 64: 64, 65: 65, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70, 71: 71, 72: 72, 73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80, 81: 81, 82: 82, 83: 83, 84: 84, 85: 85, 86: 86, 87: 87, 88: 88, 89: 89, 90: 90, 91: 91, 92: 92, 93: 93, 94: 94, 95: 95, 96: 96, 97: 97, 98: 98, 99: 99, 100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 105: 5, 106: 6, 107: 7, 108: 8, 109: 9, 110: 10, 111: 11, 112: 12, 113: 13, 114: 14, 115: 15, 116: 16, 117: 17, 118: 18, 119: 19, 120: 20, 121: 21, 122: 22, 123: 23, 124: 24, 125: 25, 126: 26, 127: 27, 128: 28, 129: 29, 130: 30, 131: 31, 132: 32, 133: 33, 134: 34, 135: 35, 136: 36, 137: 37, 138: 38, 139: 39, 140: 40, 141: 41, 142: 42, 143: 43, 144: 44, 145: 45, 146: 46, 147: 47, 148: 48, 149: 49, 150: 50, 151: 51, 152: 52, 153: 53, 154: 54, 155: 55, 156: 56, 157: 57, 158: 58, 159: 59, 160: 60, 161: 61, 162: 62, 163: 63, 164: 64, 165: 65, 166: 66, 167: 67, 168: 68, 169: 69, 170: 70, 171: 71, 172: 72, 173: 73, 174: 74, 175: 75, 176: 76, 177: 77, 178: 78, 179: 79, 180: 80, 181: 81, 182: 82, 183: 83, 184: 84, 185: 85, 186: 86, 187: 87, 188: 88, 189: 89, 190: 90, 191: 91, 192: 92, 193: 93, 194: 94, 195: 95, 196: 96, 197: 97, 198: 98, 199: 99, 200: 0, 201: 1, 202: 2, 203: 3, 204: 4, 205: 5, 206: 6, 207: 7, 208: 8, 209: 9, 210: 10, 211: 11, 212: 12, 213: 13, 214: 14, 215: 15, 216: 16, 217: 17, 218: 18, 219: 19, 220: 20, 221: 21, 222: 22, 223: 23, 224: 24, 225: 25, 226: 26, 227: 27, 228: 28, 229: 29, 230: 30, 231: 31, 232: 32, 233: 33, 234: 34, 235: 35, 236: 36, 237: 37, 238: 38, 239: 39, 240: 40, 241: 41, 242: 42, 243: 43, 244: 44, 245: 45, 246: 46, 247: 47, 248: 48, 249: 49, 250: 50, 251: 51, 252: 52, 253: 53, 254: 54, 255: 55, 256: 56, 257: 57, 258: 58, 259: 59, 260: 60, 261: 61, 262: 62, 263: 63, 264: 64, 265: 65, 266: 66, 267: 67, 268: 68, 269: 69, 270: 70, 271: 71, 272: 72, 273: 73, 274: 74, 275: 75, 276: 76, 277: 77, 278: 78, 279: 79, 280: 80, 281: 81, 282: 82, 283: 83, 284: 84, 285: 85, 286: 86, 287: 87, 288: 88, 289: 89, 290: 90, 291: 91, 292: 92, 293: 93, 294: 94, 295: 95, 296: 96, 297: 97, 298: 98, 299: 99, 300: 0, 301: 1, 302: 2, 303: 3, 304: 4, 305: 5, 306: 6, 307: 7, 308: 8, 309: 9, 310: 10, 311: 11, 312: 12, 313: 13, 314: 14, 315: 15, 316: 16, 317: 17, 318: 18, 319: 19, 320: 20, 321: 21, 322: 22, 323: 23, 324: 24, 325: 25, 326: 26, 327: 27, 328: 28, 329: 29, 330: 30, 331: 31, 332: 32, 333: 33, 334: 34, 335: 35, 336: 36, 337: 37, 338: 38, 339: 39, 340: 40, 341: 41, 342: 42, 343: 43, 344: 44, 345: 45, 346: 46, 347: 47, 348: 48, 349: 49, 350: 50, 351: 51, 352: 52, 353: 53, 354: 54, 355: 55, 356: 56, 357: 57, 358: 58, 359: 59, 360: 60, 361: 61, 362: 62, 363: 63, 364: 64, 365: 65, 366: 66, 367: 67, 368: 68, 369: 69, 370: 70, 371: 71, 372: 72, 373: 73, 374: 74, 375: 75, 376: 76, 377: 77, 378: 78, 379: 79, 380: 80, 381: 81, 382: 82, 383: 83, 384: 84, 385: 85, 386: 86, 387: 87, 388: 88, 389: 89, 390: 90, 391: 91, 392: 92, 393: 93, 394: 94, 395: 95, 396: 96, 397: 97, 398: 98, 399: 99, 400: 0, 401: 1, 402: 2, 403: 3, 404: 4, 405: 5, 406: 6, 407: 7, 408: 8, 409: 9, 410: 10, 411: 11, 412: 12, 413: 13, 414: 14, 415: 15, 416: 16, 417: 17, 418: 18, 419: 19, 420: 20, 421: 21, 422: 22, 423: 23, 424: 24, 425: 25, 426: 26, 427: 27, 428: 28, 429: 29, 430: 30, 431: 31, 432: 32, 433: 33, 434: 34, 435: 35, 436: 36, 437: 37, 438: 38, 439: 39, 440: 40, 441: 41, 442: 42, 443: 43, 444: 44, 445: 45, 446: 46, 447: 47, 448: 48, 449: 49, 450: 50, 451: 51, 452: 52, 453: 53, 454: 54, 455: 55, 456: 56, 457: 57, 458: 58, 459: 59, 460: 60, 461: 61, 462: 62, 463: 63, 464: 64, 465: 65, 466: 66, 467: 67, 468: 68, 469: 69, 470: 70, 471: 71, 472: 72, 473: 73, 474: 74, 475: 75, 476: 76, 477: 77, 478: 78, 479: 79, 480: 80, 481: 81, 482: 82, 483: 83, 484: 84, 485: 85, 486: 86, 487: 87, 488: 88, 489: 89, 490: 90, 491: 91, 492: 92, 493: 93, 494: 94, 495: 95, 496: 96, 497: 97, 498: 98, 499: 99}
class0= {33: 0, 34: 1, 35: 2, 36: 3, 37: 4, 38: 5, 49: 6, 112: 7, 113: 8, 116: 9, 118: 10, 119: 11, 120: 12, 121: 13, 122: 14, 124: 15, 126: 16, 127: 17, 129: 18, 130: 19, 131: 20, 132: 21, 135: 22, 136: 23, 137: 24, 139: 25, 145: 26, 146: 27, 147: 28, 148: 29, 149: 30, 150: 31, 151: 32, 152: 33, 153: 34, 154: 35, 155: 36, 156: 37, 157: 38, 158: 39, 159: 40, 160: 41, 189: 42, 216: 43, 217: 44, 218: 45, 254: 46, 255: 47, 277: 48, 278: 49, 288: 50, 300: 51, 301: 52, 308: 53, 309: 54, 311: 55, 312: 56, 313: 57, 314: 58, 317: 59, 318: 60, 319: 61, 336: 62, 337: 63, 338: 64, 339: 65, 343: 66, 344: 67, 373: 68, 378: 69, 379: 70, 380: 71, 388: 72, 389: 73, 397: 74, 399: 75, 400: 76, 404: 77, 405: 78, 406: 79, 407: 80, 412: 81, 437: 82, 462: 83, 463: 84, 464: 85, 465: 86, 480: 87, 483: 88, 488: 89, 489: 90, 492: 91, 494: 92, 496: 93, 497: 94, 498: 95, 499: 96}
class1= {13: 0, 14: 1, 15: 2, 16: 3, 17: 4, 18: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 64: 11, 69: 12, 74: 13, 77: 14, 79: 15, 85: 16, 99: 17, 102: 18, 110: 19, 111: 20, 134: 21, 138: 22, 161: 23, 162: 24, 163: 25, 164: 26, 167: 27, 168: 28, 169: 29, 171: 30, 173: 31, 191: 32, 192: 33, 193: 34, 194: 35, 198: 36, 204: 37, 247: 38, 252: 39, 256: 40, 259: 41, 264: 42, 267: 43, 269: 44, 270: 45, 272: 46, 281: 47, 282: 48, 283: 49, 292: 50, 293: 51, 294: 52, 299: 53, 302: 54, 303: 55, 304: 56, 305: 57, 306: 58, 315: 59, 323: 60, 325: 61, 349: 62, 350: 63, 367: 64, 369: 65, 370: 66, 371: 67, 376: 68, 377: 69, 383: 70, 394: 71, 401: 72, 416: 73, 417: 74, 418: 75, 419: 76, 420: 77, 421: 78, 449: 79, 452: 80, 453: 81, 454: 82, 458: 83, 459: 84, 473: 85, 475: 86, 476: 87, 486: 88, 490: 89, 493: 90}
class2= {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 31: 6, 43: 7, 46: 8, 47: 9, 48: 10, 50: 11, 55: 12, 57: 13, 58: 14, 70: 15, 71: 16, 75: 17, 76: 18, 78: 19, 172: 20, 180: 21, 183: 22, 188: 23, 202: 24, 203: 25, 207: 26, 223: 27, 225: 28, 226: 29, 227: 30, 229: 31, 230: 32, 231: 33, 236: 34, 239: 35, 244: 36, 245: 37, 249: 38, 310: 39, 326: 40, 327: 41, 328: 42, 329: 43, 330: 44, 345: 45, 346: 46, 347: 47, 348: 48, 363: 49, 368: 50, 372: 51, 382: 52, 384: 53, 387: 54, 396: 55, 398: 56, 402: 57, 403: 58, 413: 59, 415: 60, 432: 61, 435: 62, 450: 63, 451: 64, 455: 65, 469: 66, 471: 67, 481: 68, 487: 69}
class3= {3: 0, 7: 1, 9: 2, 11: 3, 12: 4, 24: 5, 25: 6, 26: 7, 27: 8, 28: 9, 32: 10, 51: 11, 52: 12, 53: 13, 54: 14, 56: 15, 59: 16, 60: 17, 61: 18, 62: 19, 63: 20, 65: 21, 66: 22, 67: 23, 68: 24, 72: 25, 73: 26, 80: 27, 82: 28, 83: 29, 88: 30, 89: 31, 92: 32, 93: 33, 94: 34, 95: 35, 96: 36, 97: 37, 98: 38, 100: 39, 101: 40, 103: 41, 104: 42, 105: 43, 107: 44, 108: 45, 109: 46, 144: 47, 165: 48, 166: 49, 170: 50, 174: 51, 175: 52, 176: 53, 177: 54, 178: 55, 181: 56, 182: 57, 184: 58, 186: 59, 205: 60, 206: 61, 208: 62, 211: 63, 212: 64, 213: 65, 214: 66, 215: 67, 224: 68, 228: 69, 232: 70, 233: 71, 234: 72, 235: 73, 237: 74, 238: 75, 240: 76, 241: 77, 246: 78, 258: 79, 260: 80, 261: 81, 262: 82, 263: 83, 265: 84, 268: 85, 271: 86, 273: 87, 274: 88, 275: 89, 276: 90, 279: 91, 280: 92, 284: 93, 286: 94, 287: 95, 290: 96, 291: 97, 296: 98, 297: 99, 298: 100, 316: 101, 324: 102, 331: 103, 332: 104, 333: 105, 335: 106, 352: 107, 353: 108, 356: 109, 361: 110, 362: 111, 364: 112, 374: 113, 381: 114, 385: 115, 386: 116, 390: 117, 393: 118, 395: 119, 408: 120, 409: 121, 410: 122, 411: 123, 423: 124, 424: 125, 425: 126, 426: 127, 427: 128, 428: 129, 429: 130, 430: 131, 436: 132, 440: 133, 447: 134, 460: 135, 461: 136, 468: 137, 470: 138, 472: 139, 474: 140, 478: 141, 479: 142, 482: 143, 485: 144, 495: 145}
class4= {8: 0, 10: 1, 29: 2, 30: 3, 39: 4, 40: 5, 41: 6, 42: 7, 44: 8, 45: 9, 81: 10, 84: 11, 86: 12, 87: 13, 90: 14, 91: 15, 106: 16, 114: 17, 115: 18, 117: 19, 123: 20, 125: 21, 128: 22, 133: 23, 140: 24, 141: 25, 142: 26, 143: 27, 179: 28, 185: 29, 187: 30, 190: 31, 195: 32, 196: 33, 197: 34, 199: 35, 200: 36, 201: 37, 209: 38, 210: 39, 219: 40, 220: 41, 221: 42, 222: 43, 242: 44, 243: 45, 248: 46, 250: 47, 251: 48, 253: 49, 257: 50, 266: 51, 285: 52, 289: 53, 295: 54, 307: 55, 320: 56, 321: 57, 322: 58, 334: 59, 340: 60, 341: 61, 342: 62, 351: 63, 354: 64, 355: 65, 357: 66, 358: 67, 359: 68, 360: 69, 365: 70, 366: 71, 375: 72, 391: 73, 392: 74, 414: 75, 422: 76, 431: 77, 433: 78, 434: 79, 438: 80, 439: 81, 441: 82, 442: 83, 443: 84, 444: 85, 445: 86, 446: 87, 448: 88, 456: 89, 457: 90, 466: 91, 467: 92, 477: 93, 484: 94, 491: 95}
class_zong={33: 0, 34: 1, 35: 2, 36: 3, 37: 4, 38: 5, 49: 6, 112: 7, 113: 8, 116: 9, 118: 10, 119: 11, 120: 12, 121: 13, 122: 14, 124: 15, 126: 16, 127: 17, 129: 18, 130: 19, 131: 20, 132: 21, 135: 22, 136: 23, 137: 24, 139: 25, 145: 26, 146: 27, 147: 28, 148: 29, 149: 30, 150: 31, 151: 32, 152: 33, 153: 34, 154: 35, 155: 36, 156: 37, 157: 38, 158: 39, 159: 40, 160: 41, 189: 42, 216: 43, 217: 44, 218: 45, 254: 46, 255: 47, 277: 48, 278: 49, 288: 50, 300: 51, 301: 52, 308: 53, 309: 54, 311: 55, 312: 56, 313: 57, 314: 58, 317: 59, 318: 60, 319: 61, 336: 62, 337: 63, 338: 64, 339: 65, 343: 66, 344: 67, 373: 68, 378: 69, 379: 70, 380: 71, 388: 72, 389: 73, 397: 74, 399: 75, 400: 76, 404: 77, 405: 78, 406: 79, 407: 80, 412: 81, 437: 82, 462: 83, 463: 84, 464: 85, 465: 86, 480: 87, 483: 88, 488: 89, 489: 90, 492: 91, 494: 92, 496: 93, 497: 94, 498: 95, 499: 96,13: 0, 14: 1, 15: 2, 16: 3, 17: 4, 18: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 64: 11, 69: 12, 74: 13, 77: 14, 79: 15, 85: 16, 99: 17, 102: 18, 110: 19, 111: 20, 134: 21, 138: 22, 161: 23, 162: 24, 163: 25, 164: 26, 167: 27, 168: 28, 169: 29, 171: 30, 173: 31, 191: 32, 192: 33, 193: 34, 194: 35, 198: 36, 204: 37, 247: 38, 252: 39, 256: 40, 259: 41, 264: 42, 267: 43, 269: 44, 270: 45, 272: 46, 281: 47, 282: 48, 283: 49, 292: 50, 293: 51, 294: 52, 299: 53, 302: 54, 303: 55, 304: 56, 305: 57, 306: 58, 315: 59, 323: 60, 325: 61, 349: 62, 350: 63, 367: 64, 369: 65, 370: 66, 371: 67, 376: 68, 377: 69, 383: 70, 394: 71, 401: 72, 416: 73, 417: 74, 418: 75, 419: 76, 420: 77, 421: 78, 449: 79, 452: 80, 453: 81, 454: 82, 458: 83, 459: 84, 473: 85, 475: 86, 476: 87, 486: 88, 490: 89, 493: 90,0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 31: 6, 43: 7, 46: 8, 47: 9, 48: 10, 50: 11, 55: 12, 57: 13, 58: 14, 70: 15, 71: 16, 75: 17, 76: 18, 78: 19, 172: 20, 180: 21, 183: 22, 188: 23, 202: 24, 203: 25, 207: 26, 223: 27, 225: 28, 226: 29, 227: 30, 229: 31, 230: 32, 231: 33, 236: 34, 239: 35, 244: 36, 245: 37, 249: 38, 310: 39, 326: 40, 327: 41, 328: 42, 329: 43, 330: 44, 345: 45, 346: 46, 347: 47, 348: 48, 363: 49, 368: 50, 372: 51, 382: 52, 384: 53, 387: 54, 396: 55, 398: 56, 402: 57, 403: 58, 413: 59, 415: 60, 432: 61, 435: 62, 450: 63, 451: 64, 455: 65, 469: 66, 471: 67, 481: 68, 487: 69,3: 0, 7: 1, 9: 2, 11: 3, 12: 4, 24: 5, 25: 6, 26: 7, 27: 8, 28: 9, 32: 10, 51: 11, 52: 12, 53: 13, 54: 14, 56: 15, 59: 16, 60: 17, 61: 18, 62: 19, 63: 20, 65: 21, 66: 22, 67: 23, 68: 24, 72: 25, 73: 26, 80: 27, 82: 28, 83: 29, 88: 30, 89: 31, 92: 32, 93: 33, 94: 34, 95: 35, 96: 36, 97: 37, 98: 38, 100: 39, 101: 40, 103: 41, 104: 42, 105: 43, 107: 44, 108: 45, 109: 46, 144: 47, 165: 48, 166: 49, 170: 50, 174: 51, 175: 52, 176: 53, 177: 54, 178: 55, 181: 56, 182: 57, 184: 58, 186: 59, 205: 60, 206: 61, 208: 62, 211: 63, 212: 64, 213: 65, 214: 66, 215: 67, 224: 68, 228: 69, 232: 70, 233: 71, 234: 72, 235: 73, 237: 74, 238: 75, 240: 76, 241: 77, 246: 78, 258: 79, 260: 80, 261: 81, 262: 82, 263: 83, 265: 84, 268: 85, 271: 86, 273: 87, 274: 88, 275: 89, 276: 90, 279: 91, 280: 92, 284: 93, 286: 94, 287: 95, 290: 96, 291: 97, 296: 98, 297: 99, 298: 100, 316: 101, 324: 102, 331: 103, 332: 104, 333: 105, 335: 106, 352: 107, 353: 108, 356: 109, 361: 110, 362: 111, 364: 112, 374: 113, 381: 114, 385: 115, 386: 116, 390: 117, 393: 118, 395: 119, 408: 120, 409: 121, 410: 122, 411: 123, 423: 124, 424: 125, 425: 126, 426: 127, 427: 128, 428: 129, 429: 130, 430: 131, 436: 132, 440: 133, 447: 134, 460: 135, 461: 136, 468: 137, 470: 138, 472: 139, 474: 140, 478: 141, 479: 142, 482: 143, 485: 144, 495: 145,8: 0, 10: 1, 29: 2, 30: 3, 39: 4, 40: 5, 41: 6, 42: 7, 44: 8, 45: 9, 81: 10, 84: 11, 86: 12, 87: 13, 90: 14, 91: 15, 106: 16, 114: 17, 115: 18, 117: 19, 123: 20, 125: 21, 128: 22, 133: 23, 140: 24, 141: 25, 142: 26, 143: 27, 179: 28, 185: 29, 187: 30, 190: 31, 195: 32, 196: 33, 197: 34, 199: 35, 200: 36, 201: 37, 209: 38, 210: 39, 219: 40, 220: 41, 221: 42, 222: 43, 242: 44, 243: 45, 248: 46, 250: 47, 251: 48, 253: 49, 257: 50, 266: 51, 285: 52, 289: 53, 295: 54, 307: 55, 320: 56, 321: 57, 322: 58, 334: 59, 340: 60, 341: 61, 342: 62, 351: 63, 354: 64, 355: 65, 357: 66, 358: 67, 359: 68, 360: 69, 365: 70, 366: 71, 375: 72, 391: 73, 392: 74, 414: 75, 422: 76, 431: 77, 433: 78, 434: 79, 438: 80, 439: 81, 441: 82, 442: 83, 443: 84, 444: 85, 445: 86, 446: 87, 448: 88, 456: 89, 457: 90, 466: 91, 467: 92, 477: 93, 484: 94, 491: 95}
class0_reverse = {value: i for i, value in class0.items()}
class1_reverse = {value: i for i, value in class1.items()}
class2_reverse = {value: i for i, value in class2.items()}
class3_reverse = {value: i for i, value in class3.items()}
class4_reverse = {value: i for i, value in class4.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():

    epochs = 1

    batch_size = 1

    #net = ExampleNet().to(device)
    net_fir = CNN_500_part(5).to(device)
    #print(net_fir)
    criterion = nn.MSELoss(reduce = None, size_average = None)
    cross = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(net.parameters())

    test_data, test_label_class, test_label = pre_data(bmname+'_test')
    #test_data1, test_label_class1, test_label1 = pre_data(bmname1 + '_test')

    # idx=[]
    # count=0
    # for i in range(len(test_label)):
    #     if test_jc[i]//100 == test_label[i]//100:
    #         count = count+1
    #         idx.append(i)
    # print(count/len(test_label))
    # print('idx:',idx)    #jc预测第一级准确率只有36.6%
    #test_data, test_label = pre_data('BA300_SI_m5_test')
    test_data = torch.unsqueeze(test_data, dim=1).type(torch.FloatTensor)
    test_dataset = TensorDataset(test_data, test_label_class, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #checkpoint = torch.load('models/' + 'food500_SI_A5_m10' + '_nl1.pth')
    # print(checkpoint)
    ############# first_class_result:
    net_fir.load_state_dict(torch.load('models/' + 'food500_p0.5_m10_p1' + '_p1.pth'))
    # acc_adj_data=[]
    # acc_label=[]
    # acc_label_class=[]
    # with torch.no_grad():
    #     net_fir.eval()
    #     correct = 0.
    #     total = 0.
    #     labels = []
    #     preds = []
    #     for input, target_class, target in test_dataloader:
    #         input, target_class, target = input.to(device), target_class.to(device), target.to(device)
    #         output = net_fir(input)
    #         _, predicted = torch.max(output.data, 1)
    #         total += target_class.size(0)
    #         correct += (predicted == target_class).sum()
    #         accuracy = correct.float() / total
    #         labels.append(target_class.long().numpy())
    #         preds.append(predicted.long().numpy())
    #         if(predicted == target_class):
    #             acc_adj_data.append(input.numpy())
    #             acc_label_class.append(target_class.numpy())
    #             acc_label.append(target.numpy())
    #     # net_fir.train()
    #     print("[classes - {0}]Accuracy:{1}%".format(1, (100 * accuracy)))
    # acc_adj_data = torch.Tensor(acc_adj_data).short()
    # acc_adj_data = torch.squeeze(acc_adj_data, dim=1).type(torch.FloatTensor)
    # # for x in range(len(acc_label)):
    # #     acc_label[x]=acc_label[x]%100
    # acc_label = torch.Tensor(acc_label).long()
    # acc_label = torch.squeeze(acc_label, dim=1).type(torch.FloatTensor)
    # acc_label_class = torch.Tensor(acc_label_class).long()
    # acc_label_class = torch.squeeze(acc_label_class, dim=1).type(torch.FloatTensor)
    # acc_dataset = TensorDataset(acc_adj_data,acc_label_class,acc_label)
    # acc_dataloader = DataLoader(acc_dataset, batch_size=batch_size, shuffle=False)
    ##########pred_dataloader
    pred_adj_data=[]
    pred_label=[]
    pred_label_class=[]   #数据集中，通过第一级得到的预测结果
    pred_real_label_class = [] #真实的分类结果
    with torch.no_grad():
        net_fir.eval()
        correct = 0.
        total = 0.
        labels = []
        preds = []
        for input, target_class, target in test_dataloader:
            input, target_class, target = input.to(device), target_class.to(device), target.to(device)
            output = net_fir(input)
            _, predicted = torch.max(output.data, 1)
            total += target_class.size(0)
            correct += (predicted == target_class).sum()
            accuracy = correct.float() / total
            labels.append(target_class.long().numpy())
            preds.append(predicted.long().numpy())
            pred_adj_data.append(input.numpy())
            pred_label_class.append(predicted.numpy())   ####预测到的分类结果
            pred_real_label_class.append(target_class.numpy())  ####标签对应的真实分类情况
            pred_label.append(target.numpy())            #真实label
        # net_fir.train()
        print("[classes - {0}]Accuracy:{1}%".format(1, (100 * accuracy)))
    pred_adj_data = torch.Tensor(pred_adj_data).short()
    pred_adj_data = torch.squeeze(pred_adj_data, dim=1).type(torch.FloatTensor)
    pred_label = torch.Tensor(pred_label).long()
    pred_label = torch.squeeze(pred_label, dim=1).type(torch.FloatTensor)
    pred_label_class = torch.Tensor(pred_label_class).long()
    pred_label_class = torch.squeeze(pred_label_class, dim=1).type(torch.FloatTensor)
    pred_real_label_class = torch.Tensor(pred_real_label_class).long()
    pred_real_label_class =torch.squeeze(pred_real_label_class, dim=1).type(torch.FloatTensor)
    pred_dataset = TensorDataset(pred_adj_data, pred_label_class, pred_label, pred_real_label_class)
    pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
    model_loadaddr = []
    model_loadaddr.append('models/' + 'food500_p0.5_5k0'+'_m50' + '_n2.pth')
    model_loadaddr.append('models/' + 'food500_p0.5_5k1'+'_m50' + '_n2.pth')
    model_loadaddr.append('models/' + 'food500_p0.5_5k2'+'_m65' + '_n2.pth')
    model_loadaddr.append('models/' + 'food500_p0.5_5k3'+'_m30' + '_n2.pth')
    model_loadaddr.append('models/' + 'food500_p0.5_5k4'+'_m50' + '_n2.pth')
    real_labels = []
    real_preds = []
    for num in range(5):
        a,b=model_sec(pred_dataloader, num, model_loadaddr[num])
        real_labels.extend(a)
        real_preds.extend(b)
    read_dic = np.load("food500_short_path.npy", allow_pickle=True).item()
    distance_pred = []
    count = 0
    for w in range(len(real_labels)):
        a = read_dic[real_labels[w]][real_preds[w]]
        distance_pred.append(a)
        if real_labels[w] == real_preds[w]:
            count = count + 1
    print('perdict accuracy:', count / len(real_labels))
    result_dis = {}
    ave_class = 0
    for q in set(distance_pred):
        result_dis[q] = distance_pred.count(q)
        ave_class = q * result_dis[q] + ave_class
    print('all CNN distance:', ave_class / len(real_labels))
    print("len(real_labels):",len(real_labels))
def model_sec(dataloader,i,model_name):
    model_c = [[], [], [], [], []]
    num_class = [97 , 91 , 70 , 146 , 96]
    class_tonew = [class0, class1, class2, class3, class4]
    class_reverse = [class0_reverse, class1_reverse, class2_reverse, class3_reverse, class4_reverse]
    model_c[i] = CNN_500_part(num_class[i]).to(device)
    model_c[i].load_state_dict(torch.load(model_name))
    print("model_c{0}:".format(i),model_c[i])
    with torch.no_grad():
        model_c[i].eval()
        correct = 0.
        total = 0.
        labels= []
        preds=[]
        labels_index = []
        for input, target_class, target, real_class in dataloader:
            input, target_class, target, real_class = input.to(device), target_class.to(device), target.to(device), \
                                                      real_class.to(device)
            if target_class == i:
                # print(target)
                # print(target.numpy()[0])
                # print(class_tonew[i][target.numpy()[0]])
                #target = class_tonew[i][int(target.numpy()[0])]  #acc no_use
                #target = torch.Tensor([target]).long()
                output = model_c[i](input)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
                accuracy = correct.float() / total
                labels.append(target.long().numpy())
                preds.append(predicted.long().numpy())
                labels_index.append(real_class.long().numpy())
        labels = np.hstack(labels)
        preds = np.hstack(preds)
        labels_index = np.hstack(labels_index)
        print("labels:", labels)
        print("preds:", preds)
        print("[classes2 - {0}]Accuracy:{1}%".format(i, (100 * accuracy)))#
        ##labels 0-99----- 400-499;   preds  0-99--------400-499
        real_labels = []
        # for n in labels:
        #     real_labels.append(class_reverse[i][n])
        for n in range(len(labels)):
            aaa = labels_index[n]
            bbb = labels[n]
            real_labels.append(class_reverse[aaa][bbb])
        real_preds = []
        for m in preds:
            real_preds.append(class_reverse[i][m])
        print('real-labels:', real_labels)
        print('real_preds:', real_preds)
        print("len(real_labels):",len(real_labels))
        # #labels  400-499-----400-499;preds 0-99------400-499
        # real_labels = labels
        # real_preds = []
        # for m in preds:
        #     real_preds.append(class_reverse[i][m])
        # print('reallabels:', real_labels)
        # print('realpreds:', real_preds)
        read_dic = np.load("food500_short_path.npy", allow_pickle=True).item()
        distance_pred = []
        count = 0
        for w in range(len(labels)):
            a = read_dic[real_labels[w]][real_preds[w]]
            distance_pred.append(a)
            if real_labels[w] == real_preds[w]:
                count = count + 1
        print('perdict accuracy:', count / len(labels))
        result_dis = {}
        ave_class = 0
        for q in set(distance_pred):
            result_dis[q] = distance_pred.count(q)
            ave_class = q * result_dis[q] + ave_class
        print('CNN distance:', ave_class / len(labels))
    return real_labels,real_preds
    # losses = []
    # class_reverse = [class0_reverse, class1_reverse, class2_reverse, class3_reverse, class4_reverse]
    # for i in range(epochs):
    #     with torch.no_grad():
    #         correct = 0.
    #         total = 0.
    #         labels = [[], [], [], [], []]
    #         preds = [[], [], [], [], []]
    #         for input, target_class, target in acc_dataloader:
    #             input, target_class, target = input.to(device), target_class.to(device), target
    #             for j in range(len(model_c)):
    #                 if target_class.numpy()==j:
    #                     #print("j;",j)
    #                     model_c[j].eval()
    #                     output = model_c[j](input)
    #                     _, predicted = torch.max(output.data, 1)
    #                     total += target.size(0)
    #                     correct += (predicted == target).sum()
    #                     accuracy = correct.float() / total
    #                     labels[j].append(target.long().numpy())
    #                     preds[j].append(predicted.long().numpy())
    #         #net_fir.train()
    #         print("[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))
    #         reallabels=[[], [], [], [], []]
    #         realpreds_c=[[], [], [], [], []]
    #         for i in range(len(model_c)):
    #             if len(labels[i]) != 0:
    #                 labels[i] = np.hstack(labels[i])
    #                 preds[i] = np.hstack(preds[i])
    #             for m in preds[i]:
    #                 realpreds_c[i].append(class_reverse[i][m])
    #             for n in labels[i]:
    #                 reallabels[i].append(class_reverse[i][n])
    #         reallabels = np.hstack(reallabels)
    #         realpreds_c = np.hstack(realpreds_c)
    #         print("reallabels:",reallabels)
    #         print("realpreds:",realpreds_c)
    #         # real_labels = []
    #         # for n in labels:
    #         #     real_labels.append(class1_reverse[n])
    #         # real_preds = []
    #         # for m in preds:
    #         #     real_preds.append(class1_reverse[m])
    #         read_dic = np.load("food500_short_path.npy", allow_pickle=True).item()
    #         distance_pred = []
    #         count = 0
    #         for w in range(len(reallabels)):
    #             a = read_dic[reallabels[w]][realpreds_c[w]]
    #             distance_pred.append(a)
    #             if reallabels[w] == realpreds_c[w]:
    #                 count = count + 1
    #         print('perdict accuracy:', count / len(reallabels))
    #         result_dis = {}
    #         ave_class = 0
    #         for q in set(distance_pred):
    #             result_dis[q] = distance_pred.count(q)
    #             ave_class = q * result_dis[q] + ave_class
    #         print('GNN distance:', ave_class / len(reallabels))
        #torch.save(net, "models/"+bmname+"_nl.pth")

if __name__ == "__main__":
    main()


