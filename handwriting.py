# https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format
# import các thư viện cần thiết
import numpy as np
import csv, random
import matplotlib
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical

# đọc dữ liệu
with open('A-Z-Handwritten-Data.csv', 'r') as csv_file:
    result = csv.reader(csv_file)
    rows = []
    
    # đọc từng dòng của file và thêm vào list rows, mỗi phần tử của list là một dòng
    for row in result:
        rows.append(row)
        # Mỗi dòng đại diện cho một ảnh có kích thước 28*28, ký tự đầu tiên đại diện cho class mà ảnh đó thuộc về.
        # VD: A là class 0, B là class 1
        # Do đó độ dài của list đại diện cho mỗi dòng là 28*28 + 1 = 785.
        # Việc load dữ liệu có thể sử dụng rất nhiều RAM ~6GB
    # Chúng ta hãy thử in ra một vài ảnh xem điều mà mình đã nói ở trên có đúng hay không:
    # letter = rows[30000]
    # letter = ['2', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '60', '162', '255', '255', '255', '255', '255', '255', '224', '100', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '46', '77', '233', '255', '255', '255', '236', '246', '255', '255', '255', '255', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '255', '255', '255', '255', '255', '167', '65', '116', '255', '255', '255', '255', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '19', '185', '255', '255', '255', '255', '150', '3', '0', '32', '218', '255', '236', '162', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '15', '175', '255', '255', '255', '255', '155', '12', '0', '0', '0', '37', '62', '49', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '83', '185', '255', '255', '255', '227', '116', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '17', '187', '255', '255', '255', '221', '136', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '12', '168', '255', '255', '255', '236', '42', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '102', '255', '255', '255', '229', '49', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '241', '255', '255', '224', '49', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '255', '255', '255', '153', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '255', '255', '255', '153', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '255', '255', '255', '181', '14', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '153', '255', '255', '255', '51', '0', '0', '0', '0', '0', '0', '0', '0', '3', '15', '9', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '102', '255', '255', '255', '121', '0', '0', '0', '0', '0', '0', '0', '46', '144', '255', '199', '23', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '15', '212', '255', '255', '247', '124', '62', '12', '0', '0', '12', '62', '232', '255', '255', '255', '181', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '107', '255', '255', '255', '255', '255', '181', '162', '162', '181', '255', '255', '255', '255', '252', '105', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '29', '212', '255', '255', '255', '255', '255', '255', '255', '255', '255', '255', '255', '190', '88', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '121', '223', '255', '255', '255', '255', '255', '255', '255', '255', '255', '88', '19', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '40', '131', '255', '255', '255', '255', '255', '255', '162', '100', '20', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
    # x = np.array([int(j) for j in letter[1:]])
    # x = x.reshape(28, 28)

    # print(letter)
    # result: ['2', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
    # Ký tự đầu tiên của list là 2, vậy ký tự mà ta mong chờ thu được là ký tự C. 
    # plt.imshow(x)
    # plt.show()
    
    # Để cho đơn giản và có thể ứng dụng mô hình thu được cho bài toán chấm điểm trắc nghiệm, mình sẽ chỉ xây dựng mô hình cho bốn ký tự đầu tiên của bảng chữ cái là A, B, C, D. 
    # Cách xử lý với toàn bộ 26 chữ cái có thể được thực hiện hoàn toàn tương tự.
    # Trong đoạn code trên, tôi đã lấy ra các ký tự A, B, C, D có trong tập dữ liệu của mình và lưu vào train_data và train_label.
    # Mỗi phần tử trong train_data là một mảng 28 x 28, đại diện cho một ảnh của một ký tự.
    # Label tương ứng của ký tự đó là một số được lưu trong train_label
    
    train_data = [] # dữ liệu training
    train_label = [] # label của chúng

    for letter in rows:
        if (letter[0] == '0') or (letter[0] == '1') or (letter[0] == '2') or (letter[0] == '3'):
            x = np.array([int(j) for j in letter[1:]])
            x = x.reshape(28, 28)
            train_data.append(x)
            train_label.append(int(letter[0]))
        else:
            break
    # Shuffle
    shuffle_order = list(range(56081))
    random.shuffle(shuffle_order)

    train_data = np.array(train_data)
    train_label = np.array(train_label)

    train_data = train_data[shuffle_order]
    train_label = train_label[shuffle_order]
    # Để áp dụng cho bài toán Machine Learning, chúng ta chia tập dữ liệu của mình thành ba tập riêng biệt: training set, test set và validation set
    # Trong đó x là dữ liệu đầu vào (input), y là label tương ứng (output).
    print(train_data.shape)
    train_x = train_data[:50000]
    train_y = train_label[:50000]

    val_x = train_data[50000:53000]
    val_y = train_label[50000:53000]

    test_x = train_data[53000:]
    test_y = train_label[53000:]
    
    # Xây dựng mô hình
    # Trong bài viết này, chúng ta sẽ sử dụng thư viện TFLearn cho việc xâu dựng mô hình học máy.
    # Khởi tạo các giá trị hằng số được sử dụng trong mô hình
    # BATCH_SIZE: kích thước mỗi batch dữ liệu truyền vào
    # IMG_SIZE: kích thước mỗi chiều của hình ảnh đầu vào
    # N_CLASSES: số lượng classes mà chúng ta cần huấn luyện (training)
    # LR = 0.001: tốc độ học (learning rate)
    # N_EPOCHS = 50: số lượng epoch mà chúng ta training
    BATCH_SIZE = 32
    IMG_SIZE = 28
    N_CLASSES = 4
    LR = 0.001
    N_EPOCHS = 5
    # Mô hình mà chúng ta sử dụng ở đây bao gồm 6 lớp Convolutional layer và 2 lớp Fully Connected Layer nối tiếp nhau.
    tf.reset_default_graph()

    #1: kích thước dữ liệu đầu vào là [None, IMG_SIZE, IMG_SIZE, 1]
    # None đại diện cho BATCH_SIZE
    # IMG_SIZE là kích thước mỗi chiều của ảnh
    # 1 là số dải màu của ảnh, do chúng ta sử dụng ảnh đen trắng nên chỉ có 1 dải màu, nếu chúng ta sử dụng ảnh màu thì số dải màu mà chúng ta sử dụng là 3, đại diện cho 3 dải màu RGB.
    network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1]) #1

    # Convolutional layer
    # 32: số lượng filters
    # 3: filter size 3x3
    # Bước nhảy(stride) được mặc định là 1
    # Activation function: ReLU
    network = conv_2d(network, 32, 3, activation='relu') #2
    #3: Maxpool layer
    # 2: kernel size
    network = max_pool_2d(network, 2) #3

    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    
    #4: Fully-connected layer
    # 1024: số lượng neuron
    # Activation function: ReLU
    network = fully_connected(network, 1024, activation='relu') #4
    network = dropout(network, 0.8) #5: Dropout 80%

    network = fully_connected(network, N_CLASSES, activation='softmax')#6: Fully-connected layer đại điện cho đầu ra (output). N_CLASSES: số output đầu ra. 
    # Activation function: softmax (để tổng xác suất đầu ra bằng 1)
    network = regression(network)

    model = tflearn.DNN(network) #7: #7: Xây dựng mô hình
    # Để dữ liệu đầu vào được trùng khớp với mô hình đã xây dưng, chúng ta cần phải đưa dữ liệu về định dạng phù hợp như sau
    train_x = train_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    val_x = val_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_x = test_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # Tương tự với label, đưa label về dạng onehot vector:
    original_test_y = test_y # được sử dụng để test ở bước sau
    train_y = to_categorical(train_y, N_CLASSES)
    val_y = to_categorical(val_y, N_CLASSES)
    test_y = to_categorical(test_y, N_CLASSES)
    # Bây giờ chúng ta cùng training:
    model.fit(train_x, train_y, n_epoch=N_EPOCHS, validation_set=(val_x, val_y), show_metric=True)
    # Chúng ta hãy lưu lại model đã train được như sau
    model.save('mymodel.tflearn')

    