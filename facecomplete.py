# yêu cầu đặt ra của bài toán là khi chúng ta cho đầu vào là nửa trên của một khuôn mặt, thuật toán sẽ tính toán các tham số và dự đoán ra nửa dưới của khuôn mặt tương ứng.
# Một biểu diễn rât trực quan là chúng ta sẽ có vector X đại diện cho các đại lượng đã biết chính là các nửa trên của ma trận điểm ảnh.
# vector y cần dự đoán chính là nửa dưới của ma trận đó.
# Hai ma trận này có số chiều bằng nhau do đó việc áp dụng các mô hình dự đoán bản chất là tìm ra hàm phụ thuộc y = f(X)
# https://viblo.asia/p/so-sanh-cac-mo-hinh-du-doan-trong-bai-toan-nhan-dang-khuon-mat-va-vi-du-thuc-te-aWeGmgmdKBD

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

# Load the faces datasets
# Thông qua hàm fetch_olivetti_faces() chúng ta sẽ lấy được tập dữ liệu Olivetti với targets từ 0 đến 39. 
# Dây là các chỉ mục được đánh số từ 0 đến 40 tương ứng với 40 người khác nhau trong tập dữ liệu khuôn mặt.
data = fetch_olivetti_faces()
targets = data.target

# Chúng ta sử dụng 10 người đầu tiên để làm tập dữ liệu kiểu trong cho mô hình trong khi sử dụng 30 người còn lại để làm tập dữ liệu huấn luyện.
data = data.images.reshape((len(data.images), -1))
train = data[targets >= 6]
test = data[targets < 6]  # Test on independent people

# Test on a subset of people
# chúng ta chỉ lựa chọn 6 trong 10 đối tượng của tập dữ liệu kiểm tra nhằm dễ dàng cho việc quan sát.
n_faces = 6
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

# Sử dụng các hàm **np.floor **và np.ceil chúng ta sẽ lấy được các phần trên và dưới của hình ảnh dưới dạng ma trận các điểm ảnh.
n_pixels = data.shape[1]
print("n_pixels: {0}. ceil: {1}. floor: {2}".format(n_pixels, np.ceil(0.5 * n_pixels), np.floor(0.5 * n_pixels)) )
X_train = train[:, :int(np.ceil(0.5 * n_pixels))]  # Upper half of the faces
y_train = train[:, int(np.floor(0.5 * n_pixels)):]  # Lower half of the faces
X_test = test[:, :int(np.ceil(0.5 * n_pixels))]
y_test = test[:, int(np.floor(0.5 * n_pixels)):]

# Vậy là đã xong việc xác định tập dữ liệu huấn luyện và tập dữ liệu kiểm tra. 
# Việc còn lại của chúng ta là đi cài đặt các thuật toán trên tập dữ liệu này và đánh giá hiệu quả của từng thuật toán.
# Sau khi có được tập dữ liệu, việc cần làm là xử lý thế nào với tập dữ liệu đó, không thì chúng ta sẽ chẳng thu được kết quả gì cả.
# Fit estimators
# Đầu tiên chúng ta sẽ định nghĩa một object tên là ESTIMATORS chứa tất cả các ESTIMATORS của chúng ta - chính là các phương pháp mà mình đã liệt kê tại phần trên.
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),
    # "Linear regression": LinearRegression(),
    # "Ridge": RidgeCV(),
}

# Sau đó chúng ta sử dụng một vòng for để áp dụng các mô hình cho tập dữ liệu huấn luyện, đồng thời kiểm tra luôn trên tập dữ liệu kiểm tra và lưu vào y_test_predict
# y_test_predict lúc này sẽ lưu lại kết quả của từng thuật toán và ta sẽ sử dụng chúng trong việc vẽ đồ thị biểu diễn kết quả.
y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# Vẽ đồ thị biểu diễn kết quả
# Chúng ta hãy tưởng tượng với một một khuôn mặt được dự doán bằng 4 thuật toán khác nhau nên chúng ta sẽ cần phải vẽ 5 hình cho mỗi khuôn mặt 
# nhằm so sánh kết quả của 4 hình tìm ra bởi 4 thuật toán và 1 hình gốc ban đầu. Mỗi hình có kích cỡ là 64 x 64 pixels.
# Plot the completed faces
image_shape = (64, 64)
n_cols = 1 + len(ESTIMATORS)

# Sau khi đã có 5 cột biểu diễn 5 bức ảnh chúng ta thực hiện vẽ các ảnh tương ứng với dữ liệu vừa dự đoán được
for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                          title="true faces")


    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

plt.show()
# Một cách trực quan chúng ta có thể thấy rằng các thuật toán Hồi quy tuyến tính , Cây quyết định, Hồi quy Ridge và K Nearest Neighbor lần lượt cho kết quả dự đoán nửa dưới của khuôn mặt khác nhau. 
# Có thể thấy rằng Hồi quy tuyến tính cho độ mượt kém nhất so với các phương pháp còn lại. 
# Cá nhân mình nghĩ rằng Cây quyết định hoặc Hồi quy Ridge thích hợp cho bài toán nhận dạng này