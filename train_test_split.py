from sklearn import datasets
from sklearn.model_selection import train_test_split

# 將 Iris 資料集切割為訓練集和測試集
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"], test_size=0.3)

print("X_train = " + str(X_train))
print("X_test = " + str(X_test))
