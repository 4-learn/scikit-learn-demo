from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

def main():
    # 1. 載入 Iris 資料集
    iris = datasets.load_iris()

    X = iris.data      # 特徵 (Features)
    y = iris.target    # 標籤 (Labels)

    print("Original dataset")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Overall label distribution:", Counter(y))
    print("-" * 50)

    # 2. 切割資料集（70% train / 30% test）
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # 3. 檢查 shape
    print("After train_test_split")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)
    print("-" * 50)

    # 4. 檢查類別分佈
    print("Training label distribution:", Counter(y_train))
    print("Test label distribution    :", Counter(y_test))
    print("-" * 50)

    # 5. 最低限度的 sanity check（概念用）
    assert X_train.shape[0] == y_train.shape[0], "Train X/y size mismatch"
    assert X_test.shape[0] == y_test.shape[0], "Test X/y size mismatch"

    print("✅ Data split looks reasonable.")

if __name__ == "__main__":
    main()

