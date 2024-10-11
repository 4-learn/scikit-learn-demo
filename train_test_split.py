from sklearn.model_selection import train_test_split

data = [n for n in range(1, 11)]
print(data)

train_data, test_data = train_test_split(data, random_state=777, train_size=0.8)
print(train_data)
print(test_data)
