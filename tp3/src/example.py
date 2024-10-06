from sklearn.model_selection import KFold

def main():
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]):
        print(train_index.size, test_index.size)
        print(train_index, test_index)


if __name__ == "__main__":
    main()
