# main file. Used this to test spam fil

from filter_test import FilterTest


def enron6():
    """
    Enron 6 Test
    Trains Naive Bayes Classifier on Enron 6 Dataset.
    Link to dataset: https://www2.aueb.gr/users/ion/data/enron-spam/

    Results:
    Precision: 0.996
    Recall: 0.594
    """
    spam_folder = "" # Insert Folder for Enron 6 spam
    ham_folder = "" # Insert Folder for Enron 6 ham
    my_test = FilterTest(spam_folder, ham_folder)
    # call functions to train and test data
    my_test.extract_data()
    my_test.train_test_split()
    my_test.train_filter()
    my_test.test_filter()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    enron6()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
