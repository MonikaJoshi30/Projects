from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Continue in the same file
def load_and_split_data() -> tuple:
    """
    Returns:
        tuple: [X_train, X_test, y_train, y_test]
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# Continue in the same file
def create_and_train_SVM(X_train: list, y_train: list) -> SVC:
    """
    Args:
        X_train: [features for training]
        y_train: [labels for training]

    Returns:
        SVC: [Trained Support Vector Classifier model]
    """
    svm = SVC()
    svm.fit(X_train, y_train)
    return svm

def make_predictions(model: SVC, X_test: list) -> list:
    """
    Args:
        model: [Trained Support Vector Classifier model]
        x_test: [features for testing]
        
    Returns:
        list: [Predictions]
    """
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()

    svm_model = create_and_train_SVM(X_train, y_train)

    predictions = make_predictions(svm_model, X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy : {accuracy: .2f}")


    print("Classification Report : ")
    print(classification_report(y_test, predictions))