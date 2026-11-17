import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA = {
    "text": [
        "Free entry in 2 a weekly comp to win FA Cup final tickets 21st May 2005",
        "Hey are we still meeting for lunch tomorrow at 2 PM",
        "WINNER as a valued network customer you have been selected to receive a 900 prize reward",
        "Can you please send me the assignment notes by tonight",
        "URGENT your bank account has been locked verify your password now via this link",
        "Hello brother how is everything going with your final year project",
        "Claim your free lottery ticket now click here to win cash",
        "Let us review the software requirements specification document tomorrow",
        "Get a cheap loan with zero percent interest rate right now call quick",
        "Are you coming to the graduation ceremony this weekend",
        "Private 2013 account statement shows 7860 cash callid BASS",
        "Good morning my friend hope you have a wonderful day ahead",
    ],
    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
    ],
}


def create_dataset():
    df = pd.DataFrame(DATA)
    df["label_num"] = df["label"].map({"spam": 1, "ham": 0})
    return df


def train_model():
    df = create_dataset()

    X = df["text"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    return model, vectorizer, X_test_vectorized, y_test


def show_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel trained successfully with TF-IDF + Naive Bayes.")
    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    return accuracy


def predict_messages(model, vectorizer, messages):
    message_vectors = vectorizer.transform(messages)
    predictions = model.predict(message_vectors)

    print("\n--- New Message Predictions ---")
    for message, pred in zip(messages, predictions):
        result = "SPAM 🚨" if pred == 1 else "HAM ✅"
        print(f"Message: '{message}'\nPrediction: {result}\n")

    return predictions


if __name__ == "__main__":
    model, vectorizer, X_test, y_test = train_model()
    show_metrics(model, X_test, y_test)

    sample_messages = [
        "Win a brand new car by clicking this secure link",
        "Can we schedule a meeting for the database project update",
    ]

    predict_messages(model, vectorizer, sample_messages)